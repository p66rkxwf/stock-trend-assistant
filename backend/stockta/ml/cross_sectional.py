"""Cross-sectional 相對強弱排序（experiment_log #8）。

與 3 類絕對方向管線並存、互不干擾。標籤＝未來 LABEL_HORIZON_DAYS 日報酬是否贏過
「當日全池中位數」（二分類，設計上約 50%）；模型分數＝P(贏過中位數)，用於全池排序。
模型用 sklearn RF/XGB（自動二分類，不動 3 類模型的 N_CLASSES）。

用法（backend/ 下）:
    python -m stockta.ml.cross_sectional train --model rf     # 訓練並存 artifacts_cs/rf/
    python -m stockta.ml.cross_sectional train --model xgb
    python -m stockta.ml.cross_sectional report                # 選型(驗證期 Rank IC)+產報告

無前視：特徵只用 ≤t 的資料（build_features）；中位數是標籤側的未來量，屬合法標籤計算。
選型/持有期只用驗證期(2025)決定，測試期(2026)僅最終驗證。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from stockta.config import (
    ARTIFACTS_CS_DIR,
    BACKEND_ROOT,
    CS_COST_BPS,
    CS_HOLDING_DAYS_CANDIDATES,
    CS_TOP_FRACTION,
    LABEL_HORIZON_DAYS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    WINDOW_LENGTH_DAYS,
)
from stockta.features.pipeline import FEATURE_COLUMNS, build_features
from stockta.ml.cs_metrics import portfolio_backtest, quantile_spread, rank_ic
from stockta.ml.train import load_market_context, load_pool_ohlcv

DOCS_DIR = BACKEND_ROOT.parent / "docs"
_HORIZONS = tuple(sorted(set((LABEL_HORIZON_DAYS,)) | set(CS_HOLDING_DAYS_CANDIDATES)))


@dataclass
class CSData:
    X: np.ndarray
    y: np.ndarray  # 1 = 贏過當日中位數
    dates: np.ndarray
    tickers: np.ndarray
    ret_label: np.ndarray  # LABEL_HORIZON_DAYS 日報酬（Rank IC 用）
    rets_by_h: dict[int, np.ndarray]  # 各持有期的未來報酬（組合回測用）
    is_train: np.ndarray
    is_val: np.ndarray
    is_test: np.ndarray


def build_cs_dataset(ohlcv_by_ticker: dict[str, pd.DataFrame], context: pd.DataFrame) -> CSData:
    train_end = pd.Timestamp(SPLIT_TRAIN_END)
    val_end = pd.Timestamp(SPLIT_VAL_END)

    Xs, dates, tkrs, ret_label = [], [], [], []
    rets_h: dict[int, list] = {h: [] for h in _HORIZONS}
    for t, ohlcv in ohlcv_by_ticker.items():
        feats = build_features(ohlcv, context)
        if len(feats) < WINDOW_LENGTH_DAYS:
            continue
        arr = feats.to_numpy(dtype=np.float32)
        w = sliding_window_view(arr, WINDOW_LENGTH_DAYS, axis=0).transpose(0, 2, 1)
        w = w.reshape(w.shape[0], -1)
        end_dates = feats.index[WINDOW_LENGTH_DAYS - 1:]
        close = ohlcv["close"].astype("float64").reindex(feats.index).to_numpy()
        pos = np.arange(WINDOW_LENGTH_DAYS - 1, len(feats))

        # 標籤 horizon 的報酬決定樣本有效性；各持有期報酬缺資料處為 NaN（回測時該期跳過）
        lab_fwd = pos + LABEL_HORIZON_DAYS
        ok = lab_fwd < len(close)
        if not ok.any():
            continue
        Xs.append(w[ok]); dates.append(end_dates[ok].values); tkrs.append(np.array([t] * ok.sum()))
        ret_label.append(close[lab_fwd[ok]] / close[pos[ok]] - 1)
        for h in _HORIZONS:
            fwd = pos[ok] + h
            r = np.full(ok.sum(), np.nan)
            v = fwd < len(close)
            r[v] = close[fwd[v]] / close[pos[ok][v]] - 1
            rets_h[h].append(r)

    X = np.concatenate(Xs)
    dts = pd.DatetimeIndex(np.concatenate(dates))
    tkr = np.concatenate(tkrs)
    rl = np.concatenate(ret_label)
    rets_by_h = {h: np.concatenate(rets_h[h]) for h in _HORIZONS}

    # cross-sectional 標籤：當日報酬 > 當日中位數（用全部股票算中位數，非子取樣）
    med = pd.DataFrame({"r": rl, "d": dts}).groupby("d")["r"].transform("median").to_numpy()
    y = (rl > med).astype(int)

    is_train = np.asarray(dts <= train_end)
    is_val = np.asarray((dts > train_end) & (dts <= val_end))
    is_test = np.asarray(dts > val_end)
    return CSData(X, y, dts.values, tkr, rl, rets_by_h, is_train, is_val, is_test)


def _make_model(name: str):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=42
        )
    raise SystemExit(f"未知模型 {name!r}（rf|xgb）")


def _artifact_dir(name: str):
    d = ARTIFACTS_CS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def train(model_name: str, stride: int = 2) -> None:
    ohlcv = load_pool_ohlcv()
    context = load_market_context(ohlcv)
    ds = build_cs_dataset(ohlcv, context)
    tr = np.where(ds.is_train)[0][::stride]  # 訓練集子取樣（中位數已用全體算）
    scaler = StandardScaler().fit(ds.X[tr])
    model = _make_model(model_name)
    print(f"訓練 cross-sectional {model_name}：train {len(tr):,}（stride {stride}）…")
    model.fit(scaler.transform(ds.X[tr]).astype(np.float32), ds.y[tr])

    # 驗證期 Rank IC（選型依據）
    va = ds.is_val
    val_score = model.predict_proba(scaler.transform(ds.X[va]).astype(np.float32))[:, 1]
    ic = rank_ic(ds.dates[va], val_score, ds.ret_label[va])

    d = _artifact_dir(model_name)
    joblib.dump({"model": model, "scaler": scaler}, d / "model.joblib")
    meta = {
        "model_name": model_name,
        "kind": "cross_sectional",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
        "window_length_days": WINDOW_LENGTH_DAYS,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "val_rank_ic": ic["mean"],
        "val_rank_ic_t": ic["t"],
        "n_train": len(tr),
    }
    (d / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  驗證期 Rank IC {ic['mean']:+.4f}（t={ic['t']:.1f}）；已存 {d}")


def load_cs_model(name: str):
    d = ARTIFACTS_CS_DIR / name
    bundle = joblib.load(d / "model.joblib")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return bundle["model"], bundle["scaler"], meta


def _score_split(model, scaler, ds: CSData, mask: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(ds.X[mask]).astype(np.float32))[:, 1]


def report() -> None:
    """選型（驗證期 Rank IC）+ 對測試期產出 docs/cross_sectional_report.md。"""
    ohlcv = load_pool_ohlcv()
    context = load_market_context(ohlcv)
    ds = build_cs_dataset(ohlcv, context)

    # 選型：載入所有已訓練 CS 模型，比驗證期 Rank IC
    candidates = [p.name for p in ARTIFACTS_CS_DIR.iterdir() if (p / "model.joblib").exists()] \
        if ARTIFACTS_CS_DIR.exists() else []
    if not candidates:
        raise SystemExit("尚無 CS 模型，請先 python -m stockta.ml.cross_sectional train --model rf")
    scored = {}
    for name in candidates:
        model, scaler, _ = load_cs_model(name)
        vs = _score_split(model, scaler, ds, ds.is_val)
        scored[name] = (model, scaler, rank_ic(ds.dates[ds.is_val], vs, ds.ret_label[ds.is_val]))
    best = max(scored, key=lambda n: scored[n][2]["mean"])
    model, scaler, val_ic = scored[best]
    print(f"選型（驗證期 Rank IC）：{best}（{val_ic['mean']:+.4f}）")

    # 測試期指標
    te = ds.is_test
    ts = _score_split(model, scaler, ds, te)
    test_ic = rank_ic(ds.dates[te], ts, ds.ret_label[te])
    spread = quantile_spread(ds.dates[te], ts, ds.ret_label[te], CS_TOP_FRACTION)

    # 持有期掃描：以「驗證期(2025)淨報酬」選最佳持有期
    val_s = _score_split(model, scaler, ds, ds.is_val)
    best_h, best_val_net = CS_HOLDING_DAYS_CANDIDATES[0], -1e9
    for h in CS_HOLDING_DAYS_CANDIDATES:
        vb = portfolio_backtest(ds.dates[ds.is_val], ds.tickers[ds.is_val], val_s,
                                ds.rets_by_h[h][ds.is_val], h, CS_TOP_FRACTION, CS_COST_BPS)
        if vb.get("periods") and vb["net_cum"] > best_val_net:
            best_val_net, best_h = vb["net_cum"], h
    # 組合報酬為示意圖：跑完整樣本外(2025+2026)較有意義（持有期已於 2025 選定）
    oos = ds.is_val | ds.is_test
    oos_s = _score_split(model, scaler, ds, oos)
    bt = portfolio_backtest(ds.dates[oos], ds.tickers[oos], oos_s,
                            ds.rets_by_h[best_h][oos], best_h, CS_TOP_FRACTION, CS_COST_BPS)

    _write_report(best, val_ic, test_ic, spread, best_h, bt)
    print(f"測試期 Rank IC {test_ic['mean']:+.4f}（t={test_ic['t']:.1f}）；報告已寫入 "
          f"{DOCS_DIR / 'cross_sectional_report.md'}")


def _write_report(name, val_ic, test_ic, spread, hold, bt) -> None:
    from datetime import date

    yr = "、".join(f"{y} {v:+.4f}" for y, v in test_ic["by_year"].items())
    lines = [
        "# Cross-sectional 相對強弱排序報告（experiment_log #8）",
        "",
        f"產出日：{date.today().isoformat()}；模型 `{name}`（cross-sectional，"
        f"以驗證期 Rank IC 選出）。標的：台灣 50 全池。",
        "",
        f"標籤：未來 {LABEL_HORIZON_DAYS} 個交易日報酬**是否贏過當日全池中位數**（相對強弱，"
        "非絕對漲跌）；模型分數＝P(贏過中位數)，用於全池排序。",
        "",
        "> **性質**：point-in-time 樣本外（權重僅訓練到 "
        f"{SPLIT_TRAIN_END}、選型只用驗證期 {SPLIT_VAL_END}，測試期僅驗證）。",
        "",
        "## Rank IC（選股技能的標準指標）",
        "",
        "| 期間 | 平均 Rank IC | t 值 | 有效天數 |",
        "|---|---|---|---|",
        f"| 驗證期(2025) | {val_ic['mean']:+.4f} | {val_ic['t']:.1f} | {val_ic['n_days']} |",
        f"| 測試期(2026) | {test_ic['mean']:+.4f} | {test_ic['t']:.1f} | {test_ic['n_days']} |",
        "",
        f"測試期分年：{yr}。判準：**IC 0.02~0.05 可用、0.05~0.10 好、0.10 頂尖**；"
        "t>2 即統計顯著。",
        "",
        f"前/後 {int(CS_TOP_FRACTION*100)}% 多空價差（gross，每 {LABEL_HORIZON_DAYS} 日）："
        f"{spread:+.2%}。",
        "",
        f"## 組合回測（long-only 前 {int(CS_TOP_FRACTION*100)}%，每 {hold} 交易日換股，vs 等權全池）",
        "",
        "樣本外 2025-2026（持有期以 2025 淨報酬選定；為示意圖，非嚴謹績效）。",
        "",
    ]
    if bt.get("periods"):
        lines += [
            f"- 持有期 {hold} 日；{bt['periods']} 期、約 {bt['years']:.1f} 年",
            f"- **扣成本後策略累積 {bt['net_cum']:+.1%}（年化 {bt['net_ann']:+.1%}）**",
            f"- 基準（等權全池）累積 {bt['bench_cum']:+.1%}（年化 {bt['bench_ann']:+.1%}）",
            f"- **扣成本後超額 {bt['net_excess_cum']:+.1%}**；gross 累積 {bt['gross_cum']:+.1%}",
            f"- 平均換手 {bt['avg_turnover']:.0%}／期（成本 {CS_COST_BPS:.0f}bp 來回）；"
            f"每期贏基準 {bt['win_rate']:.0%}",
            "",
        ]
    lines += [
        "## 誠實警告（勿誇大）",
        "",
        "- **市場 beta**：這段期間台股強多頭，基準本身漲很多；策略的絕對報酬多為市場給的，"
        "真正屬於模型的是**超額**與 **Rank IC**。",
        "- **成本估計**：已扣約 " + f"{CS_COST_BPS:.0f}bp 來回，但未計滑價、借券成本、放空限制；"
        "多空價差為 gross，僅概念參考。",
        "- **樣本短**：測試期約 1.5 年、49 檔大型股；模型多半學到**動量**，反轉市場會回吐。",
        "- Rank IC 是主證據（不受多頭影響）；組合報酬為示意，非投資建議。",
        "",
        "> 由 `python -m stockta.ml.cross_sectional report` 產生；與 3 類絕對方向管線"
        "（docs/backtest_report.md）並存對照——絕對方向難、相對排序可行。",
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "cross_sectional_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="cross-sectional 相對強弱排序")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--model", required=True, choices=["rf", "xgb"])
    p_tr.add_argument("--stride", type=int, default=2)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.model, args.stride)
    else:
        report()


if __name__ == "__main__":
    main()
