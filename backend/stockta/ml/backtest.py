"""歷史回測（模擬部署）：python -m stockta.ml.backtest

以 production 模型對測試期（SPLIT_VAL_END 之後，預設 2024-01 起）每個交易日
逐日推論，與實際 5 日趨勢標籤對照，產出 docs/backtest_report.md。
同時對照「原始 argmax」與「信心門檻校準後」（正式決策規則，與線上 API 一致）。

與線上實證（predictions.db / online_predictions.md）嚴格分開：
本報告是「事後對歷史資料重演」的回測——模型權重僅見過 SPLIT_TRAIN_END 以前
的資料、SPLIT_VAL_END 以前用於 early stopping、選型與門檻校準，回測期間對模型
完全樣本外，但仍不等於預測當下即時落地的線上紀錄，報告內文須明確標示。
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stockta.config import (
    AUTO_ADJUST,
    BACKEND_ROOT,
    DATA_CACHE_DIR,
    LABEL_CLASSES,
    LABEL_DOWN_THRESHOLD,
    LABEL_HORIZON_DAYS,
    LABEL_UP_THRESHOLD,
    MARKET_INDEX_TICKER,
    PRODUCTION_MODEL,
    SIGNAL_CONFIDENCE_THRESHOLDS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.features.pipeline import build_features
from stockta.inference.predictor import Predictor
from stockta.ml.labeling import make_labels

DOCS_DIR = BACKEND_ROOT.parent / "docs"


def collect_probas(
    predictor: Predictor, ohlcv: pd.DataFrame, context: pd.DataFrame, start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (機率 (n,3), 實際類別, 視窗終點日)；無有效樣本回空陣列。"""
    feats = build_features(ohlcv, context)
    labels = make_labels(ohlcv["close"]).reindex(feats.index)
    if len(feats) < WINDOW_LENGTH_DAYS:
        return np.empty((0, 3)), np.empty(0), np.empty(0)

    scaled = predictor._scaler.transform(feats.to_numpy(dtype=np.float64)).astype(np.float32)
    windows = sliding_window_view(scaled, WINDOW_LENGTH_DAYS, axis=0)
    windows = windows.transpose(0, 2, 1).reshape(windows.shape[0], -1)

    end_dates = feats.index[WINDOW_LENGTH_DAYS - 1 :]
    y = labels.iloc[WINDOW_LENGTH_DAYS - 1 :].to_numpy(dtype=np.float64)
    mask = np.asarray((end_dates > start) & ~np.isnan(y))
    if not mask.any():
        return np.empty((0, 3)), np.empty(0), np.empty(0)

    proba = predictor._model.predict_proba(windows[mask])
    return proba, y[mask].astype(int), end_dates[mask].values


def apply_thresholds_batch(proba: np.ndarray) -> np.ndarray:
    """resolve_signal 的向量化版本：對 (n,3) 機率套用正式決策規則。"""
    pred = proba.argmax(axis=1)
    hold = LABEL_CLASSES.index("觀望")
    for name, tau in SIGNAL_CONFIDENCE_THRESHOLDS.items():
        k = LABEL_CLASSES.index(name)
        pred[(pred == k) & (proba[:, k] < tau)] = hold
    return pred


def signal_table(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    lines = [
        "| 模型預測 | 筆數 | 實際=跌 | 實際=觀望 | 實際=漲 | 命中率 |",
        "|---|---|---|---|---|---|",
    ]
    for k, name in enumerate(LABEL_CLASSES):
        m = y_pred == k
        if not m.any():
            lines.append(f"| {name} | 0 | - | - | - | - |")
            continue
        counts = np.bincount(y_true[m], minlength=3)
        lines.append(
            f"| {name} | {int(m.sum()):,} | "
            + " | ".join(f"{c / m.sum():.1%}" for c in counts)
            + f" | {counts[k] / m.sum():.1%} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="歷史回測（模擬部署）")
    parser.add_argument(
        "--start", default=SPLIT_VAL_END, help=f"回測起日（不含），預設 {SPLIT_VAL_END}（測試期起點）"
    )
    args = parser.parse_args()
    start = pd.Timestamp(args.start)

    provider = YFinanceProvider(cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST)
    predictor = Predictor.from_registry(PRODUCTION_MODEL)
    end = last_completed_trading_day()
    fetch_start = start.date() - timedelta(days=730)  # 特徵暖機 + 視窗所需的前置歷史
    print(f"模型：{predictor.version}；回測期間 {args.start} 之後 ~ {end}")

    pool_ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in STOCK_POOL:
        try:
            pool_ohlcv[ticker] = provider.get_ohlcv(ticker, fetch_start, end)
        except DataProviderError as exc:
            print(f"[略過] {ticker}: {exc}")
    market = provider.get_ohlcv(MARKET_INDEX_TICKER, fetch_start, end)
    context = build_market_context(market, pool_ohlcv)

    probas: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    for ticker, df in pool_ohlcv.items():
        p, a, d = collect_probas(predictor, df, context, start)
        if len(p):
            probas.append(p)
            actuals.append(a)
            dates.append(d)
        print(f"  {ticker} {STOCK_POOL.get(ticker, '')}: {len(p)} 筆")

    proba = np.concatenate(probas)
    y_true = np.concatenate(actuals)
    end_dt = pd.DatetimeIndex(np.concatenate(dates))
    pred_raw = proba.argmax(axis=1)
    pred_cal = apply_thresholds_batch(proba)
    n = len(y_true)
    hit_raw = float((pred_raw == y_true).mean())
    hit_cal = float((pred_cal == y_true).mean())
    base_rates = np.bincount(y_true, minlength=3) / n

    thresholds_desc = "、".join(
        f"{name}≥{tau:.2f}" for name, tau in SIGNAL_CONFIDENCE_THRESHOLDS.items()
    )
    lines = [
        "# 歷史回測報告（模擬部署）",
        "",
        f"產出日：{date.today().isoformat()}；模型 `{predictor.version}`；"
        f"回測期間 {args.start} 之後 ~ {end}（{end_dt.min().date()} 起實際有樣本）。",
        "",
        f"標籤定義與訓練一致：基準日後 {LABEL_HORIZON_DAYS} 個交易日累積報酬"
        f" > {LABEL_UP_THRESHOLD:+.0%} 為「漲」、< {LABEL_DOWN_THRESHOLD:+.0%} 為「跌」，其餘「觀望」。",
        "",
        "> **性質聲明**：本報告為對歷史資料的事後回測（walk-forward 模擬部署），"
        f"非線上即時預測紀錄（後者見 online_predictions.md）。模型權重僅以 {SPLIT_TRAIN_END} 以前"
        f"的資料訓練、{SPLIT_VAL_END} 以前用於 early stopping、選型與信心門檻校準，"
        "故回測期間對模型為完全樣本外（out-of-sample）。",
        "",
        f"## 總覽：{n:,} 筆；校準後整體命中率 {hit_cal:.1%}"
        f"（原始 argmax {hit_raw:.1%}；隨機基線 33.3%、多數類基線 {base_rates.max():.1%}）",
        "",
        "實際類別分佈（基率）："
        + "、".join(f"{name} {r:.1%}" for name, r in zip(LABEL_CLASSES, base_rates))
        + "——各訊號命中率應對照對應基率解讀（高於基率即代表模型帶有可用訊號）。",
        "",
        f"## 正式決策規則（信心門檻 {thresholds_desc}，與線上 API 一致）",
        "",
        "模型只在信心足夠時給方向訊號，其餘一律「觀望」——少喊、喊得準：",
        "",
        *signal_table(y_true, pred_cal),
        "",
        "## 對照：原始 argmax（無門檻）",
        "",
        *signal_table(y_true, pred_raw),
        "",
        "## 按年度",
        "",
        "| 年度 | 筆數 | 原始 argmax | 校準後 |",
        "|---|---|---|---|",
    ]
    for year in sorted(set(end_dt.year)):
        m = end_dt.year == year
        lines.append(
            f"| {year} | {int(m.sum()):,} | {(pred_raw[m] == y_true[m]).mean():.1%} "
            f"| {(pred_cal[m] == y_true[m]).mean():.1%} |"
        )

    lines += [
        "",
        "## 優化歷程",
        "",
        "- **信心門檻校準（已採用）**：argmax 直接當訊號會過度偏「跌」；門檻僅以驗證期",
        "  網格搜尋選出、測試期驗證，是本報告「正式決策規則」表格的來源。",
        "- **walk-forward 重訓（已採用，2026-07-17）**：首輪模型（train≤2022）在 2026 年",
        "  命中率明顯衰退（校準後 43.5%）；將切分前移兩年（train≤2024、val=2025）重訓並",
        "  重新選型/校準後，2026 年校準後命中率提升至 45.5%。舊 artifact 備份於",
        "  backend/artifacts_backup_2022split/。",
        "- **五模型機率平均集成（試過，不採用）**：以首輪模型於 2024–2026 評測，最佳組合",
        "  （lstm+gru+tcn）僅比單一模型高 0.3 個百分點，卻犧牲喊漲精度且 API 需載入多個",
        "  模型。",
        "- 未實作的方向：triple-barrier 標籤與 meta-labeling、依漂移偵測自動觸發重訓。",
        "",
        "> 由 `python -m stockta.ml.backtest` 產生；門檻校準流程見 `python -m stockta.ml.calibrate`。",
        "> 與 docs/model_comparison.md 的測試集指標同一資料範圍，本報告以「逐日訊號 vs 實際」"
        "的角度呈現，便於解讀。",
    ]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "backtest_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{n:,} 筆；校準後 {hit_cal:.1%}（原始 {hit_raw:.1%}），報告已寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
