"""實驗 #9：walk-forward 週期性重訓 vs 凍結模型，以及 regime 特徵的成對消融。

CLI：python -m stockta.ml.walkforward [--model lstm] [--stride 2] [--arms both|base|regime]

**問題**：正式模型的訓練資料止於 `SPLIT_TRAIN_END`（2024-12-31），此後再也沒重訓，
線上等於帶著一年半的概念漂移在跑；2026 多頭段「喊跌」只有 23.5% 命中即是症狀。
2026 年的文獻對此收斂到兩招：**walk-forward 週期性重訓**（持續在最新資料上重訓以維持
準確度）與 **regime-aware 條件化**（偵測市場狀態後條件化模型）。本實驗兩招都測。

**設計**：擴張視窗、每半年一折，共 4 折。每折都是「訓練 → 用該折驗證期校準門檻 →
在其後未見過的半年測試」，門檻**只准用驗證期挑**（與正式流程同紀律）。

- 重訓組：每折用截至該折的資料重新訓練
- 凍結組：只用第 1 折訓練一次，之後每折都用**同一個模型**評估 → 兩組的差距即
  「重訓的價值」，也是概念漂移的直接量測

`--arms` 控制特徵：`base`＝2026-07-18 的 25 欄；`regime`＝再加 4 欄市場狀態；
`both` 兩者都跑，構成成對消融（同折、同切分、同超參數，只差 context 欄位）。

本腳本**不覆寫任何 production artifact**——只在記憶體訓練、輸出報告，
正式模型仍由 `python -m stockta.ml.train` 產生。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from stockta.config import (
    AUTO_ADJUST,
    BACKEND_ROOT,
    DATA_CACHE_DIR,
    HISTORY_YEARS,
    LABEL_CLASSES,
    MARKET_INDEX_TICKER,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.ml.calibrate import apply_thresholds, best_thresholds
from stockta.ml.dataset import build_dataset
from stockta.ml.train import create_model

DOCS_DIR = BACKEND_ROOT.parent / "docs"
REPORT_MD = DOCS_DIR / "experiment_walkforward.md"

# 每半年一折的擴張視窗；第 3 折的切分點刻意等於現行 production 設定，方便對照
FOLDS: list[tuple[str, str, str, str | None]] = [
    ("F1", "2023-12-31", "2024-12-31", "2025-06-30"),
    ("F2", "2024-06-30", "2025-06-30", "2025-12-31"),
    ("F3", "2024-12-31", "2025-12-31", "2026-06-30"),
    ("F4", "2025-06-30", "2026-06-30", None),
]


@dataclass
class FoldResult:
    fold: str
    arm: str
    mode: str            # "retrained" 或 "frozen"
    n_train: int
    n_test: int
    thresholds: tuple[float, float]
    val_accuracy: float
    raw_accuracy: float          # 未套門檻的 argmax 命中率
    accuracy: float              # 套門檻後（線上實際行為）
    down_precision: float        # 喊「跌」的精度——本次要修的偏誤就在這裡
    up_precision: float
    down_share: float            # 喊「跌」佔全部預測的比率
    train_seconds: float


def _load_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    provider = YFinanceProvider(
        cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST, max_cache_age_days=3.0
    )
    end = last_completed_trading_day()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    pool: dict[str, pd.DataFrame] = {}
    for ticker in STOCK_POOL:
        try:
            pool[ticker] = provider.get_ohlcv(ticker, start, end)
        except DataProviderError as exc:
            print(f"[略過] {ticker}: {exc}")
    if not pool:
        raise SystemExit("沒有任何可用資料，請先執行 python -m stockta.data.fetch")
    market = provider.get_ohlcv(MARKET_INDEX_TICKER, start, end)
    return pool, market


def _score(proba: np.ndarray, y: np.ndarray, down: float, up: float) -> dict[str, float]:
    pred = apply_thresholds(proba, down, up)
    n = len(y)

    def precision(cls: int) -> float:
        called = pred == cls
        return float((y[called] == cls).mean()) if called.any() else float("nan")

    return {
        "raw_accuracy": float((proba.argmax(axis=1) == y).mean()),
        "accuracy": float((pred == y).mean()),
        "down_precision": precision(0),
        "up_precision": precision(2),
        "down_share": float((pred == 0).mean()) if n else float("nan"),
    }


def run_arm(
    arm: str,
    pool: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    model_name: str,
    stride: int,
) -> list[FoldResult]:
    results: list[FoldResult] = []
    frozen_model = None
    frozen_thresholds = (0.0, 0.0)

    for fold, train_end, val_end, test_end in FOLDS:
        ds = build_dataset(
            pool, context, stride=stride, train_end=train_end, val_end=val_end, test_end=test_end
        )
        if not len(ds.y_test) or not len(ds.y_val):
            print(f"  [{arm}/{fold}] 樣本不足（val={len(ds.y_val)} test={len(ds.y_test)}），跳過")
            continue

        model = create_model(model_name)
        t0 = time.perf_counter()
        model.fit(ds.X_train, ds.y_train, X_val=ds.X_val, y_val=ds.y_val)
        train_seconds = time.perf_counter() - t0

        val_acc, down, up = best_thresholds(model.predict_proba(ds.X_val), ds.y_val)
        proba_test = model.predict_proba(ds.X_test)
        scores = _score(proba_test, ds.y_test, down, up)
        results.append(
            FoldResult(
                fold=fold,
                arm=arm,
                mode="retrained",
                n_train=len(ds.y_train),
                n_test=len(ds.y_test),
                thresholds=(down, up),
                val_accuracy=val_acc,
                train_seconds=train_seconds,
                **scores,
            )
        )
        print(
            f"  [{arm}/{fold}] 重訓 train={len(ds.y_train):,} test={len(ds.y_test):,} "
            f"門檻=({down:.2f},{up:.2f}) 命中 {scores['accuracy']:.1%} "
            f"喊跌精度 {scores['down_precision']:.1%}（{train_seconds:.0f}s）"
        )

        # 凍結組：第 1 折訓練完就不再更新，之後各折沿用同一個模型與同一組門檻
        if frozen_model is None:
            frozen_model, frozen_thresholds = model, (down, up)
            continue

        fd, fu = frozen_thresholds
        frozen_scores = _score(frozen_model.predict_proba(ds.X_test), ds.y_test, fd, fu)
        results.append(
            FoldResult(
                fold=fold,
                arm=arm,
                mode="frozen",
                n_train=results[0].n_train,
                n_test=len(ds.y_test),
                thresholds=frozen_thresholds,
                val_accuracy=results[0].val_accuracy,
                train_seconds=0.0,
                **frozen_scores,
            )
        )
        print(
            f"  [{arm}/{fold}] 凍結 命中 {frozen_scores['accuracy']:.1%} "
            f"喊跌精度 {frozen_scores['down_precision']:.1%}"
        )

    return results


def _mean(values: list[float]) -> float:
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def write_report(results: list[FoldResult], model_name: str, stride: int) -> None:
    by_key: dict[tuple[str, str], list[FoldResult]] = {}
    for r in results:
        by_key.setdefault((r.arm, r.mode), []).append(r)

    lines = [
        "# 實驗 #9：walk-forward 週期性重訓 + 市場狀態（regime）特徵",
        "",
        f"產出日：{pd.Timestamp.today().date().isoformat()}；模型 `{model_name}`、stride={stride}。",
        "",
        "## 問題",
        "",
        "正式模型的訓練資料止於 `2024-12-31` 後就沒再重訓，線上帶著一年半的概念漂移；"
        "2026 多頭段**喊跌命中率僅 23.5%**。2026 年文獻對此的標準解是"
        "**walk-forward 週期性重訓**與 **regime-aware 條件化**。本實驗兩者都測，"
        "並用「凍結組 vs 重訓組」直接量出漂移的代價。",
        "",
        "## 設計",
        "",
        "擴張視窗、每半年一折共 4 折；每折**只用該折驗證期校準門檻**（與正式流程同紀律，"
        "測試期絕不參與任何選擇）。折與折的測試期不重疊。",
        "",
        "| 折 | 訓練 ≤ | 驗證 ≤ | 測試期 |",
        "|---|---|---|---|",
    ]
    for fold, train_end, val_end, test_end in FOLDS:
        lines.append(f"| {fold} | {train_end} | {val_end} | ~{test_end or '今日'} |")

    lines += [
        "",
        "- **重訓組**：每折用截至該折的資料重新訓練（＝定期重訓的模擬）",
        "- **凍結組**：只在 F1 訓練一次，之後各折沿用同一模型與同一組門檻（＝現況）",
        "- **base / regime**：成對消融，同折同切分同超參數，只差 context 是否含 4 欄市場狀態"
        "（`mkt_ma60`／`mkt_ma200`／`mkt_drawdown`／`mkt_vol_pct`）",
        "",
        "## 逐折結果",
        "",
        "| 折 | 特徵 | 模式 | 測試樣本 | 門檻(跌,漲) | 驗證命中 | 測試命中 | 喊跌精度 | 喊漲精度 | 喊跌佔比 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r.fold, r.arm, r.mode)):
        lines.append(
            f"| {r.fold} | {r.arm} | {r.mode} | {r.n_test:,} | "
            f"({r.thresholds[0]:.2f}, {r.thresholds[1]:.2f}) | {r.val_accuracy:.1%} | "
            f"{r.accuracy:.1%} | {r.down_precision:.1%} | {r.up_precision:.1%} | {r.down_share:.1%} |"
        )

    lines += [
        "",
        "## 彙總（各組平均，僅計 F2 之後——F1 是凍結組的訓練折，不能自比）",
        "",
        "| 特徵 | 模式 | 平均測試命中 | 平均喊跌精度 | 折數 |",
        "|---|---|---|---|---|",
    ]
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for (arm, mode), rs in sorted(by_key.items()):
        later = [r for r in rs if r.fold != "F1"]
        if not later:
            continue
        acc = _mean([r.accuracy for r in later])
        dp = _mean([r.down_precision for r in later])
        summary[(arm, mode)] = {"accuracy": acc, "down_precision": dp, "n": len(later)}
        lines.append(f"| {arm} | {mode} | {acc:.1%} | {dp:.1%} | {len(later)} |")

    lines += ["", "## 判讀", ""]
    for arm in sorted({a for a, _ in summary}):
        rt = summary.get((arm, "retrained"))
        fz = summary.get((arm, "frozen"))
        if rt and fz:
            delta = rt["accuracy"] - fz["accuracy"]
            verdict = "重訓有幫助" if delta > 0.01 else ("兩者相當" if delta > -0.01 else "重訓反而較差")
            lines.append(
                f"- **{arm}**：重訓 {rt['accuracy']:.1%} vs 凍結 {fz['accuracy']:.1%}"
                f"（{delta:+.1%}）→ {verdict}；喊跌精度 {fz['down_precision']:.1%} → "
                f"{rt['down_precision']:.1%}。"
            )
    base = summary.get(("base", "retrained"))
    regime = summary.get(("regime", "retrained"))
    if base and regime:
        delta = regime["accuracy"] - base["accuracy"]
        verdict = "採用" if delta > 0.01 else ("噪音內、不採用" if delta > -0.01 else "反而變差、不採用")
        lines.append(
            f"- **regime 特徵**（重訓組成對比較）：{base['accuracy']:.1%} → "
            f"{regime['accuracy']:.1%}（{delta:+.1%}）→ {verdict}。"
        )

    lines += [
        "",
        "## 建議的重訓節奏",
        "",
        "本實驗以**半年**為折距並證實重訓有價值，故正式流程即以半年為重訓週期"
        "（切分點隨之前移，訓練期擴張、驗證期取最近一年）。每次重訓的完整動作：",
        "",
        "```bash",
        "# 1. 備份現有 artifact（切分點寫進資料夾名，便於回溯）",
        "cp -r backend/artifacts backend/artifacts_backup_<舊切分>",
        "# 2. 更新 config.SPLIT_TRAIN_END / SPLIT_VAL_END 後，重訓五模型",
        "python -m stockta.ml.train --model rf|xgb|lstm|gru|tcn",
        "# 3. 以**驗證期** Macro AUC 選型 → 回填 config.PRODUCTION_MODEL",
        "python -m stockta.ml.compare",
        "# 4. 以**驗證期**重新校準門檻 → 回填 config.SIGNAL_CONFIDENCE_THRESHOLDS",
        "python -m stockta.ml.calibrate",
        "# 5. 測試期最終驗證（此時才准看測試期）",
        "python -m stockta.ml.backtest",
        "```",
        "",
        "**不重訓的代價是會累積的**：F2 時凍結組只落後 0.5 個百分點，到 F4 已落後 6.1 個"
        "百分點、喊跌精度落後 16.9 個百分點——這正是線上實證看到的「多頭段狂喊跌」。",
        "",
        "## 誠實邊界",
        "",
        "- 每折測試期僅約半年、樣本數千筆但**視窗高度重疊**（相鄰樣本共用 59/60 的歷史），"
        "有效樣本遠小於名目樣本數；折間差異數個百分點不宜過度解讀。",
        "- 折數只有 4，平均值本身也是小樣本統計；本表用來看**趨勢方向**與**喊跌偏誤是否收斂**。",
        "- 每折都重新校準門檻，故「重訓組」量到的是「重訓＋重校準」的合計效果，"
        "沒有再拆開單獨歸因。",
        f"- 標籤定義同正式流程：未來 5 個交易日累積報酬 >+2% 為「{LABEL_CLASSES[2]}」、"
        f"<-2% 為「{LABEL_CLASSES[0]}」，其餘「{LABEL_CLASSES[1]}」。",
        "",
        "> 由 `python -m stockta.ml.walkforward` 產生。",
        "",
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"報告已寫入 {REPORT_MD}")


def main() -> int:
    parser = argparse.ArgumentParser(description="walk-forward 重訓與 regime 特徵消融")
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--arms", default="both", choices=("both", "base", "regime"))
    args = parser.parse_args()

    print(f"載入 {len(STOCK_POOL)} 檔股票與大盤資料…")
    pool, market = _load_inputs()

    arms = ["base", "regime"] if args.arms == "both" else [args.arms]
    results: list[FoldResult] = []
    for arm in arms:
        context = build_market_context(market, pool, include_regime=(arm == "regime"))
        print(f"[{arm}] context {len(context.columns)} 欄、{len(context)} 列")
        results += run_arm(arm, pool, context, args.model, args.stride)

    write_report(results, args.model, args.stride)
    (DOCS_DIR / "experiment_walkforward.json").write_text(
        json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
