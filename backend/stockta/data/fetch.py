"""Phase 1 資料擷取 CLI：python -m stockta.data.fetch

依 config.STOCK_POOL 抓取近 HISTORY_YEARS 年日線 OHLCV 進 parquet 快取，
並產出 docs/data_exploration.md（資料範圍、缺值、標籤三類分佈——
Phase 3 據此決定是否調整 ±2% 門檻或 class weight）。

快取更新策略（PLAN.md Phase 1 要求文件化）：增量合併——每次執行下載
最新區間並與既有 parquet 以日期去重合併（keep=last），不整包重抓。
"""

from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd

from stockta.config import (
    AUTO_ADJUST,
    BACKEND_ROOT,
    DATA_CACHE_DIR,
    HISTORY_YEARS,
    LABEL_HORIZON_DAYS,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.ml.labeling import make_labels

DOCS_DIR = BACKEND_ROOT.parent / "docs"
PAUSE_SECONDS = 0.6  # 對 Yahoo 溫柔一點，避免整批抓取被限流


def main() -> None:
    provider = YFinanceProvider(
        cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST, max_cache_age_days=0.0
    )
    end = last_completed_trading_day()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    rows = []
    label_totals = {"跌": 0, "觀望": 0, "漲": 0}
    failed: list[str] = []
    for i, (ticker, name) in enumerate(STOCK_POOL.items(), 1):
        try:
            df = provider.get_ohlcv(ticker, start, end)
        except DataProviderError as exc:
            print(f"[{i}/{len(STOCK_POOL)}] {ticker} {name} 失敗: {exc}")
            failed.append(f"{ticker} {name}")
            continue

        labels = make_labels(df["close"]).dropna()
        dist = {"跌": int((labels == 0).sum()), "觀望": int((labels == 1).sum()), "漲": int((labels == 2).sum())}
        for k, v in dist.items():
            label_totals[k] += v

        rows.append(
            {
                "代號": ticker,
                "名稱": name,
                "起日": df.index.min().date(),
                "迄日": df.index.max().date(),
                "交易日數": len(df),
                "缺值列": int(df.isna().any(axis=1).sum()),
                "跌": dist["跌"],
                "觀望": dist["觀望"],
                "漲": dist["漲"],
            }
        )
        print(f"[{i}/{len(STOCK_POOL)}] {ticker} {name}: {len(df)} 個交易日")
        time.sleep(PAUSE_SECONDS)

    _write_report(rows, label_totals, failed, str(start), str(end))


def _write_report(rows, label_totals, failed, start, end) -> None:
    total = sum(label_totals.values()) or 1
    dist_line = "、".join(f"{k} {v}（{v / total:.1%}）" for k, v in label_totals.items())

    lines = [
        "# 資料探索報告（Phase 1）",
        "",
        f"- 抓取區間：{start} ～ {end}（近 {HISTORY_YEARS} 年日線，auto_adjust={AUTO_ADJUST}）",
        f"- 股票池：{len(rows)} 檔成功" + (f"、{len(failed)} 檔失敗（{'、'.join(failed)}）" if failed else ""),
        f"- 標籤定義：未來 {LABEL_HORIZON_DAYS} 個交易日累積報酬 >+2% 漲／<−2% 跌／其餘觀望",
        f"- **全池標籤分佈：{dist_line}**",
        "",
        "「觀望」為多數類——Accuracy 會被灌水，評估必看 Macro F1／Macro AUC，",
        "訓練以 class weight 抵銷不平衡（見 PLAN.md Phase 3）。",
        "",
        "## 各檔明細",
        "",
    ]
    if rows:
        headers = list(rows[0])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines += [
        "",
        "> 由 `python -m stockta.data.fetch` 自動產生。快取更新策略：增量合併",
        "> （下載最新區間與既有 parquet 以日期去重合併），團隊成員重跑即可對齊快取。",
        "",
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "data_exploration.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n標籤分佈：{dist_line}")
    print(f"報告已寫入 {out}")


if __name__ == "__main__":
    main()
