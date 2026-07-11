"""線上預測實證報告：python -m stockta.ml.report_predictions

讀 predictions.db 的線上預測紀錄，對每筆以 parquet 快取中 base_date 之後
第 LABEL_HORIZON_DAYS 個交易日的實際累積報酬對照（與訓練標籤同一定義），
產出 docs/online_predictions.md——「線上預測 vs 實際走勢」比測試集 AUC
更有競賽說服力（PLAN.md Phase 6/9）。

尚未到期的預測（base_date 後不足 5 個交易日）列為「待驗證」，
之後重跑本指令即自動更新。
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

from stockta.config import (
    BACKEND_ROOT,
    DATA_CACHE_DIR,
    LABEL_DOWN_THRESHOLD,
    LABEL_HORIZON_DAYS,
    LABEL_UP_THRESHOLD,
    PREDICTIONS_DB_PATH,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache

DOCS_DIR = BACKEND_ROOT.parent / "docs"


def actual_signal(cache: ParquetCache, ticker: str, base_date: str) -> tuple[str | None, float | None]:
    """回傳 (實際訊號, 實際 5 日累積報酬)；資料尚不足時回 (None, None)。"""
    df = cache.read(ticker)
    if df is None:
        return None, None
    ts = pd.Timestamp(base_date)
    if ts not in df.index:
        return None, None
    pos = df.index.get_loc(ts)
    target = pos + LABEL_HORIZON_DAYS
    if target >= len(df):
        return None, None  # 尚未到期
    ret = float(df["close"].iloc[target] / df["close"].iloc[pos] - 1)
    if ret > LABEL_UP_THRESHOLD:
        return "漲", ret
    if ret < LABEL_DOWN_THRESHOLD:
        return "跌", ret
    return "觀望", ret


def main() -> None:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    rows = conn.execute(
        "SELECT ticker, base_date, signal, confidence, model_version, created_at "
        "FROM predictions ORDER BY base_date DESC, ticker"
    ).fetchall()
    conn.close()

    cache = ParquetCache(DATA_CACHE_DIR)
    matured: list[tuple] = []
    pending: list[tuple] = []
    n_correct = 0
    for ticker, base_date, signal, confidence, version, created in rows:
        actual, ret = actual_signal(cache, ticker, base_date)
        name = STOCK_POOL.get(ticker, "")
        if actual is None:
            pending.append((ticker, name, base_date, signal, confidence, version))
        else:
            ok = actual == signal
            n_correct += ok
            matured.append((ticker, name, base_date, signal, confidence, actual, ret, ok))

    lines = [
        "# 線上預測實證報告",
        "",
        f"產出日：{date.today().isoformat()}；資料來源：`backend/predictions.db`"
        f"（API `/prediction` 每次推論自動落地，同 ticker+基準日+版本去重）。",
        "",
        f"標籤定義與訓練一致：基準日後 {LABEL_HORIZON_DAYS} 個交易日累積報酬"
        f" > {LABEL_UP_THRESHOLD:+.0%} 為「漲」、< {LABEL_DOWN_THRESHOLD:+.0%} 為「跌」，"
        "其餘「觀望」。",
        "",
        f"## 總覽：已到期 {len(matured)} 筆"
        + (f"、命中 {n_correct} 筆（{n_correct / len(matured):.1%}）" if matured else "")
        + f"；待驗證 {len(pending)} 筆",
        "",
    ]

    if matured:
        lines += [
            "## 已到期預測 vs 實際",
            "",
            "| 代號 | 名稱 | 基準日 | 預測 | 信心 | 實際 | 5日報酬 | 命中 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, name, b, sig, conf, actual, ret, ok in matured:
            lines.append(
                f"| {t} | {name} | {b} | {sig} | {conf:.2f} | {actual} | {ret:+.2%} "
                f"| {'✓' if ok else '✗'} |"
            )
        lines.append("")

    if pending:
        lines += [
            "## 待驗證（基準日後尚不足 5 個交易日）",
            "",
            "| 代號 | 名稱 | 基準日 | 預測 | 信心 | 模型版本 |",
            "|---|---|---|---|---|---|",
        ]
        for t, name, b, sig, conf, version in pending:
            lines.append(f"| {t} | {name} | {b} | {sig} | {conf:.2f} | {version} |")
        lines.append("")

    lines += [
        "> 由 `python -m stockta.ml.report_predictions` 產生；每日 demo 後重跑即可累積實證。",
        "> 注意：單週樣本極少，命中率波動大，需累積數週才有統計意義。",
    ]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "online_predictions.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已到期 {len(matured)} 筆、待驗證 {len(pending)} 筆，報告已寫入 {out}")


if __name__ == "__main__":
    main()
