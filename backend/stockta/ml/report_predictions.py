"""線上預測實證報告：python -m stockta.ml.report_predictions

讀 predictions.db 的線上預測紀錄，對每筆以 parquet 快取中 base_date 之後
第 LABEL_HORIZON_DAYS 個交易日的實際累積報酬對照（與訓練標籤同一定義），
產出 docs/online_predictions.md——「線上預測 vs 實際走勢」比測試集 AUC
更有競賽說服力（PLAN.md Phase 6/9）。

**依模型版本分層**：predictions.db 會累積歷次換模型的紀錄，混在一起算命中率
會誤導。報告以「現行 production 版本」為主體，其餘（已淘汰）版本另列對照表，
避免把不同模型的表現混為一談。尚未到期者（base_date 後不足 5 個交易日）列為待驗證。
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import date

import pandas as pd

from stockta.config import (
    ARTIFACTS_DIR,
    BACKEND_ROOT,
    DATA_CACHE_DIR,
    LABEL_DOWN_THRESHOLD,
    LABEL_HORIZON_DAYS,
    LABEL_UP_THRESHOLD,
    PREDICTIONS_DB_PATH,
    PRODUCTION_MODEL,
    SIGNAL_CONFIDENCE_THRESHOLDS,
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


def current_production_version() -> str | None:
    """現行 production 模型的版本字串（與 inference.predictor.Predictor.version 同一構成，
    不載入模型權重，僅讀 metadata.json）。"""
    meta_path = ARTIFACTS_DIR / PRODUCTION_MODEL / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    suffix = "+cal" if SIGNAL_CONFIDENCE_THRESHOLDS else ""
    return f"{meta['model_name']}-{meta['trained_at'][:10]}{suffix}"


def main() -> None:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    rows = conn.execute(
        "SELECT ticker, base_date, signal, confidence, model_version, created_at "
        "FROM predictions ORDER BY base_date DESC, ticker"
    ).fetchall()
    conn.close()

    cache = ParquetCache(DATA_CACHE_DIR)
    # 依版本分桶
    matured: dict[str, list] = defaultdict(list)
    pending: dict[str, list] = defaultdict(list)
    first_date: dict[str, str] = {}
    for ticker, base_date, signal, confidence, version, _created in rows:
        first_date[version] = min(first_date.get(version, base_date), base_date)
        actual, ret = actual_signal(cache, ticker, base_date)
        name = STOCK_POOL.get(ticker, "")
        if actual is None:
            pending[version].append((ticker, name, base_date, signal, confidence))
        else:
            matured[version].append((ticker, name, base_date, signal, confidence, actual, ret, actual == signal))

    prod = current_production_version()
    prod_matured = matured.get(prod, [])
    prod_pending = pending.get(prod, [])
    prod_hits = sum(1 for r in prod_matured if r[7])

    lines = [
        "# 線上預測實證報告",
        "",
        f"產出日：{date.today().isoformat()}；資料來源：`backend/predictions.db`"
        f"（API `/prediction` 與每日排程每次推論自動落地，同 ticker+基準日+版本去重）。",
        "",
        f"標籤定義與訓練一致：基準日後 {LABEL_HORIZON_DAYS} 個交易日累積報酬"
        f" > {LABEL_UP_THRESHOLD:+.0%} 為「漲」、< {LABEL_DOWN_THRESHOLD:+.0%} 為「跌」，其餘「觀望」。",
        "",
        f"**現行 production 模型：`{prod}`**。以下總覽與明細僅計此版本；歷次換模型的"
        "已淘汰版本另見文末對照表，不與現行版本混算。",
        "",
    ]

    if prod_matured:
        lines.append(
            f"## 總覽（現行模型）：已到期 {len(prod_matured)} 筆、命中 {prod_hits} 筆"
            f"（{prod_hits / len(prod_matured):.1%}）；待驗證 {len(prod_pending)} 筆"
        )
    else:
        lines.append(
            f"## 總覽（現行模型）：已到期 0 筆（上線未滿 {LABEL_HORIZON_DAYS} 個交易日，尚無可驗證樣本）；"
            f"待驗證 {len(prod_pending)} 筆"
        )
    lines.append("")

    if prod_matured:
        lines += [
            "## 現行模型：已到期預測 vs 實際",
            "",
            "| 代號 | 名稱 | 基準日 | 預測 | 信心 | 實際 | 5日報酬 | 命中 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, name, b, sig, conf, actual, ret, ok in prod_matured:
            lines.append(
                f"| {t} | {name} | {b} | {sig} | {conf:.2f} | {actual} | {ret:+.2%} | {'✓' if ok else '✗'} |"
            )
        lines.append("")

    if prod_pending:
        lines += [
            "## 現行模型：待驗證（基準日後尚不足 5 個交易日）",
            "",
            "| 代號 | 名稱 | 基準日 | 預測 | 信心 |",
            "|---|---|---|---|---|",
        ]
        for t, name, b, sig, conf in prod_pending:
            lines.append(f"| {t} | {name} | {b} | {sig} | {conf:.2f} |")
        lines.append("")

    # 各版本對照（含已淘汰）——透明呈現，但不與現行版本混算
    all_versions = sorted(set(matured) | set(pending), key=lambda v: first_date.get(v, ""))
    lines += [
        "## 各版本命中率對照（含已淘汰，僅供透明對照，勿混算）",
        "",
        "| 模型版本 | 現行 | 起始基準日 | 已到期 | 命中 | 命中率 | 待驗證 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in all_versions:
        mv = matured.get(v, [])
        hits = sum(1 for r in mv if r[7])
        rate = f"{hits / len(mv):.1%}" if mv else "—"
        lines.append(
            f"| `{v}` | {'✔' if v == prod else ''} | {first_date.get(v, '')} | "
            f"{len(mv)} | {hits} | {rate} | {len(pending.get(v, []))} |"
        )
    lines.append("")

    lines += [
        "> 由 `python -m stockta.ml.report_predictions` 產生；每日排程後自動重跑累積實證。",
        "> 注意：換模型過渡期各版本樣本少且互不可混算；現行模型需累積數週（每批預測皆有"
        f" {LABEL_HORIZON_DAYS} 個交易日到期延遲）才有統計意義。統計主證據仍為 docs/backtest_report.md"
        "（2.9 萬筆樣本外回測）。",
    ]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "online_predictions.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    total_matured = sum(len(v) for v in matured.values())
    print(
        f"現行模型 {prod}：已到期 {len(prod_matured)} 命中 {prod_hits}；"
        f"全版本已到期合計 {total_matured}，報告已寫入 {out}"
    )


if __name__ == "__main__":
    main()
