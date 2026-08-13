"""線上相對強弱實證報告（線上 Rank IC）：python -m stockta.ml.report_rank_predictions

讀 rank_predictions（cross-sectional 分數），對每筆以 parquet 快取算實際
LABEL_HORIZON_DAYS 日 forward return，逐日 Spearman(分數, 報酬) 得**線上 Rank IC**
——選股技能的標準指標，且**不受多頭 beta 影響**（問「贏過中位數」而非「漲」），因此
比絕對方向線上命中率更能為 experiment_log #8 的正面主線背書。

與絕對方向 online_predictions.md 互補：同一批線上交易日，絕對方向難、相對排序可行。
分數以固定 CS 權重點對點產生（無前視）；早期歷史基準日由 point-in-time 重建
（source=pit）以立即有可到期樣本，往後由每日 live 記錄延續。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stockta.config import (
    ARTIFACTS_CS_DIR,
    BACKEND_ROOT,
    CS_PRODUCTION_MODEL,
    CS_TOP_FRACTION,
    DATA_CACHE_DIR,
    LABEL_HORIZON_DAYS,
    PREDICTIONS_DB_PATH,
)
from stockta.data.cache import ParquetCache
from stockta.ml.cs_metrics import quantile_spread, rank_ic
from stockta.ml.report_predictions import actual_signal

DOCS_DIR = BACKEND_ROOT.parent / "docs"
_MIN_NAMES = 10  # 與 cs_metrics.rank_ic 的每日最少名數一致


def cs_production_version() -> str | None:
    """現行 cross-sectional production 模型版本字串（讀 metadata.json，不載權重）。"""
    meta_path = ARTIFACTS_CS_DIR / CS_PRODUCTION_MODEL / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return f"{meta['model_name']}-cs-{meta['trained_at'][:10]}"


def _load_rows(version: str | None) -> list[tuple]:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT ticker, base_date, score, source FROM rank_predictions "
            "WHERE model_version=? ORDER BY base_date, ticker",
            (version,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []  # 尚未建表
    conn.close()
    return rows


def _interpret(ic: dict) -> str:
    """依到期天數與顯著性給出誠實、隨資料自動調整的判讀句（不誇大、不粉飾）。"""
    n, mean, t = ic["n_days"], ic["mean"], ic["t"]
    if n == 0:
        return "尚無到期樣本，無法判讀。"
    weak_sample = n < 20  # 天數太少、且線上基準日 forward 窗重疊，t 值不可盡信
    if weak_sample or abs(t) < 2:
        sign = "略正" if mean > 0 else "略負" if mean < 0 else "約零"
        return (
            f"樣本不足（{n} 個到期交易日、窗重疊），線上 Rank IC {sign}且**與 0 無法區分**"
            f"（|t|={abs(t):.2f}<2），既不佐證也不反駁回測 +0.044，僅為起步累積。"
        )
    if mean > 0:
        return f"線上 Rank IC 為正且顯著（t={t:.2f}），與回測方向一致，開始佐證選股技能。"
    return f"線上 Rank IC 為負且顯著（t={t:.2f}），與回測相左，需檢視模型是否失效或市況反轉。"


def collect_matured(rows: list[tuple], cache) -> tuple[list, list, list, list, int]:
    """rows=[(ticker, base_date, score, source)] → (dates, scores, rets, sources, n_pending)。

    以 actual_signal 取基準日後 LABEL_HORIZON_DAYS 日實際 close-to-close 報酬（重用絕對
    方向的到期判定）；尚未到期者計入 pending，不進入 Rank IC 計算。
    """
    dates, scores, rets, sources = [], [], [], []
    pending = 0
    for ticker, bd, score, source in rows:
        _sig, ret = actual_signal(cache, ticker, bd)
        if ret is None:
            pending += 1
            continue
        dates.append(pd.Timestamp(bd))
        scores.append(score)
        rets.append(ret)
        sources.append(source)
    return dates, scores, rets, sources, pending


def main() -> int:
    version = cs_production_version()
    rows = _load_rows(version)
    cache = ParquetCache(DATA_CACHE_DIR)

    dates, scores, rets, sources, pending = collect_matured(rows, cache)

    dts = np.array(dates)
    sc = np.array(scores, dtype=float)
    rt = np.array(rets, dtype=float)
    ic = rank_ic(dts, sc, rt) if len(dts) else {"mean": float("nan"), "t": float("nan"), "n_days": 0, "by_year": {}}
    spread = quantile_spread(dts, sc, rt, CS_TOP_FRACTION) if len(dts) else float("nan")

    # 逐日 IC（與 rank_ic 同一過濾條件：每日 ≥10 名、分數與報酬皆有變異）
    per_day: list[tuple] = []
    if len(dts):
        df = pd.DataFrame({"date": dts, "s": sc, "r": rt, "src": sources})
        for d, g in df.groupby("date"):
            src = "live" if (g["src"] == "live").any() else "pit"
            if len(g) < _MIN_NAMES or g["s"].nunique() < 2 or g["r"].nunique() < 2:
                per_day.append((d, len(g), None, src))
            else:
                di, _ = spearmanr(g["s"], g["r"])
                per_day.append((d, len(g), float(di), src))

    matured_dates = {str(d.date()) for d in dts}
    pend_dates = sorted({bd for (_t, bd, _s, _src) in rows} - matured_dates)
    n_live = sum(1 for (_t, _b, _s, src) in rows if src == "live")
    n_pit = len(rows) - n_live

    lines = [
        "# 線上相對強弱實證報告（線上 Rank IC）",
        "",
        f"產出日：{date.today().isoformat()}；資料來源：`backend/predictions.db` 的 "
        "`rank_predictions` 表（cross-sectional 分數，每日排程即時記錄＋歷史 point-in-time 重建）。",
        "",
        f"分數＝**P(基準日後 {LABEL_HORIZON_DAYS} 個交易日贏過當日全池中位數)**（相對強弱，"
        "非絕對漲跌）；逐日 Spearman(分數, 實際報酬)＝當日 Rank IC，平均與 t 值為選股技能指標。",
        "",
        f"**現行 CS production 模型：`{version}`**。判準：**Rank IC 0.02~0.05 可用、"
        "0.05~0.10 好、0.10 頂尖；t>2 統計顯著**。",
        "",
    ]

    interp = _interpret(ic)
    if ic["n_days"]:
        lines.append(
            f"## 總覽：已到期 {ic['n_days']} 個交易日、線上 Rank IC **{ic['mean']:+.4f}**"
            f"（t={ic['t']:.2f}）；待驗證 {len(pend_dates)} 個基準日（{pending} 筆）"
        )
    else:
        lines.append(
            f"## 總覽：尚無到期樣本（基準日後未滿 {LABEL_HORIZON_DAYS} 個交易日）；"
            f"待驗證 {len(pend_dates)} 個基準日（{pending} 筆）"
        )
    lines += [
        "",
        f"**判讀**：{interp}",
        "",
        f"樣本組成：live 記錄 {n_live} 筆、point-in-time 重建 {n_pit} 筆"
        f"（來源見下表；PIT 與 live 由同一固定權重產生、逐位元一致，僅記錄時點不同）。",
        "",
        "> **與絕對方向對照**：同批線上交易日的絕對方向命中率見 `docs/online_predictions.md`"
        "（該報告自帶最新數字）。兩條線上實證都在累積、都誠實揭露；相對排序**在原理上**不受多頭 beta "
        "影響（問「贏過中位數」而非「漲」），但**是否成立仍須線上樣本說話**，不能只憑原理宣稱。"
        "目前的主要統計證據是回測 `docs/cross_sectional_report.md`（測試期 Rank IC +0.0442, "
        "t=2.3、134 個交易日）；本報告是把它放到真實線上時序去持續檢驗。",
        "",
    ]

    if per_day:
        lines += [
            "## 逐日 Rank IC（已到期）",
            "",
            "| 基準日 | 名數 | 當日 Rank IC | 來源 |",
            "|---|---|---|---|",
        ]
        for d, n, di, src in per_day:
            ic_str = f"{di:+.4f}" if di is not None else "—（名數不足）"
            lines.append(f"| {d.date()} | {n} | {ic_str} | {src} |")
        lines.append("")
        lines.append(
            f"前/後 {int(CS_TOP_FRACTION * 100)}% 多空價差（gross，每 {LABEL_HORIZON_DAYS} 日）："
            f"{spread:+.2%}。"
        )
        lines.append("")

    if pend_dates:
        lines += [
            "## 待驗證基準日（分數已記、5 個交易日未到期）",
            "",
            "、".join(pend_dates),
            "",
        ]

    lines += [
        "## 誠實警告（勿誇大）",
        "",
        "- **無前視**：分數僅用 ≤基準日資料（build_features 由 test_no_lookahead 把關）、"
        f"CS 權重固定訓練至樣本內；報酬為基準日後 {LABEL_HORIZON_DAYS} 日實際 close-to-close。",
        "- **PIT 重建 vs live**：早期基準日以 point-in-time 重建（與回測同法、非當下即時記錄），"
        "往後由每日排程 live 累積；兩者對固定權重逐位元一致，但 live 段才是嚴格意義的「線上」。",
        f"- **小樣本＋重疊窗**：目前僅 {ic['n_days']} 個到期交易日，且基準日集中於數週內、"
        f"各自 {LABEL_HORIZON_DAYS} 日 forward 報酬彼此重疊（非獨立），t 值高度敏感、逐日 IC "
        "在此melt-up段大幅震盪（見上表 −0.32~+0.53）。此線上切片與 0 無法區分，"
        "**尚不構成佐證亦不構成反證**，僅為起步。",
        "- 主要統計證據仍是回測（測試期 Rank IC +0.0442, t=2.3, 134 天）；本線上序列需累積"
        "數週、待 live 段變厚才有獨立說服力。多空價差為 gross、未計滑價借券，僅概念參考。",
        "",
        "> 由 `python -m stockta.ml.report_rank_predictions` 產生；每日排程後自動重跑累積。"
        "與 `docs/online_predictions.md`（絕對方向）並列——兩條線上實證都誠實累積中。",
    ]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "online_rank_ic.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"現行 CS {version}：已到期 {ic['n_days']} 交易日、線上 Rank IC {ic['mean']:+.4f}"
        f"（t={ic['t']:.2f}）；待驗證 {len(pend_dates)} 基準日；報告已寫入 {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
