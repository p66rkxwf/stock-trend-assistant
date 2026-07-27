"""Cross-sectional 排序的標準評估指標（experiment_log #8）。

- Rank IC：逐日「模型分數 vs 實際報酬」的 Spearman 相關，平均與 t 值——業界衡量
  選股技能的標準指標（Alphalens / Grinold 主動管理基本定律 IR=IC×√Breadth）。
- 分位數多空價差：做多分數前段、放空後段的報酬差（gross）。
- 組合回測：long-only 前 K 分位、等權、非重疊換股，vs 等權全池基準；同時算
  gross 與扣交易成本後（net），供誠實呈現。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(dates: np.ndarray, scores: np.ndarray, rets: np.ndarray) -> dict:
    """逐日 Spearman(分數, 未來報酬)，回傳平均 IC、t 值、有效天數與分年 IC。"""
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "s": scores, "r": rets})
    daily = []
    for d, g in df.groupby("date"):
        if len(g) < 10 or g["s"].nunique() < 2 or g["r"].nunique() < 2:
            continue
        ic, _ = spearmanr(g["s"], g["r"])
        if not np.isnan(ic):
            daily.append((d, ic))
    if not daily:
        return {"mean": float("nan"), "t": float("nan"), "n_days": 0, "by_year": {}}
    dd = pd.Series({d: ic for d, ic in daily})
    t = dd.mean() / dd.std(ddof=1) * np.sqrt(len(dd)) if dd.std(ddof=1) > 0 else float("nan")
    by_year = {int(y): float(dd[dd.index.year == y].mean()) for y in sorted(set(dd.index.year))}
    return {"mean": float(dd.mean()), "t": float(t), "n_days": len(dd), "by_year": by_year}


def quantile_spread(
    dates: np.ndarray, scores: np.ndarray, rets: np.ndarray, top_fraction: float = 0.2
) -> float:
    """逐日「前 top_fraction 平均報酬 − 後 top_fraction 平均報酬」，取平均（gross）。"""
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "s": scores, "r": rets})
    spreads = []
    for _d, g in df.groupby("date"):
        if len(g) < 10:
            continue
        k = max(3, int(len(g) * top_fraction))
        gg = g.sort_values("s")
        spreads.append(gg["r"].iloc[-k:].mean() - gg["r"].iloc[:k].mean())
    return float(np.mean(spreads)) if spreads else float("nan")


def portfolio_backtest(
    dates: np.ndarray,
    tickers: np.ndarray,
    scores: np.ndarray,
    rets: np.ndarray,
    holding_days: int,
    top_fraction: float = 0.2,
    cost_bps: float = 0.0,
) -> dict:
    """long-only 前 top_fraction 等權組合，每 holding_days 個交易日非重疊換股，
    vs 等權全池基準。回傳 gross/net 累積與年化報酬、換手、勝率。

    net：每次換股依「新舊持股差異比例」計交易成本（單邊 cost_bps/2 的粗估以來回 cost_bps 計）。
    holding 期報酬直接用該基準日的未來 holding-period 報酬（呼叫端須以對應 horizon 的 rets 傳入）。
    """
    df = pd.DataFrame(
        {"date": pd.DatetimeIndex(dates), "tkr": tickers, "s": scores, "r": rets}
    )
    uniq = np.sort(df["date"].unique())
    rebal = uniq[::holding_days]  # 非重疊持有期

    port_gross, port_net, bench = 1.0, 1.0, 1.0
    prev_holdings: set[str] = set()
    wins = periods = 0
    turnovers = []
    for d in rebal:
        g = df[df["date"] == d].dropna(subset=["r"])  # 未來報酬不足者剔除（資料末端）
        if len(g) < 10:
            continue
        k = max(3, int(len(g) * top_fraction))
        top = g.sort_values("s").iloc[-k:]
        holdings = set(top["tkr"])
        p_ret = float(top["r"].mean())
        b_ret = float(g["r"].mean())
        # 換手：本期持股與上期的差異比例（0~1）；成本 = 換手 × cost_bps
        turnover = len(holdings - prev_holdings) / max(len(holdings), 1) if prev_holdings else 1.0
        cost = turnover * (cost_bps / 1e4)
        turnovers.append(turnover)
        port_gross *= 1 + p_ret
        port_net *= 1 + p_ret - cost
        bench *= 1 + b_ret
        wins += p_ret > b_ret
        periods += 1
        prev_holdings = holdings

    if periods == 0:
        return {"periods": 0}
    years = max((uniq[-1] - uniq[0]) / np.timedelta64(365, "D"), 1e-9)
    return {
        "periods": periods,
        "holding_days": holding_days,
        "gross_cum": port_gross - 1,
        "net_cum": port_net - 1,
        "bench_cum": bench - 1,
        "net_ann": port_net ** (1 / years) - 1,
        "bench_ann": bench ** (1 / years) - 1,
        "net_excess_cum": port_net - bench,
        "avg_turnover": float(np.mean(turnovers)),
        "win_rate": wins / periods,
        "years": float(years),
    }
