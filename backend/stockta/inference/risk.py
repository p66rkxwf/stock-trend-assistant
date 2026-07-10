"""風險等級：近 N 日日報酬的年化波動率 → 低/中/高（純函式，門檻讀 config）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockta.config import (
    RISK_VOL_HIGH_MIN,
    RISK_VOL_LOW_MAX,
    RISK_WINDOW_DAYS,
    TRADING_DAYS_PER_YEAR,
)


def annualized_volatility(close: pd.Series, window: int = RISK_WINDOW_DAYS) -> float:
    returns = close.pct_change().dropna().iloc[-window:]
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def risk_level(volatility: float) -> str:
    if np.isnan(volatility):
        return "中"  # 資料不足時保守回中風險
    if volatility <= RISK_VOL_LOW_MAX:
        return "低"
    if volatility >= RISK_VOL_HIGH_MIN:
        return "高"
    return "中"
