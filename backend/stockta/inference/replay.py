"""歷史回放（point-in-time 推論）：以 production 模型重現「模型在過去某日會做的預測」。

與訓練/線上推論共用 build_features 同一條特徵路徑（已由 test_no_lookahead 把關
特徵僅用當日與更早資料），故任一過去日的重算都是誠實的樣本外推論；到期後
可對照實際 5 日走勢。供 /api/scan?date= 與 /api/stocks/{t}/history 使用。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from stockta.config import (
    INDICATOR_WARMUP_DAYS,
    MARKET_INDEX_TICKER,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.calendar import calendar_lookback_days
from stockta.data.provider import DataProvider, DataProviderError
from stockta.features.market import build_market_context
from stockta.features.pipeline import build_features
from stockta.inference.predictor import Predictor, resolve_signal
from stockta.config import STOCK_POOL

_CONTEXT_EXTRA_DAYS = 40


def load_pool_context(
    provider: DataProvider, fetch_start: date, end: date
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """載入全池 OHLCV 並建出市場情境 context（歷史查詢共用）。回傳 (pool_ohlcv, context)。"""
    pool_ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in STOCK_POOL:
        try:
            pool_ohlcv[ticker] = provider.get_ohlcv(ticker, fetch_start, end)
        except DataProviderError:
            continue
    market = provider.get_ohlcv(MARKET_INDEX_TICKER, fetch_start, end)
    return pool_ohlcv, build_market_context(market, pool_ohlcv)


def context_fetch_start(as_of: date) -> date:
    """歷史查詢時，涵蓋一個視窗 + 暖機所需回溯的抓取起日。"""
    return as_of - timedelta(
        days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS) + _CONTEXT_EXTRA_DAYS
    )


def scan_asof(
    predictor: Predictor, ohlcv: pd.DataFrame, context: pd.DataFrame, as_of: date
) -> tuple[np.ndarray, date] | None:
    """回傳 (三類機率, 基準日)：以截至 as_of（含）的資料推論最後一個視窗。
    暖機後不足一個視窗則回 None。基準日為 ≤ as_of 的最後一個特徵日。
    """
    feats = build_features(ohlcv, context)
    feats = feats.loc[feats.index <= pd.Timestamp(as_of)]
    if len(feats) < WINDOW_LENGTH_DAYS:
        return None
    window = feats.iloc[-WINDOW_LENGTH_DAYS:]
    scaled = predictor._scaler.transform(window.to_numpy(dtype=np.float64)).astype(np.float32)
    proba = predictor._model.predict_proba(scaled.reshape(1, -1))[0]
    return proba, window.index[-1].date()


def signal_of(proba: np.ndarray) -> str:
    """機率 → 正式決策規則（含信心門檻）的訊號字串。"""
    from stockta.config import LABEL_CLASSES

    return LABEL_CLASSES[resolve_signal(proba)]


def history_series(
    predictor: Predictor,
    ohlcv: pd.DataFrame,
    context: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    """單一標的在 (start, end] 每個已到期交易日的 point-in-time 預測 vs 實際。
    僅回傳已到期（基準日後滿 LABEL_HORIZON_DAYS 個交易日）的樣本。
    """
    from stockta.config import LABEL_CLASSES, LABEL_HORIZON_DAYS
    from stockta.ml.backtest import collect_probas

    proba, y_true, dates = collect_probas(predictor, ohlcv, context, start)
    if not len(proba):
        return []
    dt = pd.DatetimeIndex(dates)
    keep = dt <= end
    proba, y_true, dt = proba[keep], y_true[keep], dt[keep]

    close = ohlcv["close"].astype("float64")
    pos = close.index.get_indexer(dt)
    fwd = pos + LABEL_HORIZON_DAYS
    base = close.to_numpy()

    rows: list[dict] = []
    for i in range(len(proba)):
        sig_idx = resolve_signal(proba[i])
        ret = float(base[fwd[i]] / base[pos[i]] - 1) if fwd[i] < len(base) else None
        rows.append(
            {
                "date": dt[i].date(),
                "signal": LABEL_CLASSES[sig_idx],
                "confidence": float(proba[i].max()),
                "actual": LABEL_CLASSES[int(y_true[i])],
                "actual_return": ret,
                "hit": int(y_true[i]) == sig_idx,
            }
        )
    rows.reverse()  # 新到舊
    return rows
