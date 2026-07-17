"""API 端市場情境服務：^TWII 走 DataProvider（快取＋網路降級），
市場寬度只讀本地 parquet 快取（單檔預測不打 49 檔網路）。

寬度依賴全池快取的新鮮度——每日排程 record_predictions 會刷新全池；
若快取落後大盤（例如排程沒跑），拋 DataProviderError 提示，而不是
默默用舊寬度或讓 base_date 悄悄倒退。以 end 日期記憶化，同日多次請求零成本。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockta.config import (
    INDICATOR_WARMUP_DAYS,
    MARKET_INDEX_TICKER,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days
from stockta.data.provider import DataProvider, DataProviderError
from stockta.features.market import build_market_context

# context 需涵蓋個股特徵的整段回溯，再加自身 rolling 暖機（20+5 日）的餘裕
_CONTEXT_EXTRA_DAYS = 60


class MarketContextService:
    def __init__(self, provider: DataProvider, cache: ParquetCache, pool: list[str]):
        self._provider = provider
        self._cache = cache
        self._pool = pool
        self._lookback_days = (
            calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS) + _CONTEXT_EXTRA_DAYS
        )
        self._memo: dict[date, pd.DataFrame] = {}

    def get(self, end: date) -> pd.DataFrame:
        if end in self._memo:
            return self._memo[end]
        start = end - timedelta(days=self._lookback_days)

        market = self._provider.get_ohlcv(MARKET_INDEX_TICKER, start, end)

        pool_ohlcv: dict[str, pd.DataFrame] = {}
        for ticker in self._pool:
            df = self._cache.read(ticker)
            if df is None:
                continue
            sliced = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
            if not sliced.empty:
                pool_ohlcv[ticker] = sliced

        context = build_market_context(market, pool_ohlcv)
        if context.empty or context.index[-1] < market.index[-1]:
            raise DataProviderError(
                "市場寬度資料落後大盤（股票池快取過舊或不足），"
                "請先執行 python -m stockta.ml.record_predictions 或 python -m stockta.data.fetch"
            )
        self._memo[end] = context
        return context
