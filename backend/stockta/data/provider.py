"""OHLCV 資料存取介面。

訓練管線與 API 都只依賴 DataProvider 這個介面，不直接呼叫 yfinance——
這是本專案唯一與不穩定外部服務（yfinance 爬 Yahoo 非官方介面，會限流、會改版）
的接觸點，測試一律注入假的 DataProvider 實作，不打真實網路。
"""

from __future__ import annotations

import abc
from datetime import date, timedelta

import pandas as pd

from stockta.data.cache import ParquetCache

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProviderError(RuntimeError):
    """資料來源無法提供資料（連網失敗且無可用快取，或指定區間內無資料）。"""


class DataProvider(abc.ABC):
    @abc.abstractmethod
    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """回傳欄位為 open/high/low/close/volume、以日期為索引（遞增排序）的 DataFrame。"""
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """以 yfinance 為資料源，內建 parquet 快取與連網失敗降級。

    策略：快取夠新則直接用快取；否則嘗試下載最新資料並與快取合併回存；
    下載失敗時退回舊快取（若涵蓋所需區間），只有兩者皆不可用才拋出例外。
    """

    def __init__(self, cache: ParquetCache, auto_adjust: bool, max_cache_age_days: float = 1.0):
        self._cache = cache
        self._auto_adjust = auto_adjust
        self._max_cache_age_days = max_cache_age_days

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        cached = self._cache.read(ticker)

        if cached is not None and self._cache.is_fresh(ticker, self._max_cache_age_days):
            sliced = _slice(cached, start, end)
            if sliced is not None:
                return sliced

        try:
            fresh = _download(ticker, start, end, self._auto_adjust)
        except Exception as exc:
            sliced = _slice(cached, start, end) if cached is not None else None
            if sliced is not None:
                return sliced
            raise DataProviderError(f"無法取得 {ticker} 的資料，且無可用快取") from exc

        merged = _merge(cached, fresh) if cached is not None else fresh
        self._cache.write(ticker, merged)

        sliced = _slice(merged, start, end)
        if sliced is None:
            raise DataProviderError(f"{ticker} 在 {start}–{end} 區間內無資料")
        return sliced


def _download(ticker: str, start: date, end: date, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),  # yfinance 的 end 是不含端點，+1 天才含 end 當天
        auto_adjust=auto_adjust,
        progress=False,
    )
    if df.empty:
        raise DataProviderError(f"yfinance 回傳 {ticker} 空資料")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.lower)
    return df[_OHLCV_COLUMNS].sort_index()


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame | None:
    sliced = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    return sliced if not sliced.empty else None
