"""Parquet 快取讀寫，以及 ticker 格式驗證。

ticker 會被用來組出快取檔名，未經驗證直接使用等於開放路徑穿越
（例如 ticker="../../etc/passwd"）。所有存取快取的路徑都必須先經過 validate_ticker()。
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_TICKER_RE = re.compile(r"^\d{4,6}\.TW$")


class InvalidTickerError(ValueError):
    """ticker 格式不符合台股代號規則。"""


def validate_ticker(ticker: str) -> str:
    if not _TICKER_RE.match(ticker):
        raise InvalidTickerError(f"不合法的股票代號格式: {ticker!r}")
    return ticker


class ParquetCache:
    """單一股票一個 parquet 檔，以日期為索引存放 OHLCV。"""

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        validate_ticker(ticker)
        return self._dir / f"{ticker}.parquet"

    def read(self, ticker: str) -> pd.DataFrame | None:
        path = self._path(ticker)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def write(self, ticker: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(ticker))

    def is_fresh(self, ticker: str, max_age_days: float) -> bool:
        path = self._path(ticker)
        if not path.exists():
            return False
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return age_seconds < max_age_days * 86400
