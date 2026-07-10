"""ticker 白名單驗證 —— 兩支 /candles、/prediction API 共用同一處檢查。

先驗格式（防路徑穿越，422）再驗是否在股票池內（模型/快取沒訓練過的代號，404）。
"""

from fastapi import Path

from stockta.api.errors import InvalidTickerFormatError, TickerNotFoundError
from stockta.config import STOCK_POOL
from stockta.data.cache import InvalidTickerError, validate_ticker


def get_valid_ticker(ticker: str = Path(...)) -> str:
    try:
        validate_ticker(ticker)
    except InvalidTickerError:
        raise InvalidTickerFormatError(ticker)
    if ticker not in STOCK_POOL:
        raise TickerNotFoundError(ticker)
    return ticker
