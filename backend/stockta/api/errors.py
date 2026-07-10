"""統一錯誤回應格式：{"error": {"code": ..., "message": ...}}。

404 = 代號不存在、422 = 格式錯誤、503 = 資料源失敗 —— 這是 Phase 0 凍結的契約，
前後端都依此判斷錯誤類型，不要用字串比對 message 內容做分支邏輯。
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


class TickerNotFoundError(ApiError):
    def __init__(self, ticker: str):
        super().__init__(404, "TICKER_NOT_FOUND", f"股票代號 {ticker} 不在支援清單中")


class InvalidTickerFormatError(ApiError):
    def __init__(self, ticker: str):
        super().__init__(422, "INVALID_TICKER_FORMAT", f"股票代號格式錯誤: {ticker}")


class InvalidRangeError(ApiError):
    def __init__(self, range_value: str):
        super().__init__(422, "INVALID_RANGE", f"不支援的 range 參數: {range_value}")


class DataSourceUnavailableError(ApiError):
    def __init__(self, detail: str):
        super().__init__(503, "DATA_SOURCE_UNAVAILABLE", detail)


class DataInsufficientError(ApiError):
    def __init__(self, detail: str):
        super().__init__(422, "INSUFFICIENT_DATA", detail)


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
