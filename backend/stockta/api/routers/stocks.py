from datetime import timedelta

from fastapi import APIRouter, Depends, Query, Request

from stockta.api.deps import get_valid_ticker
from stockta.api.errors import DataSourceUnavailableError, InvalidRangeError
from stockta.api.schemas import Candle, CandlesResponse, PredictionResponse
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProvider, DataProviderError

router = APIRouter(prefix="/api/stocks", tags=["stocks"])

_RANGE_TO_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


@router.get("/{ticker}/candles", response_model=CandlesResponse)
def get_candles(
    request: Request,
    ticker: str = Depends(get_valid_ticker),
    range: str = Query("1y", alias="range"),
) -> CandlesResponse:
    days = _RANGE_TO_DAYS.get(range)
    if days is None:
        raise InvalidRangeError(range)

    provider: DataProvider = request.app.state.data_provider
    end = last_completed_trading_day()
    start = end - timedelta(days=days)

    try:
        df = provider.get_ohlcv(ticker, start, end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc

    candles = [
        Candle(
            time=idx.date(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        )
        for idx, row in df.iterrows()
    ]
    return CandlesResponse(ticker=ticker, candles=candles)


@router.get("/{ticker}/prediction", response_model=PredictionResponse)
def get_prediction(ticker: str = Depends(get_valid_ticker)) -> PredictionResponse:
    # Phase 6 前的 mock 回應：尚未整合訓練模型，回傳固定結構供前端開發使用。
    # 待 stockta/inference/predictor.py 完成後，改為呼叫真實推論並移除 is_mock。
    return PredictionResponse(
        ticker=ticker,
        base_date=last_completed_trading_day(),
        signal="觀望",
        confidence=0.34,
        risk="中",
        model_version="mock-0.0.0",
        is_mock=True,
    )
