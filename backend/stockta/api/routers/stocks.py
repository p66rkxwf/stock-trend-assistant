from datetime import timedelta

from fastapi import APIRouter, Depends, Query, Request

from stockta.api.deps import get_valid_ticker
from stockta.api.errors import (
    DataInsufficientError,
    DataSourceUnavailableError,
    InvalidRangeError,
)
from stockta.api.schemas import Candle, CandlesResponse, PredictionResponse
from stockta.config import INDICATOR_WARMUP_DAYS, WINDOW_LENGTH_DAYS
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProvider, DataProviderError
from stockta.inference.predictor import InsufficientDataError, Predictor
from stockta.inference.risk import annualized_volatility, risk_level

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
def get_prediction(
    request: Request, ticker: str = Depends(get_valid_ticker)
) -> PredictionResponse:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        # 尚未訓練出 artifact 時的 mock 回應，前端以 is_mock 判斷顯示提示
        return PredictionResponse(
            ticker=ticker,
            base_date=last_completed_trading_day(),
            signal="觀望",
            confidence=0.34,
            risk="中",
            model_version="mock-0.0.0",
            is_mock=True,
        )

    provider: DataProvider = request.app.state.data_provider
    end = last_completed_trading_day()
    start = end - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS))

    try:
        df = provider.get_ohlcv(ticker, start, end)
        context = request.app.state.market_context.get(end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc

    try:
        pred = predictor.predict(df, context)
    except InsufficientDataError as exc:
        raise DataInsufficientError(str(exc)) from exc

    risk = risk_level(annualized_volatility(df["close"]))

    store = getattr(request.app.state, "prediction_store", None)
    if store is not None:
        # 線上預測落地（同 ticker+基準日+版本只記第一筆），累積競賽實證資料
        store.record(ticker, pred.base_date, pred.signal, pred.confidence, predictor.version)

    return PredictionResponse(
        ticker=ticker,
        base_date=pred.base_date,
        signal=pred.signal,
        confidence=pred.confidence,
        risk=risk,
        model_version=predictor.version,
        is_mock=False,
    )
