import sqlite3
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query, Request

from stockta.api.deps import get_valid_ticker
from stockta.api.errors import (
    DataInsufficientError,
    DataSourceUnavailableError,
    InvalidRangeError,
)
from stockta.api.schemas import (
    Candle,
    CandlesResponse,
    HistoryRecord,
    IndicatorsResponse,
    PastPrediction,
    PredictionHistoryResponse,
    PredictionResponse,
    StockHistoryResponse,
)
from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    LABEL_CLASSES,
    PREDICTIONS_DB_PATH,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProvider, DataProviderError, YFinanceProvider
from stockta.features.pipeline import build_features
from stockta.inference.predictor import InsufficientDataError, Predictor
from stockta.inference.replay import context_fetch_start, history_series, load_pool_context
from stockta.inference.risk import annualized_volatility, risk_level
from stockta.ml.report_predictions import actual_signal

router = APIRouter(prefix="/api/stocks", tags=["stocks"])

_RANGE_TO_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


@router.get("/{ticker}/candles", response_model=CandlesResponse)
def get_candles(
    request: Request,
    ticker: str = Depends(get_valid_ticker),
    range: str = Query("1y", alias="range"),
    start: str | None = Query(None, description="自訂起日 YYYY-MM-DD；與 end 並用時覆蓋 range"),
    end: str | None = Query(None, description="自訂迄日 YYYY-MM-DD"),
) -> CandlesResponse:
    provider: DataProvider = request.app.state.data_provider
    last_day = last_completed_trading_day()

    if start or end:
        # 自訂日期區間（覆蓋 range 預設）
        try:
            start_d = date.fromisoformat(start) if start else last_day - timedelta(days=365)
            end_d = date.fromisoformat(end) if end else last_day
        except ValueError as exc:
            raise InvalidRangeError(f"日期格式需為 YYYY-MM-DD：start={start!r} end={end!r}") from exc
        if start_d >= end_d:
            raise InvalidRangeError(f"起日需早於迄日：{start_d} ~ {end_d}")
    else:
        days = _RANGE_TO_DAYS.get(range)
        if days is None:
            raise InvalidRangeError(range)
        end_d = last_day
        start_d = end_d - timedelta(days=days)

    try:
        df = provider.get_ohlcv(ticker, start_d, end_d)
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

    proba = dict(zip(LABEL_CLASSES, pred.proba))

    return PredictionResponse(
        ticker=ticker,
        base_date=pred.base_date,
        signal=pred.signal,
        confidence=pred.confidence,
        risk=risk,
        model_version=predictor.version,
        is_mock=False,
        proba=proba,
    )


@router.get("/{ticker}/indicators", response_model=IndicatorsResponse)
def get_indicators(
    request: Request, ticker: str = Depends(get_valid_ticker)
) -> IndicatorsResponse:
    """當前技術指標快照——與模型走同一條 build_features 路徑，展示值與模型輸入一致。"""
    provider: DataProvider = request.app.state.data_provider
    end = last_completed_trading_day()
    start = end - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS))

    try:
        df = provider.get_ohlcv(ticker, start, end)
        context = request.app.state.market_context.get(end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc

    feats = build_features(df, context)
    if feats.empty:
        raise DataInsufficientError(f"{ticker} 暖機後無可用特徵列")
    row = feats.iloc[-1]

    return IndicatorsResponse(
        ticker=ticker,
        as_of=feats.index[-1].date(),
        rsi14=float(row["rsi14"]),
        kd_k=float(row["kd_k"]),
        kd_d=float(row["kd_d"]),
        macd_hist=float(row["macd_hist"]),
        bb_pctb=float(row["bb_pctb"]),
        bb_width=float(row["bb_width"]),
        vol_ratio=float(row["vol_ratio"]),
        ma_bias_5=float(row["close_ma5"]),
        ma_bias_20=float(row["close_ma20"]),
        ma_bias_60=float(row["close_ma60"]),
    )


@router.get("/{ticker}/predictions", response_model=PredictionHistoryResponse)
def get_prediction_history(
    ticker: str = Depends(get_valid_ticker),
    limit: int = Query(30, ge=1, le=100),
) -> PredictionHistoryResponse:
    """該股歷史線上預測 vs 實際結果（誠實展示：命中與失誤都列）。"""
    try:
        conn = sqlite3.connect(PREDICTIONS_DB_PATH)
        rows = conn.execute(
            "SELECT base_date, signal, confidence, model_version FROM predictions "
            "WHERE ticker = ? ORDER BY base_date DESC, model_version DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []

    cache = ParquetCache(DATA_CACHE_DIR)
    records = []
    for base_date, signal, confidence, version in rows:
        actual, ret = actual_signal(cache, ticker, base_date)
        records.append(
            PastPrediction(
                base_date=date.fromisoformat(base_date),
                signal=signal,
                confidence=confidence,
                model_version=version,
                actual=actual,
                actual_return=ret,
                hit=(actual == signal) if actual is not None else None,
            )
        )
    return PredictionHistoryResponse(ticker=ticker, records=records)


@router.get("/{ticker}/history", response_model=StockHistoryResponse)
def get_history(
    request: Request,
    ticker: str = Depends(get_valid_ticker),
    start: str | None = Query(None, description="起日 YYYY-MM-DD；省略＝近 180 天"),
    end: str | None = Query(None, description="迄日 YYYY-MM-DD；省略＝最後交易日"),
) -> StockHistoryResponse:
    """單一標的的歷史預測回放（point-in-time 重算 vs 實際）；僅含已到期樣本。"""
    import pandas as pd

    last_day = last_completed_trading_day()
    try:
        end_d = date.fromisoformat(end) if end else last_day
        start_d = date.fromisoformat(start) if start else end_d - timedelta(days=180)
    except ValueError as exc:
        raise InvalidRangeError(f"日期格式需為 YYYY-MM-DD：start={start!r} end={end!r}") from exc
    if start_d >= end_d:
        raise InvalidRangeError(f"起日需早於迄日：{start_d} ~ {end_d}")

    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return StockHistoryResponse(
            ticker=ticker, start=start_d, end=end_d, count=0, hits=0, records=[]
        )

    # 歷史回放讀本地全歷史快取，不打網路
    provider = YFinanceProvider(
        cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST, max_cache_age_days=9999
    )
    fetch_start = context_fetch_start(start_d)
    # 多抓 end 之後約 3 週：邊界附近的預測才算得出實際 5 日走勢（否則末幾日誤判為未到期）
    fetch_end = end_d + timedelta(days=21)
    try:
        pool_ohlcv, context = load_pool_context(provider, fetch_start, fetch_end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc
    df = pool_ohlcv.get(ticker)
    if df is None:
        try:
            df = provider.get_ohlcv(ticker, fetch_start, fetch_end)
        except DataProviderError as exc:
            raise DataSourceUnavailableError(str(exc)) from exc

    series = history_series(
        predictor, df, context, pd.Timestamp(start_d), pd.Timestamp(end_d)
    )
    hits = sum(1 for r in series if r["hit"])
    return StockHistoryResponse(
        ticker=ticker,
        start=start_d,
        end=end_d,
        count=len(series),
        hits=hits,
        hit_rate=(hits / len(series)) if series else None,
        records=[HistoryRecord(**r) for r in series],
    )
