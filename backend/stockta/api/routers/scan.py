"""全股票池掃描：GET /api/scan（可選 ?date=YYYY-MM-DD 查歷史某日）

無 date：對每檔以 production 模型即時推論，回傳當前趨勢訊號總覽。
帶 date：對每檔做 point-in-time 重算（模型在該日會做的預測），並對照
實際 5 日走勢顯示命中與否（歷史回放，見 inference/replay.py）。

市場情境 context 全池共用；個股 OHLCV 走既有 parquet 快取（快取新鮮時不打網路），
與 /prediction 同一條推論路徑。唯讀分析端點：不寫入 predictions.db（線上實證的
落地由每日排程 record_predictions 統一負責，避免頁面載入產生寫入副作用）。
"""

from datetime import date, timedelta

from fastapi import APIRouter, Query, Request

from stockta.api.errors import DataSourceUnavailableError, InvalidRangeError
from stockta.api.schemas import ScanResponse, ScanResult
from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    LABEL_CLASSES,
    MARKET_INDEX_TICKER,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProvider, DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.inference.predictor import InsufficientDataError, Predictor
from stockta.inference.risk import annualized_volatility, risk_level
from stockta.inference.replay import scan_asof, signal_of
from stockta.ml.report_predictions import actual_signal

router = APIRouter(tags=["scan"])


def _live_scan(request: Request, predictor: Predictor, end: date) -> ScanResponse:
    provider: DataProvider = request.app.state.data_provider
    start = end - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS))
    try:
        context = request.app.state.market_context.get(end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc

    counts = {"漲": 0, "觀望": 0, "跌": 0}
    results: list[ScanResult] = []
    for ticker, name in STOCK_POOL.items():
        try:
            df = provider.get_ohlcv(ticker, start, end)
            pred = predictor.predict(df, context)
            risk = risk_level(annualized_volatility(df["close"]))
        except (DataProviderError, InsufficientDataError):
            continue
        counts[pred.signal] += 1
        results.append(
            ScanResult(
                ticker=ticker, name=name, signal=pred.signal, confidence=pred.confidence,
                risk=risk, proba=dict(zip(LABEL_CLASSES, pred.proba)),
            )
        )
    return ScanResponse(
        base_date=end, model_version=predictor.version, is_historical=False,
        up=counts["漲"], hold=counts["觀望"], down=counts["跌"], results=results,
    )


def _historical_scan(predictor: Predictor, as_of: date) -> ScanResponse:
    # 歷史回放：以本地快取全歷史重算，不打網路
    cache = ParquetCache(DATA_CACHE_DIR)
    fetch_start = as_of - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS) + 40)
    provider = YFinanceProvider(cache=cache, auto_adjust=AUTO_ADJUST, max_cache_age_days=9999)

    pool_ohlcv = {}
    for ticker in STOCK_POOL:
        try:
            pool_ohlcv[ticker] = provider.get_ohlcv(ticker, fetch_start, as_of + timedelta(days=20))
        except DataProviderError:
            continue
    try:
        market = provider.get_ohlcv(MARKET_INDEX_TICKER, fetch_start, as_of + timedelta(days=20))
    except DataProviderError as exc:
        raise DataSourceUnavailableError(f"歷史大盤資料不足：{exc}") from exc
    context = build_market_context(market, pool_ohlcv)

    counts = {"漲": 0, "觀望": 0, "跌": 0}
    matured = hits = 0
    base_date_seen: date | None = None
    results: list[ScanResult] = []
    for ticker, name in STOCK_POOL.items():
        df = pool_ohlcv.get(ticker)
        if df is None:
            continue
        out = scan_asof(predictor, df, context, as_of)
        if out is None:
            continue
        proba, base_date = out
        base_date_seen = base_date
        signal = signal_of(proba)
        counts[signal] += 1
        actual, ret = actual_signal(cache, ticker, base_date.isoformat())
        hit = None
        if actual is not None:
            matured += 1
            hit = actual == signal
            hits += int(hit)
        results.append(
            ScanResult(
                ticker=ticker, name=name, signal=signal, confidence=float(proba.max()),
                risk=None, proba=dict(zip(LABEL_CLASSES, (float(p) for p in proba))),
                actual=actual, actual_return=ret, hit=hit,
            )
        )
    if not results:
        raise InvalidRangeError(f"{as_of} 無足夠歷史資料可重算（過早或非交易日）")
    return ScanResponse(
        base_date=base_date_seen or as_of, model_version=predictor.version, is_historical=True,
        up=counts["漲"], hold=counts["觀望"], down=counts["跌"],
        matured=matured, hits=hits, results=results,
    )


@router.get("/api/scan", response_model=ScanResponse)
def scan(
    request: Request,
    date_param: str | None = Query(None, alias="date", description="歷史查詢日 YYYY-MM-DD；省略＝即時"),
) -> ScanResponse:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    end = last_completed_trading_day()
    if predictor is None:
        return ScanResponse(
            base_date=end, model_version="mock-0.0.0", up=0, hold=0, down=0, results=[], is_mock=True
        )
    if date_param is None:
        return _live_scan(request, predictor, end)
    try:
        as_of = date.fromisoformat(date_param)
    except ValueError as exc:
        raise InvalidRangeError(f"日期格式需為 YYYY-MM-DD：{date_param!r}") from exc
    if as_of >= end:
        return _live_scan(request, predictor, end)
    return _historical_scan(predictor, as_of)
