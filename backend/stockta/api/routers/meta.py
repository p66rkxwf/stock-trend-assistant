import sqlite3
from datetime import date

from fastapi import APIRouter, Request

from stockta.api.errors import DataSourceUnavailableError
from stockta.api.schemas import (
    HealthResponse,
    MarketResponse,
    ModelInfoResponse,
    StockInfo,
    StockListResponse,
    TrackRecordResponse,
)
from stockta.config import DATA_CACHE_DIR, PREDICTIONS_DB_PATH, STOCK_POOL
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError
from stockta.ml.report_predictions import actual_signal

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(status="ok", model_loaded=predictor is not None)


@router.get("/api/stocks", response_model=StockListResponse)
def list_stocks() -> StockListResponse:
    return StockListResponse(
        stocks=[StockInfo(ticker=ticker, name=name) for ticker, name in STOCK_POOL.items()]
    )


@router.get("/api/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return ModelInfoResponse(model_version="mock-0.0.0", is_mock=True)
    meta = predictor.metadata
    return ModelInfoResponse(
        model_version=predictor.version,
        trained_at=meta.get("trained_at"),
        test_auc=meta.get("metrics", {}).get("test", {}).get("macro_auc"),
        is_mock=False,
    )


@router.get("/api/market", response_model=MarketResponse)
def market_snapshot(request: Request) -> MarketResponse:
    """大盤情境快照（^TWII + 股票池寬度）——與模型特徵同一條 context 路徑。"""
    end = last_completed_trading_day()
    try:
        context = request.app.state.market_context.get(end)
    except DataProviderError as exc:
        raise DataSourceUnavailableError(str(exc)) from exc

    row = context.iloc[-1]
    return MarketResponse(
        as_of=context.index[-1].date(),
        ret_1d=float(row["mkt_ret_1d"]),
        ret_5d=float(row["mkt_ret_5d"]),
        ma20_bias=float(row["mkt_ma20"]),
        vol20=float(row["mkt_vol20"]),
        breadth_up=float(row["breadth_up"]),
        breadth_ma5=float(row["breadth_ma5"]),
    )


@router.get("/api/track-record", response_model=TrackRecordResponse)
def track_record() -> TrackRecordResponse:
    """全站線上實證摘要：predictions.db 已到期預測的命中統計（即時計算）。"""
    try:
        conn = sqlite3.connect(PREDICTIONS_DB_PATH)
        rows = conn.execute(
            "SELECT ticker, base_date, signal FROM predictions ORDER BY base_date"
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        return TrackRecordResponse(total=0, matured=0, hits=0)

    cache = ParquetCache(DATA_CACHE_DIR)
    matured = hits = 0
    for ticker, base_date, signal in rows:
        actual, _ = actual_signal(cache, ticker, base_date)
        if actual is None:
            continue
        matured += 1
        hits += actual == signal
    return TrackRecordResponse(
        total=len(rows),
        matured=matured,
        hits=hits,
        hit_rate=(hits / matured) if matured else None,
        since=date.fromisoformat(rows[0][1]),
    )
