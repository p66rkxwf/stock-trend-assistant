from fastapi import APIRouter, Request

from stockta.api.schemas import HealthResponse, ModelInfoResponse, StockInfo, StockListResponse
from stockta.config import STOCK_POOL

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
