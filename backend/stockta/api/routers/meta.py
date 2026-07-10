from fastapi import APIRouter

from stockta.api.schemas import HealthResponse, ModelInfoResponse, StockInfo, StockListResponse
from stockta.config import STOCK_POOL

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Phase 6 起改為回報 registry 載入的真實模型狀態
    return HealthResponse(status="ok", model_loaded=False)


@router.get("/api/stocks", response_model=StockListResponse)
def list_stocks() -> StockListResponse:
    return StockListResponse(
        stocks=[StockInfo(ticker=ticker, name=name) for ticker, name in STOCK_POOL.items()]
    )


@router.get("/api/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    # Phase 5 模型選定後，改由 stockta/ml/registry.py 讀取真實 metadata.json
    return ModelInfoResponse(model_version="mock-0.0.0", is_mock=True)
