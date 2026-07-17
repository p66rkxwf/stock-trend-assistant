import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from stockta.api.errors import ApiError, api_error_handler
from stockta.api.routers import meta, stocks
from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    PREDICTIONS_DB_PATH,
    PRODUCTION_MODEL,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache
from stockta.data.provider import YFinanceProvider
from stockta.inference.market_context import MarketContextService
from stockta.inference.predictor import Predictor
from stockta.inference.store import PredictionStore
from stockta.ml.registry import ArtifactContractError

logger = logging.getLogger("stockta")

# demo 環境明確 origin 清單，勿用 "*"（見 PLAN.md 安全性章節）
ALLOWED_ORIGINS = ["http://localhost:3000"]

# 每 IP 限流，保護 yfinance 抓取不被放大攻擊（PLAN.md 安全性章節）
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = ParquetCache(DATA_CACHE_DIR)
    app.state.data_provider = YFinanceProvider(cache=cache, auto_adjust=AUTO_ADJUST)
    app.state.market_context = MarketContextService(
        app.state.data_provider, cache, list(STOCK_POOL)
    )

    try:
        app.state.predictor = Predictor.from_registry(PRODUCTION_MODEL)
        app.state.prediction_store = PredictionStore(PREDICTIONS_DB_PATH)
        logger.info("已載入模型 %s", app.state.predictor.version)
    except FileNotFoundError:
        # 尚未訓練（開發初期）：以 mock 模式提供 API，前端照常對接
        app.state.predictor = None
        app.state.prediction_store = None
        logger.warning("找不到模型 artifact（%s），/prediction 以 mock 模式運作", PRODUCTION_MODEL)
    except ArtifactContractError:
        # metadata 與程式碼不一致 = 模型會默默輸出錯誤預測，寧可拒絕啟動
        raise

    yield


app = FastAPI(title="Stock Trend Assistant API", lifespan=lifespan)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.add_exception_handler(ApiError, api_error_handler)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # 維持 Phase 0 凍結的統一錯誤格式
    return JSONResponse(
        status_code=429,
        content={"error": {"code": "RATE_LIMITED", "message": f"請求過於頻繁（{exc.detail}）"}},
    )


app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

app.include_router(meta.router)
app.include_router(stocks.router)
