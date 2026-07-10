from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stockta.api.errors import ApiError, api_error_handler
from stockta.api.routers import meta, stocks
from stockta.config import AUTO_ADJUST, DATA_CACHE_DIR
from stockta.data.cache import ParquetCache
from stockta.data.provider import YFinanceProvider

# demo 環境明確 origin 清單，勿用 "*"（見 PLAN.md 安全性章節）
ALLOWED_ORIGINS = ["http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = ParquetCache(DATA_CACHE_DIR)
    app.state.data_provider = YFinanceProvider(cache=cache, auto_adjust=AUTO_ADJUST)
    # Phase 6：於此以 stockta/ml/registry.py 載入模型 artifact 並掛到 app.state.predictor
    # Phase 6：加入 slowapi 限流，保護 yfinance 抓取不被放大攻擊
    yield


app = FastAPI(title="Stock Trend Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.add_exception_handler(ApiError, api_error_handler)

app.include_router(meta.router)
app.include_router(stocks.router)
