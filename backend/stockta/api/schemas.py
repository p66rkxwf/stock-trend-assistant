"""Pydantic 回應模型 — Phase 0 凍結的 API 契約。

前端從第一週開始就對這份契約開發（先吃 mock 回應），後續 Phase 不應隨意
變更欄位名稱或型別；若真的需要變更，視為破壞性變更並同步通知前端。
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Signal = Literal["漲", "跌", "觀望"]
RiskLevel = Literal["低", "中", "高"]


class Candle(BaseModel):
    time: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class CandlesResponse(BaseModel):
    ticker: str
    candles: list[Candle]


class PredictionResponse(BaseModel):
    ticker: str
    base_date: date = Field(description="預測所依據的最後一個已完成交易日")
    signal: Signal
    confidence: float = Field(ge=0, le=1)
    risk: RiskLevel
    model_version: str
    is_mock: bool = Field(default=False, description="Phase 6 模型整合前為 True，前端可據此顯示提示")


class StockInfo(BaseModel):
    ticker: str
    name: str


class StockListResponse(BaseModel):
    stocks: list[StockInfo]


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str | None = None
    test_auc: float | None = None
    is_mock: bool = Field(default=False, description="Phase 6 前尚未整合真實模型時為 True")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
