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
    proba: dict[str, float] | None = Field(
        default=None, description="三類機率（跌/觀望/漲）；mock 模式為 None。additive 欄位，不破壞既有契約"
    )


class ScanResult(BaseModel):
    """全池掃描的單一標的結果。歷史日期查詢時附實際 5 日結果。"""

    ticker: str
    name: str
    signal: Signal
    confidence: float = Field(ge=0, le=1)
    risk: RiskLevel | None = None
    proba: dict[str, float]
    actual: Signal | None = Field(default=None, description="歷史查詢且已到期時的實際 5 日趨勢")
    actual_return: float | None = None
    hit: bool | None = None


class ScanResponse(BaseModel):
    """全股票池掃描：對每檔以 production 模型推論的趨勢訊號。
    無 date 參數＝即時掃描；帶 date＝歷史某日的 point-in-time 重算（附實際結果）。"""

    base_date: date = Field(description="推論所依據的最後一個交易日")
    model_version: str
    is_historical: bool = Field(default=False)
    up: int = Field(description="訊號為「漲」的檔數")
    hold: int = Field(description="訊號為「觀望」的檔數")
    down: int = Field(description="訊號為「跌」的檔數")
    matured: int = Field(default=0, description="歷史查詢中已到期（可對照實際）的檔數")
    hits: int = Field(default=0, description="其中命中的檔數")
    results: list[ScanResult]
    is_mock: bool = Field(default=False)


class IndicatorsResponse(BaseModel):
    """當前技術指標快照（與模型特徵同一條 build_features 路徑，數值一致）。"""

    ticker: str
    as_of: date
    rsi14: float = Field(description="0–1 縮放（0.5 中性）")
    kd_k: float
    kd_d: float
    macd_hist: float
    bb_pctb: float
    bb_width: float
    vol_ratio: float = Field(description="量能相對 20 日均量的偏離")
    ma_bias_5: float = Field(description="收盤價相對 5 日均線乖離")
    ma_bias_20: float
    ma_bias_60: float


class MarketResponse(BaseModel):
    """大盤情境快照（^TWII + 股票池寬度）。"""

    as_of: date
    ret_1d: float
    ret_5d: float
    ma20_bias: float = Field(description="加權指數相對其 20 日均線乖離")
    vol20: float = Field(description="日報酬 20 日標準差")
    breadth_up: float = Field(description="股票池當日上漲家數比 0–1")
    breadth_ma5: float


class PastPrediction(BaseModel):
    base_date: date
    signal: Signal
    confidence: float
    model_version: str
    actual: Signal | None = Field(default=None, description="未到期（不足 5 個交易日）為 None")
    actual_return: float | None = None
    hit: bool | None = None


class PredictionHistoryResponse(BaseModel):
    ticker: str
    records: list[PastPrediction]


class HistoryRecord(BaseModel):
    date: date
    signal: Signal
    confidence: float
    actual: Signal
    actual_return: float | None = None
    hit: bool


class StockHistoryResponse(BaseModel):
    """單一標的的歷史預測回放（point-in-time 重算 vs 實際），僅含已到期樣本。"""

    ticker: str
    start: date
    end: date
    count: int
    hits: int
    hit_rate: float | None = None
    records: list[HistoryRecord]


class TrackRecordResponse(BaseModel):
    """全站線上實證摘要：predictions.db 已到期預測的即時命中統計。"""

    total: int
    matured: int
    hits: int
    hit_rate: float | None = Field(default=None, description="matured=0 時為 None")
    since: date | None = None


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
