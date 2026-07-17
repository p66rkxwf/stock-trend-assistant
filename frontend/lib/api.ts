/**
 * 後端 API client — 型別對應 backend/stockta/api/schemas.py（Phase 0 凍結契約）。
 * 錯誤格式統一為 {"error": {"code", "message"}}；以 code 判斷錯誤類型，勿比對 message。
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type Signal = "漲" | "跌" | "觀望";
export type RiskLevel = "低" | "中" | "高";

export interface Candle {
  time: string; // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CandlesResponse {
  ticker: string;
  candles: Candle[];
}

export interface PredictionResponse {
  ticker: string;
  base_date: string;
  signal: Signal;
  confidence: number;
  risk: RiskLevel;
  model_version: string;
  is_mock: boolean;
  proba: Record<Signal, number> | null;
}

export interface IndicatorsResponse {
  ticker: string;
  as_of: string;
  rsi14: number;
  kd_k: number;
  kd_d: number;
  macd_hist: number;
  bb_pctb: number;
  bb_width: number;
  vol_ratio: number;
  ma_bias_5: number;
  ma_bias_20: number;
  ma_bias_60: number;
}

export interface MarketResponse {
  as_of: string;
  ret_1d: number;
  ret_5d: number;
  ma20_bias: number;
  vol20: number;
  breadth_up: number;
  breadth_ma5: number;
}

export interface PastPrediction {
  base_date: string;
  signal: Signal;
  confidence: number;
  model_version: string;
  actual: Signal | null;
  actual_return: number | null;
  hit: boolean | null;
}

export interface PredictionHistoryResponse {
  ticker: string;
  records: PastPrediction[];
}

export interface TrackRecordResponse {
  total: number;
  matured: number;
  hits: number;
  hit_rate: number | null;
  since: string | null;
}

export interface StockInfo {
  ticker: string;
  name: string;
}

export interface ModelInfoResponse {
  model_version: string;
  trained_at: string | null;
  test_auc: number | null;
  is_mock: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`);
  } catch {
    throw new ApiError(0, "NETWORK_ERROR", "無法連線到後端服務，請確認 API 已啟動");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const err = body?.error;
    throw new ApiError(res.status, err?.code ?? "UNKNOWN", err?.message ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  stocks: () => request<{ stocks: StockInfo[] }>("/api/stocks"),
  candles: (ticker: string, range: string) =>
    request<CandlesResponse>(`/api/stocks/${encodeURIComponent(ticker)}/candles?range=${range}`),
  prediction: (ticker: string) =>
    request<PredictionResponse>(`/api/stocks/${encodeURIComponent(ticker)}/prediction`),
  modelInfo: () => request<ModelInfoResponse>("/api/model"),
  indicators: (ticker: string) =>
    request<IndicatorsResponse>(`/api/stocks/${encodeURIComponent(ticker)}/indicators`),
  market: () => request<MarketResponse>("/api/market"),
  predictionHistory: (ticker: string) =>
    request<PredictionHistoryResponse>(`/api/stocks/${encodeURIComponent(ticker)}/predictions`),
  trackRecord: () => request<TrackRecordResponse>("/api/track-record"),
};
