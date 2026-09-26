/**
 * 後端 API client — 型別對應 backend/stockta/api/schemas.py（Phase 0 凍結契約）。
 * 錯誤格式統一為 {"error": {"code", "message"}}；以 code 判斷錯誤類型，勿比對 message。
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/**
 * 靜態站模式（Cloudflare Pages）：沒有後端，改讀每日排程匯出的 /data/*.json
 * （backend/stockta/export_static.py）。帶參數的端點由下方 staticApi 在瀏覽器端切片，
 * 規則逐一對照後端路由；本機開發不設此變數，照舊呼叫 uvicorn。
 */
export const STATIC_DATA = process.env.NEXT_PUBLIC_STATIC_DATA === "1";
const DATA_BASE = "/data";

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

export interface HistoryRecord {
  date: string;
  signal: Signal;
  confidence: number;
  actual: Signal;
  actual_return: number | null;
  hit: boolean;
}

export interface StockHistoryResponse {
  ticker: string;
  start: string;
  end: string;
  count: number;
  hits: number;
  hit_rate: number | null;
  records: HistoryRecord[];
}

export interface RankResult {
  ticker: string;
  name: string;
  score: number;
  rank: number;
  percentile: number;
  quantile: "top" | "mid" | "bottom";
}

export interface RankResponse {
  base_date: string;
  model: string;
  is_historical: boolean;
  results: RankResult[];
  is_mock: boolean;
}

export interface RankSummaryResponse {
  available: boolean;
  model: string | null;
  test_rank_ic: number | null;
  test_rank_ic_t: number | null;
  val_rank_ic: number | null;
  holding_days: number | null;
  net_cum: number | null;
  bench_cum: number | null;
  net_excess_cum: number | null;
  win_rate: number | null;
  cost_bps: number | null;
}

export interface ScanResult {
  ticker: string;
  name: string;
  signal: Signal;
  confidence: number;
  risk: RiskLevel | null;
  proba: Record<Signal, number>;
  actual: Signal | null;
  actual_return: number | null;
  hit: boolean | null;
}

export interface ScanResponse {
  base_date: string;
  model_version: string;
  is_historical: boolean;
  up: number;
  hold: number;
  down: number;
  matured: number;
  hits: number;
  results: ScanResult[];
  is_mock: boolean;
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

async function request<T>(path: string, base: string = API_BASE): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${base}${path}`);
  } catch {
    throw new ApiError(
      0,
      "NETWORK_ERROR",
      STATIC_DATA ? "資料暫時無法取得，請稍後再試" : "無法連線到後端服務，請確認 API 已啟動",
    );
  }
  if (!res.ok) {
    if (STATIC_DATA && res.status === 404) {
      throw new ApiError(404, "NOT_FOUND", "此項資料今日未產生（每日排程匯出時失敗）");
    }
    const body = await res.json().catch(() => null);
    const err = body?.error;
    throw new ApiError(res.status, err?.code ?? "UNKNOWN", err?.message ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const liveApi = {
  stocks: () => request<{ stocks: StockInfo[] }>("/api/stocks"),
  candles: (ticker: string, range: string, start?: string, end?: string) => {
    const q =
      start && end
        ? `start=${start}&end=${end}`
        : `range=${range}`;
    return request<CandlesResponse>(`/api/stocks/${encodeURIComponent(ticker)}/candles?${q}`);
  },
  prediction: (ticker: string) =>
    request<PredictionResponse>(`/api/stocks/${encodeURIComponent(ticker)}/prediction`),
  modelInfo: () => request<ModelInfoResponse>("/api/model"),
  indicators: (ticker: string) =>
    request<IndicatorsResponse>(`/api/stocks/${encodeURIComponent(ticker)}/indicators`),
  market: () => request<MarketResponse>("/api/market"),
  predictionHistory: (ticker: string) =>
    request<PredictionHistoryResponse>(`/api/stocks/${encodeURIComponent(ticker)}/predictions`),
  trackRecord: () => request<TrackRecordResponse>("/api/track-record"),
  scan: (date?: string) =>
    request<ScanResponse>(`/api/scan${date ? `?date=${encodeURIComponent(date)}` : ""}`),
  rank: (date?: string) =>
    request<RankResponse>(`/api/rank${date ? `?date=${encodeURIComponent(date)}` : ""}`),
  rankSummary: () => request<RankSummaryResponse>("/api/rank/summary"),
  stockHistory: (ticker: string, start: string, end: string) =>
    request<StockHistoryResponse>(
      `/api/stocks/${encodeURIComponent(ticker)}/history?start=${start}&end=${end}`,
    ),
};

/** 靜態站的匯出摘要（export_static.py 的 meta.json）。 */
export interface SiteMeta {
  generated_at: string; // ISO UTC
  last_trading_day: string; // API 推論錨點；candles 區間的終點
  data_as_of: string; // 實際最後一根 K 棒
  model_version: string;
  cs_model: string | null;
  range_days: Record<string, number>;
  candles_start: string;
  history_start: string;
}

interface DatedIndex {
  latest: string;
  sessions: string[];
  dates: string[];
}

const staticGet = <T>(path: string) => request<T>(path, DATA_BASE);
const enc = encodeURIComponent;

let metaPromise: Promise<SiteMeta> | null = null;
export function siteMeta(): Promise<SiteMeta> {
  metaPromise ??= staticGet<SiteMeta>("/meta.json").catch((e) => {
    metaPromise = null; // 失敗不快取，下次重試
    throw e;
  });
  return metaPromise;
}

function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function assertOrdered(start: string, end: string) {
  if (start >= end) throw new ApiError(422, "INVALID_RANGE", `起日需早於迄日：${start} ~ ${end}`);
}

/** scan/rank 查歷史某日：同 API，≥ 錨點日走即時；否則對齊到 ≤ 查詢日的最後一個交易日。 */
async function staticDated<T>(kind: "scan" | "rank", date?: string): Promise<T> {
  if (!date) return staticGet<T>(`/${kind}/latest.json`);
  const [meta, idx] = await Promise.all([siteMeta(), staticGet<DatedIndex>(`/${kind}/index.json`)]);
  if (date >= meta.last_trading_day) return staticGet<T>(`/${kind}/latest.json`);
  const session = [...idx.sessions].reverse().find((d) => d <= date);
  if (!session) {
    throw new ApiError(422, "OUT_OF_RANGE", `靜態站保留最近 ${idx.sessions.length} 個交易日（${idx.sessions[0]} 起）`);
  }
  if (!idx.dates.includes(session)) {
    throw new ApiError(404, "NOT_FOUND", `${session} 的資料今日未產生（每日排程匯出時失敗）`);
  }
  return staticGet<T>(`/${kind}/${session}.json`);
}

const staticApi: typeof liveApi = {
  stocks: () => staticGet("/stocks.json"),
  candles: async (ticker, range, start, end) => {
    const [meta, full] = await Promise.all([
      siteMeta(),
      staticGet<CandlesResponse>(`/stocks/${enc(ticker)}/candles.json`),
    ]);
    let lo: string;
    let hi: string;
    if (start && end) {
      assertOrdered(start, end);
      if (start < meta.candles_start) {
        throw new ApiError(422, "OUT_OF_RANGE", `靜態站 K 線僅提供 ${meta.candles_start} 之後的資料`);
      }
      [lo, hi] = [start, end];
    } else {
      const days = meta.range_days[range];
      if (days == null) throw new ApiError(422, "INVALID_RANGE", `不支援的 range 參數: ${range}`);
      hi = meta.last_trading_day;
      lo = addDays(hi, -days);
    }
    // 後端 _slice 兩端皆含
    return { ticker: full.ticker, candles: full.candles.filter((c) => c.time >= lo && c.time <= hi) };
  },
  prediction: (ticker) => staticGet(`/stocks/${enc(ticker)}/prediction.json`),
  modelInfo: () => staticGet("/model.json"),
  indicators: (ticker) => staticGet(`/stocks/${enc(ticker)}/indicators.json`),
  market: () => staticGet("/market.json"),
  predictionHistory: (ticker) => staticGet(`/stocks/${enc(ticker)}/predictions.json`),
  trackRecord: () => staticGet("/track-record.json"),
  scan: (date) => staticDated("scan", date),
  rank: (date) => staticDated("rank", date),
  rankSummary: () => staticGet("/rank/summary.json"),
  stockHistory: async (ticker, start, end) => {
    const [meta, full] = await Promise.all([
      siteMeta(),
      staticGet<StockHistoryResponse>(`/stocks/${enc(ticker)}/history.json`),
    ]);
    assertOrdered(start, end);
    if (start < meta.history_start) {
      throw new ApiError(422, "OUT_OF_RANGE", `靜態站僅提供 ${meta.history_start} 之後的歷史回放`);
    }
    // 後端 history_series 取 (start, end]、新到舊；命中率依篩選後重算。
    // 匯出檔從 history_start 起算指標暖機，與以使用者 start 起算的差異在
    // test_feature_parity 的 1e-4 容忍度內。
    const records = full.records.filter((r) => r.date > start && r.date <= end);
    const hits = records.filter((r) => r.hit).length;
    return {
      ticker: full.ticker,
      start,
      end,
      count: records.length,
      hits,
      hit_rate: records.length ? hits / records.length : null,
      records,
    };
  },
};

export const api = STATIC_DATA ? staticApi : liveApi;
