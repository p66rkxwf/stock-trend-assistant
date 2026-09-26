/**
 * news-sentiment-monitor 後端 client（跨專案整合，:8001）。
 * 情緒模型只吃英文財經新聞，台股僅少數有美股 ADR 對照；無對照者以 QQQ
 * 呈現「美股科技大盤參考情緒」。此服務未啟動時由呼叫端降級顯示，不影響主頁。
 */

const NEWS_API_BASE = process.env.NEXT_PUBLIC_NEWS_API_BASE ?? "http://localhost:8001";

export type SentimentLabel = "negative" | "neutral" | "positive";

export interface KeywordScore {
  word: string;
  score: number;
}

export interface SentimentResponse {
  ticker: string;
  score: number; // [-1, +1]
  label: SentimentLabel;
  article_count: number;
  keywords: KeywordScore[];
  model_version: string;
  as_of: string;
  stale: boolean;
  is_mock: boolean;
}

/** 台股 → 美股 ADR 對照（只收流動性足、有英文新聞覆蓋的） */
const ADR_MAP: Record<string, string> = {
  "2330.TW": "TSM",
  "2303.TW": "UMC",
  "3711.TW": "ASX",
  "2412.TW": "CHT",
};

const MARKET_PROXY = "QQQ";

export function sentimentTickerFor(twTicker: string): { ticker: string; isProxy: boolean } {
  const adr = ADR_MAP[twTicker];
  return adr ? { ticker: adr, isProxy: false } : { ticker: MARKET_PROXY, isProxy: true };
}

const STATIC_DATA = process.env.NEXT_PUBLIC_STATIC_DATA === "1";

export async function fetchSentiment(usTicker: string): Promise<SentimentResponse> {
  // 靜態站：讀新聞站每日匯出的 JSON（news.sekinv.com 以 _headers 開放跨站讀取 /data/*）
  const path = STATIC_DATA
    ? `/data/stocks/${encodeURIComponent(usTicker)}/sentiment.json`
    : `/api/stocks/${encodeURIComponent(usTicker)}/sentiment`;
  const res = await fetch(`${NEWS_API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<SentimentResponse>;
}
