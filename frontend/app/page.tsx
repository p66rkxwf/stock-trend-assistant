"use client";

/**
 * 主儀表板：代號搜尋 → K 線圖（range 切換）＋ 預測卡片（三類機率）＋ 技術指標
 * ＋ 大盤情境 ＋ 新聞情緒（跨專案）＋ 線上預測實證 ＋ 模型資訊列。
 * 主資料（K 線/預測）失敗顯示錯誤橫幅；次要卡片各自降級，不影響主頁。
 */

import { useCallback, useEffect, useState } from "react";
import CandleChart from "@/components/CandleChart";
import IndicatorPanel from "@/components/IndicatorPanel";
import MarketCard from "@/components/MarketCard";
import PredictionCard from "@/components/PredictionCard";
import SentimentCard from "@/components/SentimentCard";
import TickerSearch from "@/components/TickerSearch";
import TrackRecordCard from "@/components/TrackRecordCard";
import {
  api,
  ApiError,
  type Candle,
  type IndicatorsResponse,
  type MarketResponse,
  type ModelInfoResponse,
  type PastPrediction,
  type PredictionResponse,
  type StockInfo,
  type TrackRecordResponse,
} from "@/lib/api";

const RANGES = ["1mo", "3mo", "6mo", "1y", "2y", "5y"] as const;

export default function Home() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [ticker, setTicker] = useState("2330.TW");
  const [range, setRange] = useState<(typeof RANGES)[number]>("1y");
  const [candles, setCandles] = useState<Candle[] | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [indicators, setIndicators] = useState<IndicatorsResponse | null>(null);
  const [market, setMarket] = useState<MarketResponse | null>(null);
  const [history, setHistory] = useState<PastPrediction[]>([]);
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.stocks().then((r) => setStocks(r.stocks)).catch(() => setStocks([]));
    api.modelInfo().then(setModelInfo).catch(() => setModelInfo(null));
    api.market().then(setMarket).catch(() => setMarket(null));
    api.trackRecord().then(setTrackRecord).catch(() => setTrackRecord(null));
  }, []);

  const load = useCallback(async (t: string, r: string) => {
    setLoading(true);
    setError(null);
    // 次要資料各自載入，失敗只隱藏對應卡片
    api.indicators(t).then(setIndicators).catch(() => setIndicators(null));
    api.predictionHistory(t).then((h) => setHistory(h.records)).catch(() => setHistory([]));
    try {
      const [c, p] = await Promise.all([api.candles(t, r), api.prediction(t)]);
      setCandles(c.candles);
      setPrediction(p);
    } catch (e) {
      setCandles(null);
      setPrediction(null);
      if (e instanceof ApiError) {
        setError(e.status === 503 ? `資料源暫時無法使用：${e.message}` : e.message);
      } else {
        setError("發生未知錯誤");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(ticker, range);
  }, [ticker, range, load]);

  const stockName = stocks.find((s) => s.ticker === ticker)?.name ?? "";

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">台股趨勢預測助理</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            深度學習模型 × 技術指標 × 市場情境 × 台灣 50 成分股
          </p>
        </div>
        <TickerSearch stocks={stocks} selected={ticker} onSelect={setTicker} />
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold">
              {stockName} <span className="text-gray-400">{ticker}</span>
            </h2>
            <div className="flex gap-1">
              {RANGES.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRange(r)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    r === range
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
          {loading && (
            <div className="flex h-[380px] items-center justify-center text-sm text-gray-400">
              載入中…
            </div>
          )}
          {!loading && candles && <CandleChart candles={candles} />}
          {!loading && !candles && !error && (
            <div className="flex h-[380px] items-center justify-center text-sm text-gray-400">
              無資料
            </div>
          )}
        </section>

        <aside className="space-y-4">
          {loading && (
            <div className="flex h-64 items-center justify-center rounded-xl border border-gray-200 text-sm text-gray-400 dark:border-gray-700">
              載入中…
            </div>
          )}
          {!loading && prediction && (
            <PredictionCard prediction={prediction} testAuc={modelInfo?.test_auc ?? null} />
          )}
        </aside>
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-3">
        {indicators && <IndicatorPanel indicators={indicators} />}
        {market && <MarketCard market={market} />}
        <SentimentCard twTicker={ticker} />
      </div>

      <div className="mt-6">
        <TrackRecordCard ticker={ticker} records={history} trackRecord={trackRecord} />
      </div>

      {modelInfo && !modelInfo.is_mock && (
        <div className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-1 rounded-lg bg-gray-50 px-4 py-2.5 text-[11px] text-gray-500 dark:bg-gray-900 dark:text-gray-400">
          <span>模型版本 {modelInfo.model_version}</span>
          {modelInfo.test_auc != null && <span>測試集 Macro AUC {modelInfo.test_auc.toFixed(4)}</span>}
          <span>決策規則：方向訊號信心未達門檻時轉為觀望（驗證期校準）</span>
        </div>
      )}

      <footer className="mt-8 text-center text-xs text-gray-400 dark:text-gray-600">
        彰師大 115 年百萬專題探索 — 基於深度學習之股價趨勢預測與投資助理系統
      </footer>
    </main>
  );
}
