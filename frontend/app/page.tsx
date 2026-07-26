"use client";

/**
 * 主儀表板：代號搜尋 → K 線圖 ＋ 預測卡（三類機率）＋ 熱門標的總覽 ＋ 技術指標
 * ＋ 大盤情境 ＋ 新聞情緒（跨專案）＋ 線上預測實證 ＋ 模型資訊列。
 * 主資料（K 線/預測）失敗顯示錯誤橫幅；次要卡片各自降級，不影響主頁。
 */

import { useCallback, useEffect, useState } from "react";
import CandleChart from "@/components/CandleChart";
import IndicatorPanel from "@/components/IndicatorPanel";
import HistoryReplayCard from "@/components/HistoryReplayCard";
import MarketCard from "@/components/MarketCard";
import NavTabs from "@/components/NavTabs";
import PredictionCard from "@/components/PredictionCard";
import SentimentCard from "@/components/SentimentCard";
import ThemeToggle from "@/components/ThemeToggle";
import TickerSearch from "@/components/TickerSearch";
import TrackRecordCard from "@/components/TrackRecordCard";
import WatchlistSignals from "@/components/WatchlistSignals";
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

function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

/** K 線自訂日期區間；套用後覆蓋預設 range，清除則回預設。 */
function CustomRangeRow({
  custom,
  onApply,
}: {
  custom: { start: string; end: string } | null;
  onApply: (v: { start: string; end: string } | null) => void;
}) {
  const [start, setStart] = useState(custom?.start ?? isoDaysAgo(365));
  const [end, setEnd] = useState(custom?.end ?? isoDaysAgo(0));
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-ink-3">
      <span>自訂區間</span>
      <input
        type="date"
        value={start}
        onChange={(e) => setStart(e.target.value)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-ink outline-none focus:border-accent"
      />
      <span>~</span>
      <input
        type="date"
        value={end}
        onChange={(e) => setEnd(e.target.value)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-ink outline-none focus:border-accent"
      />
      <button
        type="button"
        onClick={() => onApply({ start, end })}
        className="rounded-md bg-accent px-2.5 py-1 font-semibold text-accent-fg transition hover:opacity-90"
      >
        套用
      </button>
      {custom && (
        <button
          type="button"
          onClick={() => onApply(null)}
          className="rounded-md bg-surface-2 px-2.5 py-1 font-medium text-ink-2 transition hover:text-ink"
        >
          清除
        </button>
      )}
    </div>
  );
}

function Card({ title, children, className = "", action }: {
  title?: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <section className={`rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm) ${className}`}>
      {title && (
        <div className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-ink-2">{title}</h2>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export default function Home() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [ticker, setTicker] = useState("2330.TW");
  const [range, setRange] = useState<(typeof RANGES)[number]>("1y");
  const [custom, setCustom] = useState<{ start: string; end: string } | null>(null);
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

  const load = useCallback(
    async (t: string, r: string, c: { start: string; end: string } | null) => {
      setLoading(true);
      setError(null);
      api.indicators(t).then(setIndicators).catch(() => setIndicators(null));
      api.predictionHistory(t).then((h) => setHistory(h.records)).catch(() => setHistory([]));
      try {
        const candlesReq = c ? api.candles(t, r, c.start, c.end) : api.candles(t, r);
        const [cd, p] = await Promise.all([candlesReq, api.prediction(t)]);
        setCandles(cd.candles);
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
    },
    [],
  );

  useEffect(() => {
    load(ticker, range, custom);
  }, [ticker, range, custom, load]);

  const stockName = stocks.find((s) => s.ticker === ticker)?.name ?? "";

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
      <header className="mb-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-accent-fg shadow-(--shadow-md)">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v18h18" />
                <path d="M19 9l-5 5-4-4-3 3" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight sm:text-2xl">台股趨勢預測助理</h1>
              <p className="mt-0.5 text-sm text-ink-3">深度學習 · 技術指標 · 市場情境 · 台灣 50</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <NavTabs current="dashboard" />
            <ThemeToggle />
          </div>
        </div>
        <TickerSearch stocks={stocks} selected={ticker} onSelect={setTicker} />
      </header>

      {error && (
        <div
          className="mb-6 rounded-xl border px-4 py-3 text-sm"
          style={{ borderColor: "var(--up)", background: "var(--up-soft)", color: "var(--up)" }}
        >
          {error}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[2fr_1fr]">
        <Card
          className="animate-fadeup"
          title={`${stockName} ${ticker}`}
          action={
            <div className="flex gap-1">
              {RANGES.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => {
                    setCustom(null);
                    setRange(r);
                  }}
                  className="rounded-md px-2.5 py-1 text-xs font-medium transition"
                  style={
                    r === range && !custom
                      ? { background: "var(--accent)", color: "var(--accent-fg)" }
                      : { background: "var(--surface-2)", color: "var(--ink-2)" }
                  }
                >
                  {r}
                </button>
              ))}
            </div>
          }
        >
          <CustomRangeRow custom={custom} onApply={setCustom} />
          {loading ? (
            <div className="flex h-[380px] items-center justify-center text-sm text-ink-3">載入中…</div>
          ) : candles ? (
            <CandleChart candles={candles} />
          ) : !error ? (
            <div className="flex h-[380px] items-center justify-center text-sm text-ink-3">無資料</div>
          ) : null}
        </Card>

        <aside className="space-y-5">
          {!loading && prediction && (
            <div className="animate-fadeup">
              <PredictionCard prediction={prediction} testAuc={modelInfo?.test_auc ?? null} />
            </div>
          )}
          {loading && (
            <div className="flex h-64 items-center justify-center rounded-2xl border border-border text-sm text-ink-3">
              載入中…
            </div>
          )}
        </aside>
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
        <Card title="熱門標的預測總覽" className="animate-fadeup md:col-span-2 lg:col-span-1">
          <WatchlistSignals stocks={stocks} selected={ticker} onSelect={setTicker} />
        </Card>
        {indicators && <div className="animate-fadeup"><IndicatorPanel indicators={indicators} /></div>}
        {market && <div className="animate-fadeup"><MarketCard market={market} /></div>}
        <div className="animate-fadeup"><SentimentCard twTicker={ticker} /></div>
      </div>

      <div className="mt-5 animate-fadeup">
        <TrackRecordCard ticker={ticker} records={history} trackRecord={trackRecord} />
      </div>

      <div className="mt-5 animate-fadeup">
        <HistoryReplayCard ticker={ticker} />
      </div>

      {modelInfo && !modelInfo.is_mock && (
        <div className="mt-5 flex flex-wrap items-center justify-center gap-x-6 gap-y-1 rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-[11px] text-ink-3">
          <span>模型版本 {modelInfo.model_version}</span>
          {modelInfo.test_auc != null && <span>測試集 Macro AUC {modelInfo.test_auc.toFixed(4)}</span>}
          <span>決策規則：方向訊號信心未達門檻時轉為觀望（驗證期校準）</span>
        </div>
      )}

      <footer className="mt-8 text-center text-xs text-ink-3">
        彰師大 115 年百萬專題探索 — 基於深度學習之股價趨勢預測與投資助理系統
      </footer>
    </main>
  );
}
