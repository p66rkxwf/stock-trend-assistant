"use client";

import { useEffect, useState } from "react";
import { fetchSentiment, sentimentTickerFor, type SentimentResponse } from "@/lib/newsApi";

// 於台股脈絡下對齊台股慣例：正面情緒＝偏多＝紅，負面＝偏空＝綠（與 news 專案相反，卡上註明）
const LABEL: Record<SentimentResponse["label"], { text: string; varName: string }> = {
  positive: { text: "偏多", varName: "--up" },
  negative: { text: "偏空", varName: "--down" },
  neutral: { text: "中性", varName: "--hold" },
};

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(((score + 1) / 2) * 100);
  return (
    <div
      className="relative mt-1.5 h-2.5 w-full overflow-hidden rounded-full"
      style={{ background: "linear-gradient(90deg, var(--down-soft), var(--surface-2), var(--up-soft))" }}
    >
      <div className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded bg-ink" style={{ left: `calc(${pct}% - 2px)` }} />
    </div>
  );
}

/** 新聞情緒卡（跨專案整合：news-sentiment-monitor 提供分析）。服務未啟動時灰卡降級。 */
export default function SentimentCard({ twTicker }: { twTicker: string }) {
  const [sentiment, setSentiment] = useState<SentimentResponse | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const { ticker: usTicker, isProxy } = sentimentTickerFor(twTicker);

  useEffect(() => {
    let cancelled = false;
    setSentiment(null);
    setUnavailable(false);
    fetchSentiment(usTicker)
      .then((s) => !cancelled && setSentiment(s))
      .catch(() => !cancelled && setUnavailable(true));
    return () => {
      cancelled = true;
    };
  }, [usTicker]);

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink-2">新聞情緒</h2>
        <span className="text-[10px] text-ink-3">{isProxy ? "美股科技大盤參考" : `ADR ${usTicker}`}</span>
      </div>

      {unavailable && (
        <p className="mt-4 rounded-lg bg-surface-2 px-3 py-2 text-xs text-ink-3">
          {process.env.NEXT_PUBLIC_STATIC_DATA === "1"
            ? "情緒資料暫時無法取得。"
            : "情緒服務未啟動（news-sentiment-monitor 後端 :8001）。"}
        </p>
      )}

      {!unavailable && !sentiment && <p className="mt-4 text-sm text-ink-3">分析中…</p>}

      {sentiment && (
        <>
          <div className="mt-3 flex items-end gap-3">
            <span className="text-3xl font-bold" style={{ color: `var(${LABEL[sentiment.label].varName})` }}>
              {LABEL[sentiment.label].text}
            </span>
            <span className="mb-0.5 text-sm tabular-nums text-ink-3">
              指數 {sentiment.score >= 0 ? "+" : ""}
              {sentiment.score.toFixed(2)}・{sentiment.article_count} 則
            </span>
          </div>
          <ScoreBar score={sentiment.score} />

          {sentiment.keywords.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {sentiment.keywords.slice(0, 5).map((k) => (
                <span
                  key={k.word}
                  className="rounded-full px-2 py-0.5 text-[10px]"
                  style={{ background: "var(--surface-2)", color: "var(--ink-2)" }}
                >
                  {k.word}
                </span>
              ))}
            </div>
          )}

          {(sentiment.is_mock || sentiment.stale) && (
            <p className="mt-2 text-[10px]" style={{ color: "var(--accent)" }}>
              {sentiment.is_mock ? "⚠ 情緒模型未載入（示意資料）" : "⚠ 新聞源暫時失效，顯示快取"}
            </p>
          )}
        </>
      )}

      <p className="mt-3 border-t border-border pt-2 text-[10px] leading-relaxed text-ink-3">
        由本團隊 news-sentiment-monitor（BERT 財經新聞情緒分析）提供；情緒模型只涵蓋英文新聞，
        台股僅供跨市場參考、不作為預測模型輸入。此處正面＝偏多以對齊台股紅漲慣例。
      </p>
    </div>
  );
}
