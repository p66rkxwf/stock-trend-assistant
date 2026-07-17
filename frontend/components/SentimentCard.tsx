"use client";

import { useEffect, useState } from "react";
import { fetchSentiment, sentimentTickerFor, type SentimentResponse } from "@/lib/newsApi";

const LABEL_TEXT: Record<SentimentResponse["label"], { text: string; tone: string }> = {
  positive: { text: "偏多", tone: "text-red-600 dark:text-red-400" },
  negative: { text: "偏空", tone: "text-green-600 dark:text-green-400" },
  neutral: { text: "中性", tone: "text-gray-500 dark:text-gray-400" },
};

/** 情緒分數 [-1,+1] 橫桿 */
function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(((score + 1) / 2) * 100);
  return (
    <div className="relative mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-gradient-to-r from-green-200 via-gray-200 to-red-200 dark:from-green-900 dark:via-gray-700 dark:to-red-900">
      <div
        className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded bg-gray-900 dark:bg-white"
        style={{ left: `calc(${pct}% - 2px)` }}
      />
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
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">新聞情緒</h2>
        <span className="text-[10px] text-gray-400">
          {isProxy ? "美股科技大盤參考" : `ADR ${usTicker}`}
        </span>
      </div>

      {unavailable && (
        <p className="mt-4 rounded-md bg-gray-50 px-3 py-2 text-xs text-gray-400 dark:bg-gray-800/60">
          情緒服務未啟動（news-sentiment-monitor 後端 :8001）。
        </p>
      )}

      {!unavailable && !sentiment && (
        <p className="mt-4 text-sm text-gray-400">分析中…</p>
      )}

      {sentiment && (
        <>
          <div className="mt-3 flex items-end gap-3">
            <span className={`text-3xl font-bold ${LABEL_TEXT[sentiment.label].tone}`}>
              {LABEL_TEXT[sentiment.label].text}
            </span>
            <span className="mb-0.5 text-sm tabular-nums text-gray-500 dark:text-gray-400">
              指數 {sentiment.score >= 0 ? "+" : ""}
              {sentiment.score.toFixed(2)}・{sentiment.article_count} 則
            </span>
          </div>
          <ScoreBar score={sentiment.score} />

          {sentiment.keywords.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {sentiment.keywords.slice(0, 5).map((k) => (
                <span
                  key={k.keyword}
                  className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] text-gray-600 dark:bg-gray-800 dark:text-gray-300"
                >
                  {k.keyword}
                </span>
              ))}
            </div>
          )}

          {(sentiment.is_mock || sentiment.stale) && (
            <p className="mt-2 text-[10px] text-amber-600 dark:text-amber-400">
              {sentiment.is_mock ? "⚠ 情緒模型未載入（示意資料）" : "⚠ 新聞源暫時失效，顯示快取"}
            </p>
          )}
        </>
      )}

      <p className="mt-3 border-t border-gray-100 pt-2 text-[10px] leading-relaxed text-gray-400 dark:border-gray-800">
        由本團隊 news-sentiment-monitor（BERT 財經新聞情緒分析）提供；情緒模型只涵蓋英文新聞，
        台股僅供跨市場參考、不作為預測模型輸入。
      </p>
    </div>
  );
}
