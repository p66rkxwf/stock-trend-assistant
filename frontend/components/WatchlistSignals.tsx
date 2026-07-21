"use client";

/**
 * 熱門標的預測總覽（新功能）：對一組權值股並行取預測，signal 徽章 + 信心，
 * 點任一列切換主檢視。與 news 專案的追蹤清單比較同一設計語言。
 */

import { useEffect, useState } from "react";
import { api, type Signal } from "@/lib/api";

const WATCH = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "3711.TW", "2603.TW"];

const SIGNAL_VAR: Record<Signal, string> = {
  漲: "--up",
  跌: "--down",
  觀望: "--hold",
};

type Row =
  | { ticker: string; signal: Signal; confidence: number; isMock: boolean }
  | { ticker: string; error: true };

export default function WatchlistSignals({
  stocks,
  selected,
  onSelect,
}: {
  stocks: { ticker: string; name: string }[];
  selected: string;
  onSelect: (t: string) => void;
}) {
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      WATCH.map((t) =>
        api
          .prediction(t)
          .then((p) => ({ ticker: t, signal: p.signal, confidence: p.confidence, isMock: p.is_mock }))
          .catch(() => ({ ticker: t, error: true as const })),
      ),
    ).then((r) => !cancelled && setRows(r));
    return () => {
      cancelled = true;
    };
  }, []);

  const nameOf = (t: string) => stocks.find((s) => s.ticker === t)?.name ?? "";

  return (
    <div>
      {rows === null ? (
        <p className="py-6 text-center text-sm text-ink-3">載入熱門標的…</p>
      ) : (
        <ul className="space-y-1">
          {rows.map((row) => {
            const active = row.ticker === selected;
            return (
              <li key={row.ticker}>
                <button
                  type="button"
                  onClick={() => onSelect(row.ticker)}
                  className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2"
                  style={active ? { background: "var(--accent-soft)" } : undefined}
                >
                  <span className={`w-16 shrink-0 text-sm font-semibold ${active ? "text-accent" : "text-ink"}`}>
                    {nameOf(row.ticker) || row.ticker}
                  </span>
                  {"error" in row ? (
                    <span className="flex-1 text-xs text-ink-3">無資料</span>
                  ) : (
                    <>
                      <span
                        className="rounded-md px-2 py-0.5 text-xs font-semibold"
                        style={{
                          color: `var(${SIGNAL_VAR[row.signal]})`,
                          background: `var(${SIGNAL_VAR[row.signal]}-soft)`,
                        }}
                      >
                        {row.signal}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.round(row.confidence * 100)}%`,
                            background: `var(${SIGNAL_VAR[row.signal]})`,
                          }}
                        />
                      </div>
                      <span className="w-9 shrink-0 text-right text-xs tabular-nums text-ink-3">
                        {Math.round(row.confidence * 100)}%
                      </span>
                    </>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-2 text-[10px] text-ink-3">六檔權值股即時預測；點選切換主檢視。</p>
    </div>
  );
}
