"use client";

/**
 * 歷史預測回放（個股）：對選定日期區間做 point-in-time 重算 vs 實際，
 * 顯示命中率摘要、命中/失誤時間軸色帶，以及可捲動明細表。
 * 與線上實證卡（predictions.db 真實落地）互補：此為歷史回放，範圍可自訂。
 */

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type HistoryRecord, type Signal, type StockHistoryResponse } from "@/lib/api";

const SIGNAL_VAR: Record<Signal, string> = { 漲: "--up", 跌: "--down", 觀望: "--hold" };

function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

/** 命中/失誤時間軸：每筆一格，命中實心、失誤空心，色隨訊號。 */
function HitStrip({ records }: { records: HistoryRecord[] }) {
  const chron = [...records].reverse(); // 舊→新
  return (
    <div className="flex flex-wrap gap-0.5">
      {chron.map((r) => (
        <span
          key={r.date}
          className="h-3.5 w-2 rounded-[2px]"
          style={{
            background: r.hit ? `var(${SIGNAL_VAR[r.signal]})` : "transparent",
            border: `1px solid var(${SIGNAL_VAR[r.signal]})`,
            opacity: r.hit ? 1 : 0.5,
          }}
          title={`${r.date}｜預測 ${r.signal}／實際 ${r.actual}｜${r.hit ? "命中" : "未中"}`}
        />
      ))}
    </div>
  );
}

export default function HistoryReplayCard({ ticker }: { ticker: string }) {
  const [start, setStart] = useState(isoDaysAgo(180));
  const [end, setEnd] = useState(isoDaysAgo(0));
  const [data, setData] = useState<StockHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (t: string, s: string, e: string) => {
      setLoading(true);
      setError(null);
      try {
        setData(await api.stockHistory(t, s, e));
      } catch (err) {
        setData(null);
        setError(err instanceof ApiError ? err.message : "查詢失敗");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // 切換股票時以目前區間自動重查
  useEffect(() => {
    run(ticker, start, end);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-ink-2">歷史預測回放（{ticker}）</h2>
        {data && data.count > 0 && data.hit_rate != null && (
          <span
            className="rounded-full px-2.5 py-0.5 text-xs font-medium"
            style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
          >
            {data.count} 筆已到期・命中率 {Math.round(data.hit_rate * 100)}%
          </span>
        )}
      </div>

      <div className="mb-3 flex flex-wrap items-end gap-2">
        <label className="text-xs text-ink-3">
          起<input
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            className="ml-1 rounded-md border border-border bg-surface px-2 py-1 text-xs text-ink outline-none focus:border-accent"
          />
        </label>
        <label className="text-xs text-ink-3">
          迄<input
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            className="ml-1 rounded-md border border-border bg-surface px-2 py-1 text-xs text-ink outline-none focus:border-accent"
          />
        </label>
        <button
          type="button"
          onClick={() => run(ticker, start, end)}
          className="rounded-md bg-accent px-3 py-1 text-xs font-semibold text-accent-fg transition hover:opacity-90"
        >
          查詢
        </button>
      </div>

      {error && <p className="text-xs text-up">{error}</p>}

      {loading ? (
        <p className="py-6 text-center text-sm text-ink-3">歷史回放推論中…（約數秒）</p>
      ) : data && data.records.length > 0 ? (
        <>
          <div className="mb-3">
            <HitStrip records={data.records} />
            <p className="mt-1.5 text-[10px] text-ink-3">
              每格一次預測（實心＝命中、空心＝未中，色隨訊號）；左舊右新。
            </p>
          </div>
          <div className="max-h-64 overflow-y-auto rounded-lg border border-border">
            <table className="w-full min-w-[420px] text-xs">
              <thead className="sticky top-0 bg-surface-2">
                <tr className="text-left text-ink-3">
                  <th className="px-3 py-2 font-medium">日期</th>
                  <th className="px-3 py-2 font-medium">預測</th>
                  <th className="px-3 py-2 font-medium">信心</th>
                  <th className="px-3 py-2 font-medium">實際</th>
                  <th className="px-3 py-2 font-medium">5日報酬</th>
                  <th className="px-3 py-2 text-center font-medium">命中</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {data.records.map((r) => (
                  <tr key={r.date}>
                    <td className="px-3 py-1.5 tabular-nums text-ink-3">{r.date}</td>
                    <td className="px-3 py-1.5 font-semibold" style={{ color: `var(${SIGNAL_VAR[r.signal]})` }}>
                      {r.signal}
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-ink-3">{Math.round(r.confidence * 100)}%</td>
                    <td className="px-3 py-1.5 font-semibold" style={{ color: `var(${SIGNAL_VAR[r.actual]})` }}>
                      {r.actual}
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-ink-3">
                      {r.actual_return != null
                        ? `${r.actual_return >= 0 ? "+" : ""}${(r.actual_return * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      {r.hit ? (
                        <span style={{ color: "var(--down)" }}>✓</span>
                      ) : (
                        <span style={{ color: "var(--up)" }}>✗</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : data ? (
        <p className="py-6 text-center text-sm text-ink-3">此區間無已到期的預測樣本。</p>
      ) : null}

      <p className="mt-3 text-[10px] leading-relaxed text-ink-3">
        歷史回放為 point-in-time 重算（特徵僅用當日與更早資料），與線上實證（推論當下寫入不可回改）互補；僅顯示已到期樣本。
      </p>
    </div>
  );
}
