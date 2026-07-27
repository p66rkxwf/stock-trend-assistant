"use client";

/**
 * 全池相對強弱排名檢視：策略摘要卡 + 依 CS 分數排序的表格（可查歷史日期）。
 * 與訊號模式並存於 /scan；資料來自 /api/rank 與 /api/rank/summary。
 */

import { useCallback, useEffect, useState } from "react";
import StrategySummaryCard from "@/components/StrategySummaryCard";
import { api, ApiError, type RankResponse, type RankSummaryResponse } from "@/lib/api";

const Q_STYLE: Record<string, { text: string; varName: string }> = {
  top: { text: "強", varName: "--up" },
  mid: { text: "中", varName: "--hold" },
  bottom: { text: "弱", varName: "--down" },
};

export default function RankView() {
  const [data, setData] = useState<RankResponse | null>(null);
  const [summary, setSummary] = useState<RankSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dateInput, setDateInput] = useState("");

  const run = useCallback(async (date?: string) => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.rank(date));
    } catch (e) {
      setData(null);
      setError(e instanceof ApiError ? e.message : "發生未知錯誤");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    run();
    api.rankSummary().then(setSummary).catch(() => setSummary(null));
  }, [run]);

  const maxScore = data ? Math.max(...data.results.map((r) => r.score), 0.6) : 1;

  return (
    <>
      <div className="mb-5">
        <StrategySummaryCard s={summary} />
      </div>

      {/* 日期查詢列 */}
      <div className="mb-5 flex flex-wrap items-end gap-3 rounded-2xl border border-border bg-surface p-4 shadow-(--shadow-sm)">
        <div>
          <label className="mb-1 block text-xs font-medium text-ink-3">查詢日期（留空＝即時）</label>
          <input
            type="date"
            value={dateInput}
            onChange={(e) => setDateInput(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-ink outline-none focus:border-accent"
          />
        </div>
        <button
          type="button"
          onClick={() => run(dateInput || undefined)}
          className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg shadow-(--shadow-sm) transition hover:opacity-90"
        >
          查詢
        </button>
        {dateInput && (
          <button
            type="button"
            onClick={() => {
              setDateInput("");
              run();
            }}
            className="rounded-lg bg-surface-2 px-3 py-1.5 text-sm font-medium text-ink-2 transition hover:text-ink"
          >
            回到即時
          </button>
        )}
        {data && (
          <span className="ml-auto text-xs text-ink-3">
            {data.is_historical ? "歷史" : "即時"} · 基準日 {data.base_date} · {data.model}
          </span>
        )}
      </div>

      {error && (
        <div
          className="mb-5 rounded-xl border px-4 py-3 text-sm"
          style={{ borderColor: "var(--up)", background: "var(--up-soft)", color: "var(--up)" }}
        >
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex h-64 items-center justify-center rounded-2xl border border-border text-sm text-ink-3">
          全池評分排序中…
        </div>
      ) : data && data.results.length > 0 ? (
        <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-(--shadow-sm)">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-ink-3">
                <th className="px-4 py-2.5 font-medium">排名</th>
                <th className="px-4 py-2.5 font-medium">標的</th>
                <th className="px-4 py-2.5 font-medium">相對強弱分數</th>
                <th className="px-4 py-2.5 font-medium">百分位</th>
                <th className="px-4 py-2.5 text-center font-medium">分組</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {data.results.map((r) => {
                const q = Q_STYLE[r.quantile];
                return (
                  <tr key={r.ticker} className="transition hover:bg-surface-2">
                    <td className="px-4 py-2.5 tabular-nums font-semibold text-ink">{r.rank}</td>
                    <td className="px-4 py-2.5">
                      <div className="font-medium text-ink">{r.name}</div>
                      <div className="text-[11px] text-ink-3">{r.ticker}</div>
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-28 overflow-hidden rounded-full bg-surface-2">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${(r.score / maxScore) * 100}%`, background: `var(${q.varName})` }}
                          />
                        </div>
                        <span className="tabular-nums text-xs text-ink-2">{r.score.toFixed(3)}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-ink-3">{Math.round(r.percentile * 100)}%</td>
                    <td className="px-4 py-2.5 text-center">
                      <span
                        className="rounded-md px-2 py-0.5 text-xs font-semibold"
                        style={{ color: `var(${q.varName})`, background: `var(${q.varName}-soft)` }}
                      >
                        {q.text}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : data && data.is_mock ? (
        <div className="rounded-2xl border border-border p-8 text-center text-sm text-ink-3">
          相對強弱模型尚未載入。
        </div>
      ) : null}

      <p className="mt-4 text-[11px] leading-relaxed text-ink-3">
        分數＝模型預測「未來 5 日贏過全池中位數」的機率；排名越前＝相對越強。歷史查詢為 point-in-time 重算。
        僅供研究參考，非投資建議。
      </p>
    </>
  );
}
