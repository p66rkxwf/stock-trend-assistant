"use client";

/**
 * 全池掃描：對台灣 50 全池即時推論（或查歷史某日的 point-in-time 重算 + 實際命中）。
 * 訊號分佈 headline + 可依信心/訊號排序、依訊號篩選的表格。
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import NavTabs from "@/components/NavTabs";
import RankView from "@/components/RankView";
import ThemeToggle from "@/components/ThemeToggle";
import { api, ApiError, type ScanResponse, type ScanResult, type Signal } from "@/lib/api";

type Mode = "signal" | "rank";

const SIGNAL_VAR: Record<Signal, string> = { 漲: "--up", 跌: "--down", 觀望: "--hold" };
type SortKey = "confidence" | "signal" | "name";
type SigFilter = "all" | Signal;

function DistributionBar({ up, hold, down }: { up: number; hold: number; down: number }) {
  const total = up + hold + down || 1;
  const segs = [
    { key: "漲" as Signal, n: up },
    { key: "觀望" as Signal, n: hold },
    { key: "跌" as Signal, n: down },
  ];
  return (
    <div>
      <div className="flex h-8 w-full gap-0.5 overflow-hidden rounded-lg">
        {segs.map((s) =>
          s.n === 0 ? null : (
            <div
              key={s.key}
              className="flex items-center justify-center text-xs font-semibold text-white"
              style={{ width: `${(s.n / total) * 100}%`, background: `var(${SIGNAL_VAR[s.key]})`, minWidth: "32px" }}
              title={`${s.key} ${s.n} 檔`}
            >
              {s.key} {s.n}
            </div>
          ),
        )}
      </div>
    </div>
  );
}

export default function ScanPage() {
  const [mode, setMode] = useState<Mode>("signal");
  const [data, setData] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dateInput, setDateInput] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const [filter, setFilter] = useState<SigFilter>("all");

  const run = useCallback(async (date?: string) => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.scan(date));
    } catch (e) {
      setData(null);
      setError(e instanceof ApiError ? e.message : "發生未知錯誤");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    run();
  }, [run]);

  const rows = useMemo(() => {
    if (!data) return [];
    const filtered = filter === "all" ? data.results : data.results.filter((r) => r.signal === filter);
    const sorted = [...filtered];
    if (sortKey === "confidence") sorted.sort((a, b) => b.confidence - a.confidence);
    else if (sortKey === "signal") {
      const order: Record<Signal, number> = { 漲: 0, 觀望: 1, 跌: 2 };
      sorted.sort((a, b) => order[a.signal] - order[b.signal] || b.confidence - a.confidence);
    } else sorted.sort((a, b) => a.name.localeCompare(b.name, "zh-Hant"));
    return sorted;
  }, [data, filter, sortKey]);

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight sm:text-2xl">全池掃描</h1>
          <p className="mt-0.5 text-sm text-ink-3">
            {mode === "signal" ? "台灣 50 全池模型訊號總覽" : "相對強弱排序（cross-sectional）"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex gap-1 rounded-lg bg-surface-2 p-1">
            {(["signal", "rank"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className="rounded-md px-3 py-1.5 text-sm font-medium transition"
                style={
                  mode === m
                    ? { background: "var(--surface)", color: "var(--ink)", boxShadow: "var(--shadow-sm)" }
                    : { color: "var(--ink-3)" }
                }
              >
                {m === "signal" ? "訊號" : "相對強弱排名"}
              </button>
            ))}
          </div>
          <NavTabs current="scan" />
          <ThemeToggle />
        </div>
      </header>

      {mode === "rank" && <RankView />}

      {mode === "signal" && (
      <>
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
            {data.is_historical ? "歷史回放" : "即時"} · 基準日 {data.base_date} · {data.model_version}
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
          全池推論中…（{data?.is_historical ? "歷史回放" : "即時"}約數秒）
        </div>
      ) : data && data.results.length > 0 ? (
        <>
          {/* 分佈 headline */}
          <section className="mb-5 rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-ink-2">
                模型訊號分佈（{data.results.length} 檔）
              </h2>
              {data.is_historical && data.matured > 0 && (
                <span
                  className="rounded-full px-2.5 py-0.5 text-xs font-medium"
                  style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                >
                  當日已到期 {data.matured} 檔・命中率 {Math.round((data.hits / data.matured) * 100)}%
                </span>
              )}
            </div>
            <DistributionBar up={data.up} hold={data.hold} down={data.down} />
          </section>

          {/* 控制列 */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="flex gap-1.5">
              {(["all", "漲", "觀望", "跌"] as SigFilter[]).map((f) => {
                const active = filter === f;
                const v = f === "all" ? "--accent" : SIGNAL_VAR[f as Signal];
                return (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setFilter(f)}
                    className="rounded-full px-2.5 py-1 text-xs font-medium transition"
                    style={
                      active
                        ? { background: `var(${v})`, color: "#fff" }
                        : { background: "var(--surface-2)", color: "var(--ink-2)" }
                    }
                  >
                    {f === "all" ? "全部" : f}
                  </button>
                );
              })}
            </div>
            <div className="ml-auto flex items-center gap-1.5 text-xs text-ink-3">
              <span>排序</span>
              {(["confidence", "signal", "name"] as SortKey[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setSortKey(k)}
                  className="rounded-md px-2 py-1 font-medium transition"
                  style={
                    sortKey === k
                      ? { background: "var(--accent-soft)", color: "var(--accent)" }
                      : { background: "var(--surface-2)", color: "var(--ink-2)" }
                  }
                >
                  {k === "confidence" ? "信心" : k === "signal" ? "訊號" : "名稱"}
                </button>
              ))}
            </div>
          </div>

          {/* 表格 */}
          <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-(--shadow-sm)">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-ink-3">
                  <th className="px-4 py-2.5 font-medium">標的</th>
                  <th className="px-4 py-2.5 font-medium">訊號</th>
                  <th className="px-4 py-2.5 font-medium">信心</th>
                  <th className="px-4 py-2.5 font-medium">三類機率</th>
                  {data.is_historical && <th className="px-4 py-2.5 font-medium">實際</th>}
                  {data.is_historical && <th className="px-4 py-2.5 text-center font-medium">命中</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {rows.map((r) => (
                  <Row key={r.ticker} r={r} historical={data.is_historical} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : data && data.is_mock ? (
        <div className="rounded-2xl border border-border p-8 text-center text-sm text-ink-3">
          模型尚未載入，無法掃描。
        </div>
      ) : null}
      </>
      )}

      <footer className="mt-8 text-center text-xs text-ink-3">
        全池掃描與模型走同一條 build_features 推論路徑；歷史查詢為 point-in-time 重算，特徵僅用當日與更早資料。
      </footer>
    </main>
  );
}

function Row({ r, historical }: { r: ScanResult; historical: boolean }) {
  const pct = Math.round(r.confidence * 100);
  return (
    <tr className="transition hover:bg-surface-2">
      <td className="px-4 py-2.5">
        <div className="font-medium text-ink">{r.name}</div>
        <div className="text-[11px] text-ink-3">{r.ticker}</div>
      </td>
      <td className="px-4 py-2.5">
        <span
          className="rounded-md px-2 py-0.5 text-xs font-semibold"
          style={{ color: `var(${SIGNAL_VAR[r.signal]})`, background: `var(${SIGNAL_VAR[r.signal]}-soft)` }}
        >
          {r.signal}
        </span>
      </td>
      <td className="px-4 py-2.5 tabular-nums text-ink-2">{pct}%</td>
      <td className="px-4 py-2.5">
        <div className="flex h-2 w-28 overflow-hidden rounded-full">
          {(["漲", "觀望", "跌"] as Signal[]).map((s) => (
            <div key={s} style={{ width: `${(r.proba[s] ?? 0) * 100}%`, background: `var(${SIGNAL_VAR[s]})` }} />
          ))}
        </div>
      </td>
      {historical && (
        <td className="px-4 py-2.5">
          {r.actual ? (
            <span className="font-medium" style={{ color: `var(${SIGNAL_VAR[r.actual]})` }}>
              {r.actual}
              {r.actual_return != null && (
                <span className="ml-1 text-[11px] text-ink-3">
                  {r.actual_return >= 0 ? "+" : ""}
                  {(r.actual_return * 100).toFixed(1)}%
                </span>
              )}
            </span>
          ) : (
            <span className="text-ink-3">未到期</span>
          )}
        </td>
      )}
      {historical && (
        <td className="px-4 py-2.5 text-center">
          {r.hit == null ? (
            <span className="text-ink-3">…</span>
          ) : r.hit ? (
            <span style={{ color: "var(--down)" }}>✓</span>
          ) : (
            <span style={{ color: "var(--up)" }}>✗</span>
          )}
        </td>
      )}
    </tr>
  );
}
