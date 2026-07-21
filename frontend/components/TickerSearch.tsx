"use client";

/**
 * 股票選擇器：清單一律來自 GET /api/stocks（唯一事實來源），前端不得內建清單。
 * 支援輸入代號或名稱過濾。
 */

import { useMemo, useState } from "react";
import type { StockInfo } from "@/lib/api";

export default function TickerSearch({
  stocks,
  selected,
  onSelect,
}: {
  stocks: StockInfo[];
  selected: string;
  onSelect: (ticker: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return stocks;
    return stocks.filter(
      (s) => s.ticker.toLowerCase().includes(q) || s.name.toLowerCase().includes(q),
    );
  }, [stocks, query]);

  const current = stocks.find((s) => s.ticker === selected);

  return (
    <div className="relative w-full max-w-sm">
      <svg
        className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-3"
        width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
      >
        <circle cx="11" cy="11" r="7" />
        <path d="M21 21l-4.3-4.3" />
      </svg>
      <input
        type="text"
        className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm shadow-(--shadow-sm) outline-none transition focus:border-accent"
        placeholder={current ? `${current.ticker} ${current.name}` : "輸入代號或名稱，如 2330 或 台積電"}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {open && filtered.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-border bg-surface py-1 shadow-(--shadow-md)">
          {filtered.map((s) => (
            <li key={s.ticker}>
              <button
                type="button"
                className="flex w-full items-center justify-between px-4 py-2 text-left text-sm text-ink transition hover:bg-surface-2"
                style={s.ticker === selected ? { background: "var(--accent-soft)", fontWeight: 600 } : undefined}
                onMouseDown={() => {
                  onSelect(s.ticker);
                  setQuery("");
                  setOpen(false);
                }}
              >
                <span>{s.name}</span>
                <span className="text-ink-3">{s.ticker}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
