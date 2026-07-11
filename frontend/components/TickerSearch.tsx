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
      <input
        type="text"
        className="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none dark:border-gray-600 dark:bg-gray-900"
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
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-900">
          {filtered.map((s) => (
            <li key={s.ticker}>
              <button
                type="button"
                className={`flex w-full items-center justify-between px-4 py-2 text-left text-sm hover:bg-blue-50 dark:hover:bg-gray-800 ${
                  s.ticker === selected ? "bg-blue-50 font-medium dark:bg-gray-800" : ""
                }`}
                onMouseDown={() => {
                  onSelect(s.ticker);
                  setQuery("");
                  setOpen(false);
                }}
              >
                <span>{s.name}</span>
                <span className="text-gray-400">{s.ticker}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
