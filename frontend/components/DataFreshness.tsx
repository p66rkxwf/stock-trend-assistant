"use client";

/**
 * 靜態站頁尾：資料截至哪個交易日、何時更新。每日排程失敗時線上會停在上一版，
 * 超過 48 小時未更新就以警示色標出，避免使用者把過期資料當成最新。
 * 本機開發（即時後端）不顯示。
 */

import { useEffect, useState } from "react";
import { siteMeta, STATIC_DATA, type SiteMeta } from "@/lib/api";

const STALE_HOURS = 48;

function taipeiTime(iso: string): string {
  return new Date(iso).toLocaleString("zh-TW", {
    timeZone: "Asia/Taipei",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export default function DataFreshness() {
  const [meta, setMeta] = useState<SiteMeta | null>(null);

  useEffect(() => {
    if (STATIC_DATA) siteMeta().then(setMeta).catch(() => setMeta(null));
  }, []);

  if (!meta) return null;
  const stale = Date.now() - Date.parse(meta.generated_at) > STALE_HOURS * 3600 * 1000;

  return (
    <p
      className="mx-auto mb-6 w-fit rounded-full px-3 py-1 text-center text-xs"
      style={
        stale
          ? { background: "var(--hold-soft)", color: "var(--hold)" }
          : { color: "var(--ink-3)" }
      }
    >
      資料截至 {meta.data_as_of}（{taipeiTime(meta.generated_at)} 更新）
      {stale && "・資料可能已過期"}
    </p>
  );
}
