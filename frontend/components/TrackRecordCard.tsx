import type { PastPrediction, TrackRecordResponse } from "@/lib/api";

const SIGNAL_TONE: Record<string, string> = {
  漲: "text-red-600 dark:text-red-400",
  跌: "text-green-600 dark:text-green-400",
  觀望: "text-gray-500 dark:text-gray-400",
};

/** 該股近期線上預測 vs 實際 + 全站實證摘要——命中與失誤都誠實列出 */
export default function TrackRecordCard({
  ticker,
  records,
  trackRecord,
}: {
  ticker: string;
  records: PastPrediction[];
  trackRecord: TrackRecordResponse | null;
}) {
  const shown = records.slice(0, 8);
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">
          線上預測實證（{ticker}）
        </h2>
        {trackRecord && trackRecord.matured > 0 && (
          <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
            全站已到期 {trackRecord.matured} 筆・命中率{" "}
            {Math.round((trackRecord.hit_rate ?? 0) * 100)}%
          </span>
        )}
      </div>

      {shown.length === 0 ? (
        <p className="mt-4 text-sm text-gray-400">
          尚無此股的線上預測紀錄——預測會在每日批次或查詢時自動落地累積。
        </p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[420px] text-xs">
            <thead>
              <tr className="border-b border-gray-100 text-left text-gray-400 dark:border-gray-800">
                <th className="py-1.5 font-medium">基準日</th>
                <th className="py-1.5 font-medium">預測</th>
                <th className="py-1.5 font-medium">信心</th>
                <th className="py-1.5 font-medium">實際</th>
                <th className="py-1.5 font-medium">5 日報酬</th>
                <th className="py-1.5 text-center font-medium">命中</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50 dark:divide-gray-800/60">
              {shown.map((r) => (
                <tr key={`${r.base_date}-${r.model_version}`}>
                  <td className="py-1.5 tabular-nums text-gray-500 dark:text-gray-400">
                    {r.base_date}
                  </td>
                  <td className={`py-1.5 font-medium ${SIGNAL_TONE[r.signal]}`}>{r.signal}</td>
                  <td className="py-1.5 tabular-nums text-gray-500">
                    {Math.round(r.confidence * 100)}%
                  </td>
                  <td className={`py-1.5 font-medium ${r.actual ? SIGNAL_TONE[r.actual] : "text-gray-300 dark:text-gray-600"}`}>
                    {r.actual ?? "未到期"}
                  </td>
                  <td className="py-1.5 tabular-nums text-gray-500">
                    {r.actual_return != null
                      ? `${r.actual_return >= 0 ? "+" : ""}${(r.actual_return * 100).toFixed(2)}%`
                      : "—"}
                  </td>
                  <td className="py-1.5 text-center">
                    {r.hit == null ? (
                      <span className="text-gray-300 dark:text-gray-600">…</span>
                    ) : r.hit ? (
                      <span className="text-emerald-600 dark:text-emerald-400">✓</span>
                    ) : (
                      <span className="text-rose-500">✗</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-gray-400">
        每筆預測於推論當下寫入資料庫、不可回改；到期後以實際 5 日趨勢對照。誠實呈現，包含失誤。
      </p>
    </div>
  );
}
