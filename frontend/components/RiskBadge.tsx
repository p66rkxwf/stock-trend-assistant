import type { RiskLevel } from "@/lib/api";

const STYLES: Record<RiskLevel, string> = {
  低: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  中: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
  高: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
};

export default function RiskBadge({ risk }: { risk: RiskLevel }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${STYLES[risk]}`}
      title="依近 60 日年化波動率換算"
    >
      {risk}風險
    </span>
  );
}
