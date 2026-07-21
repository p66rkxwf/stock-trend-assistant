import type { RiskLevel } from "@/lib/api";

// 風險為狀態型配色（低綠/中黃/高紅），與台股漲跌無關，另附文字標籤
const STYLE: Record<RiskLevel, { fg: string; bg: string }> = {
  低: { fg: "var(--down)", bg: "var(--down-soft)" },
  中: { fg: "#b45309", bg: "rgba(180,131,9,0.14)" },
  高: { fg: "var(--up)", bg: "var(--up-soft)" },
};

export default function RiskBadge({ risk }: { risk: RiskLevel }) {
  const s = STYLE[risk];
  return (
    <span
      className="inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold"
      style={{ color: s.fg, background: s.bg }}
      title="依近 60 日年化波動率換算"
    >
      {risk}風險
    </span>
  );
}
