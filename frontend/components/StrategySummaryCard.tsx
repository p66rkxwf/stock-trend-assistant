import type { RankSummaryResponse } from "@/lib/api";

const pct = (v: number | null) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);

/** Cross-sectional 排序策略摘要：Rank IC（主證據）+ 扣成本組合超額（示意）。 */
export default function StrategySummaryCard({ s }: { s: RankSummaryResponse | null }) {
  if (!s || !s.available) {
    return (
      <div className="rounded-2xl border border-border bg-surface p-5 text-sm text-ink-3 shadow-(--shadow-sm)">
        策略回測摘要尚未產生（需先跑 cross_sectional report）。
      </div>
    );
  }
  const icGood = (s.test_rank_ic ?? 0) >= 0.02 && (s.test_rank_ic_t ?? 0) >= 2;
  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink-2">相對強弱排序 · 策略驗證</h2>
        <span className="text-[10px] text-ink-3">模型 {s.model}</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-ink-3">測試期 Rank IC</div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold tabular-nums" style={{ color: icGood ? "var(--up)" : "var(--ink)" }}>
              {s.test_rank_ic == null ? "—" : `+${s.test_rank_ic.toFixed(3)}`}
            </span>
            <span className="text-xs text-ink-3">t={s.test_rank_ic_t?.toFixed(1)}</span>
          </div>
          <div className="text-[10px] text-ink-3">0.02~0.05 可用｜t&gt;2 顯著</div>
        </div>
        <div>
          <div className="text-xs text-ink-3">扣成本超額（前 20% vs 全池）</div>
          <div className="text-2xl font-bold tabular-nums" style={{ color: "var(--up)" }}>
            {pct(s.net_excess_cum)}
          </div>
          <div className="text-[10px] text-ink-3">
            每 {s.holding_days} 日換股・勝率 {s.win_rate == null ? "—" : `${Math.round(s.win_rate * 100)}%`}
          </div>
        </div>
      </div>

      <p className="mt-3 border-t border-border pt-2 text-[10px] leading-relaxed text-ink-3">
        Rank IC 為主證據（不受多頭影響，樣本外 2025-2026）；組合超額已扣約 {s.cost_bps?.toFixed(0)}bp 來回成本，
        為示意、非投資建議——絕對報酬多為市場 beta。方法依 Gu-Kelly-Xiu(2020) 與 Grinold 主動管理基本定律。
      </p>
    </div>
  );
}
