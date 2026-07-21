import type { MarketResponse } from "@/lib/api";

const signed = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
// 台股慣例：漲紅跌綠
const toneOf = (v: number) => (v > 0 ? "var(--up)" : v < 0 ? "var(--down)" : "var(--ink)");

/** 上漲家數比 gauge：0–1，>0.5 偏多（紅）、<0.5 偏空（綠） */
function BreadthGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs text-ink-3">
        <span>市場寬度（上漲家數比）</span>
        <span className="text-sm font-semibold tabular-nums text-ink">{pct}%</span>
      </div>
      <div
        className="relative mt-1.5 h-2.5 w-full overflow-hidden rounded-full"
        style={{ background: "linear-gradient(90deg, var(--down-soft), var(--surface-2), var(--up-soft))" }}
      >
        <div
          className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded bg-ink"
          style={{ left: `calc(${pct}% - 2px)` }}
        />
      </div>
    </div>
  );
}

export default function MarketCard({ market }: { market: MarketResponse }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink-2">大盤情境（加權指數）</h2>
        <span className="text-[10px] text-ink-3">{market.as_of}</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-ink-3">
        <div>
          <div>日漲跌</div>
          <div className="text-lg font-semibold tabular-nums" style={{ color: toneOf(market.ret_1d) }}>
            {signed(market.ret_1d)}
          </div>
        </div>
        <div>
          <div>5 日累積</div>
          <div className="text-lg font-semibold tabular-nums" style={{ color: toneOf(market.ret_5d) }}>
            {signed(market.ret_5d)}
          </div>
        </div>
        <div>
          <div>20 日均線乖離</div>
          <div className="text-sm font-semibold tabular-nums" style={{ color: toneOf(market.ma20_bias) }}>
            {signed(market.ma20_bias)}
          </div>
        </div>
        <div>
          <div>20 日波動（日）</div>
          <div className="text-sm font-semibold tabular-nums text-ink">{(market.vol20 * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="mt-4">
        <BreadthGauge value={market.breadth_up} />
        <p className="mt-1 text-[10px] text-ink-3">
          5 日平均 {Math.round(market.breadth_ma5 * 100)}%；樣本為台灣 50 成分股池
        </p>
      </div>

      <p className="mt-3 border-t border-border pt-2 text-[10px] leading-relaxed text-ink-3">
        大盤情境與寬度同時作為模型的市場特徵輸入。
      </p>
    </div>
  );
}
