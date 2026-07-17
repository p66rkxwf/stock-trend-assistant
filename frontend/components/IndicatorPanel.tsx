import type { IndicatorsResponse } from "@/lib/api";

/** 單一指標列：名稱 + 數值 + 簡易解讀（過熱/超賣等以顏色提示） */
function Row({
  name,
  value,
  hint,
  tone = "neutral",
}: {
  name: string;
  value: string;
  hint?: string;
  tone?: "hot" | "cold" | "neutral";
}) {
  const toneStyle =
    tone === "hot"
      ? "text-red-600 dark:text-red-400"
      : tone === "cold"
        ? "text-green-600 dark:text-green-400"
        : "text-gray-700 dark:text-gray-200";
  return (
    <div className="flex items-baseline justify-between gap-2 py-1.5">
      <span className="text-xs text-gray-500 dark:text-gray-400">{name}</span>
      <span className="text-right">
        <span className={`text-sm font-semibold tabular-nums ${toneStyle}`}>{value}</span>
        {hint && <span className="ml-1.5 text-[10px] text-gray-400">{hint}</span>}
      </span>
    </div>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function IndicatorPanel({ indicators }: { indicators: IndicatorsResponse }) {
  const rsi = indicators.rsi14;
  const k = indicators.kd_k;
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">技術指標</h2>
        <span className="text-[10px] text-gray-400">{indicators.as_of}</span>
      </div>
      <div className="mt-2 divide-y divide-gray-100 dark:divide-gray-800">
        <Row
          name="RSI (14)"
          value={(rsi * 100).toFixed(0)}
          hint={rsi > 0.7 ? "過熱" : rsi < 0.3 ? "超賣" : "中性"}
          tone={rsi > 0.7 ? "hot" : rsi < 0.3 ? "cold" : "neutral"}
        />
        <Row
          name="KD (K / D)"
          value={`${(k * 100).toFixed(0)} / ${(indicators.kd_d * 100).toFixed(0)}`}
          hint={k > 0.8 ? "高檔" : k < 0.2 ? "低檔" : undefined}
          tone={k > 0.8 ? "hot" : k < 0.2 ? "cold" : "neutral"}
        />
        <Row
          name="MACD 柱"
          value={indicators.macd_hist.toFixed(4)}
          hint={indicators.macd_hist > 0 ? "多方" : "空方"}
          tone={indicators.macd_hist > 0 ? "hot" : "cold"}
        />
        <Row
          name="布林 %B"
          value={indicators.bb_pctb.toFixed(2)}
          hint={indicators.bb_pctb > 1 ? "破上軌" : indicators.bb_pctb < 0 ? "破下軌" : undefined}
        />
        <Row name="量能比 (20日)" value={pct(indicators.vol_ratio)} hint="相對均量" />
        <Row name="乖離 5日" value={pct(indicators.ma_bias_5)} />
        <Row name="乖離 20日" value={pct(indicators.ma_bias_20)} />
        <Row name="乖離 60日" value={pct(indicators.ma_bias_60)} />
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-gray-400">
        與模型輸入同一條特徵管線計算，數值一致。
      </p>
    </div>
  );
}
