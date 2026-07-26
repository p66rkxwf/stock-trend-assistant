import type { PredictionResponse } from "@/lib/api";
import RiskBadge from "./RiskBadge";

// 台股慣例：漲紅、跌綠、觀望灰（狀態型配色，另附文字標籤）
const SIGNAL_VAR: Record<PredictionResponse["signal"], string> = {
  漲: "--up",
  跌: "--down",
  觀望: "--hold",
};

const SIGNAL_ORDER: PredictionResponse["signal"][] = ["漲", "觀望", "跌"];

function ProbaBars({ proba }: { proba: NonNullable<PredictionResponse["proba"]> }) {
  return (
    <div className="mt-4 space-y-1.5">
      {SIGNAL_ORDER.map((s) => {
        const pct = Math.round((proba[s] ?? 0) * 100);
        return (
          <div key={s} className="flex items-center gap-2 text-xs">
            <span className="w-8 shrink-0 font-semibold" style={{ color: `var(${SIGNAL_VAR[s]})` }}>
              {s}
            </span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${pct}%`, background: `var(${SIGNAL_VAR[s]})` }}
              />
            </div>
            <span className="w-9 shrink-0 text-right tabular-nums text-ink-3">{pct}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default function PredictionCard({
  prediction,
  testAuc,
}: {
  prediction: PredictionResponse;
  testAuc: number | null;
}) {
  const pct = Math.round(prediction.confidence * 100);
  const downgraded =
    prediction.proba &&
    prediction.signal === "觀望" &&
    prediction.proba["觀望"] < Math.max(prediction.proba["漲"], prediction.proba["跌"]);

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-(--shadow-sm)">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink-2">未來 5 個交易日趨勢預測</h2>
        <RiskBadge risk={prediction.risk} />
      </div>

      <div className="mt-3 flex items-end gap-4">
        <span className="text-5xl font-bold" style={{ color: `var(${SIGNAL_VAR[prediction.signal]})` }}>
          {prediction.signal}
        </span>
        <div className="mb-1">
          <div className="text-sm text-ink-3">信心機率</div>
          <div className="text-2xl font-semibold tabular-nums text-ink">{pct}%</div>
        </div>
      </div>

      {prediction.proba && <ProbaBars proba={prediction.proba} />}

      {downgraded && (
        <p className="mt-3 rounded-lg bg-surface-2 px-3 py-2 text-xs text-ink-2">
          模型的方向判斷信心未達門檻，依決策規則轉為「觀望」——寧可少喊，也不亂喊。
        </p>
      )}

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-3">
        <dt>基準日</dt>
        <dd className="text-right text-ink-2">{prediction.base_date}</dd>
        <dt>模型版本</dt>
        <dd className="text-right text-ink-2">{prediction.model_version}</dd>
        {testAuc != null && (
          <>
            <dt>測試集 Macro AUC</dt>
            <dd className="text-right text-ink-2">{testAuc.toFixed(4)}</dd>
          </>
        )}
      </dl>

      {prediction.is_mock && (
        <p className="mt-3 rounded-lg px-3 py-2 text-xs" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>
          ⚠ 目前為示意資料（模型尚未載入），預測結果不具參考性。
        </p>
      )}

      <p className="mt-4 border-t border-border pt-3 text-[11px] leading-relaxed text-ink-3">
        免責聲明：本預測為機器學習模型之統計輸出，僅供學術研究與參考，不構成任何投資建議。
        投資有風險，決策前請自行審慎評估。
      </p>
    </div>
  );
}
