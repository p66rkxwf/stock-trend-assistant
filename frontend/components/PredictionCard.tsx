import type { PredictionResponse } from "@/lib/api";
import RiskBadge from "./RiskBadge";

// 台股慣例：漲紅、跌綠、觀望灰
const SIGNAL_STYLES: Record<PredictionResponse["signal"], string> = {
  漲: "text-red-600 dark:text-red-400",
  跌: "text-green-600 dark:text-green-400",
  觀望: "text-gray-500 dark:text-gray-400",
};

export default function PredictionCard({
  prediction,
  testAuc,
}: {
  prediction: PredictionResponse;
  testAuc: number | null;
}) {
  const pct = Math.round(prediction.confidence * 100);
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">
          未來 5 個交易日趨勢預測
        </h2>
        <RiskBadge risk={prediction.risk} />
      </div>

      <div className="mt-3 flex items-end gap-4">
        <span className={`text-5xl font-bold ${SIGNAL_STYLES[prediction.signal]}`}>
          {prediction.signal}
        </span>
        <div className="mb-1">
          <div className="text-sm text-gray-500 dark:text-gray-400">信心機率</div>
          <div className="text-2xl font-semibold">{pct}%</div>
        </div>
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
        <div
          className="h-full rounded-full bg-blue-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400">
        <dt>基準日</dt>
        <dd className="text-right">{prediction.base_date}</dd>
        <dt>模型版本</dt>
        <dd className="text-right">{prediction.model_version}</dd>
        {testAuc != null && (
          <>
            <dt>測試集 Macro AUC</dt>
            <dd className="text-right">{testAuc.toFixed(4)}</dd>
          </>
        )}
      </dl>

      {prediction.is_mock && (
        <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
          ⚠ 目前為示意資料（模型尚未載入），預測結果不具參考性。
        </p>
      )}

      <p className="mt-4 border-t border-gray-100 pt-3 text-[11px] leading-relaxed text-gray-400 dark:border-gray-800 dark:text-gray-500">
        免責聲明：本預測為機器學習模型之統計輸出，僅供學術研究與參考，不構成任何投資建議。
        投資有風險，決策前請自行審慎評估。
      </p>
    </div>
  );
}
