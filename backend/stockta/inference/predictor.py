"""線上推論：載入 artifact，對單一股票的近期 OHLCV 產出趨勢預測。

Predictor 不碰網路——OHLCV 由呼叫端經 DataProvider 取得後傳入，
本模組只負責「特徵 → 視窗 → 機率」，與訓練共用 build_features 同一入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockta.config import (
    ARTIFACTS_DIR,
    LABEL_CLASSES,
    SIGNAL_CONFIDENCE_THRESHOLDS,
    WINDOW_LENGTH_DAYS,
)
from stockta.ml import registry


class InsufficientDataError(RuntimeError):
    """暖機後資料不足一個滑動視窗，無法推論。"""


def resolve_signal(proba: np.ndarray) -> int:
    """機率 → 類別索引的唯一決策規則（Predictor 與回測共用）。

    argmax 為方向訊號（漲/跌）但信心低於 SIGNAL_CONFIDENCE_THRESHOLDS 時
    降級為「觀望」——門檻由驗證期校準（見 config 該設定的註解）。
    """
    idx = int(np.argmax(proba))
    if proba[idx] < SIGNAL_CONFIDENCE_THRESHOLDS.get(LABEL_CLASSES[idx], 0.0):
        return LABEL_CLASSES.index("觀望")
    return idx


@dataclass
class Prediction:
    signal: str
    confidence: float
    base_date: date
    proba: list[float]


class Predictor:
    def __init__(self, model, scaler, metadata: dict[str, Any]):
        self._model = model
        self._scaler = scaler
        self.metadata = metadata

    @classmethod
    def from_registry(cls, name: str, artifacts_dir: Path = ARTIFACTS_DIR) -> "Predictor":
        model, scaler, metadata = registry.load(name, artifacts_dir)
        return cls(model, scaler, metadata)

    @property
    def version(self) -> str:
        # 有信心門檻時加註 +cal：同一模型在校準前後的訊號分佈不同，
        # predictions.db 以版本字串區分，避免新舊決策規則的紀錄混在一起
        suffix = "+cal" if SIGNAL_CONFIDENCE_THRESHOLDS else ""
        return f"{self.metadata['model_name']}-{self.metadata['trained_at'][:10]}{suffix}"

    def predict(self, ohlcv: pd.DataFrame) -> Prediction:
        # 延遲 import：讓 API 在無 sklearn 環境仍可以 mock 模式啟動
        from stockta.features.pipeline import build_features

        feats = build_features(ohlcv)
        if len(feats) < WINDOW_LENGTH_DAYS:
            raise InsufficientDataError(
                f"暖機後僅 {len(feats)} 列特徵，不足 {WINDOW_LENGTH_DAYS} 日視窗；"
                f"請確認抓取區間涵蓋 warmup_lookback_days()"
            )
        window = feats.iloc[-WINDOW_LENGTH_DAYS:]
        scaled = self._scaler.transform(window.to_numpy(dtype=np.float64)).astype(np.float32)
        proba = self._model.predict_proba(scaled.reshape(1, -1))[0]
        idx = resolve_signal(proba)
        return Prediction(
            signal=LABEL_CLASSES[idx],
            confidence=float(proba[idx]),
            base_date=window.index[-1].date(),
            proba=[float(p) for p in proba],
        )
