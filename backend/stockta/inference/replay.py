"""歷史回放（point-in-time 推論）：以 production 模型重現「模型在過去某日會做的預測」。

與訓練/線上推論共用 build_features 同一條特徵路徑（已由 test_no_lookahead 把關
特徵僅用當日與更早資料），故任一過去日的重算都是誠實的樣本外推論；到期後
可對照實際 5 日走勢。供 /api/scan?date= 與 /api/stocks/{t}/history 使用。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockta.config import WINDOW_LENGTH_DAYS
from stockta.features.pipeline import build_features
from stockta.inference.predictor import Predictor, resolve_signal


def scan_asof(
    predictor: Predictor, ohlcv: pd.DataFrame, context: pd.DataFrame, as_of: date
) -> tuple[np.ndarray, date] | None:
    """回傳 (三類機率, 基準日)：以截至 as_of（含）的資料推論最後一個視窗。
    暖機後不足一個視窗則回 None。基準日為 ≤ as_of 的最後一個特徵日。
    """
    feats = build_features(ohlcv, context)
    feats = feats.loc[feats.index <= pd.Timestamp(as_of)]
    if len(feats) < WINDOW_LENGTH_DAYS:
        return None
    window = feats.iloc[-WINDOW_LENGTH_DAYS:]
    scaled = predictor._scaler.transform(window.to_numpy(dtype=np.float64)).astype(np.float32)
    proba = predictor._model.predict_proba(scaled.reshape(1, -1))[0]
    return proba, window.index[-1].date()


def signal_of(proba: np.ndarray) -> str:
    """機率 → 正式決策規則（含信心門檻）的訊號字串。"""
    from stockta.config import LABEL_CLASSES

    return LABEL_CLASSES[resolve_signal(proba)]
