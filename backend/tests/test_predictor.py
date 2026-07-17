"""resolve_signal 決策規則與版本字串的單元測試。"""

import numpy as np
import pytest

from stockta.config import LABEL_CLASSES, SIGNAL_CONFIDENCE_THRESHOLDS
from stockta.inference.predictor import Predictor, resolve_signal

DOWN, HOLD, UP = 0, 1, 2


def test_high_confidence_directional_signal_kept():
    proba = np.array([0.10, 0.20, 0.70])
    assert resolve_signal(proba) == UP


def test_low_confidence_up_downgraded_to_hold():
    tau = SIGNAL_CONFIDENCE_THRESHOLDS["漲"]
    proba = np.array([0.30, 0.30, 0.40])
    assert proba[UP] < tau, "測資前提：漲的信心需低於門檻"
    assert resolve_signal(proba) == HOLD


def test_low_confidence_down_downgraded_to_hold():
    tau = SIGNAL_CONFIDENCE_THRESHOLDS["跌"]
    proba = np.array([0.40, 0.31, 0.29])
    assert proba[DOWN] < tau, "測資前提：跌的信心需低於門檻"
    assert resolve_signal(proba) == HOLD


def test_hold_argmax_unaffected_by_thresholds():
    proba = np.array([0.30, 0.40, 0.30])
    assert resolve_signal(proba) == HOLD


def test_exact_threshold_confidence_kept():
    proba = np.zeros(3)
    proba[DOWN] = SIGNAL_CONFIDENCE_THRESHOLDS["跌"]
    proba[HOLD] = (1 - proba[DOWN]) / 2
    proba[UP] = 1 - proba[DOWN] - proba[HOLD]
    assert resolve_signal(proba) == DOWN


def test_version_marks_calibrated_rule():
    predictor = Predictor(
        model=None, scaler=None, metadata={"model_name": "gru", "trained_at": "2026-07-11T00:00:00"}
    )
    assert predictor.version == "gru-2026-07-11+cal"


def test_label_order_assumed_by_rule():
    # resolve_signal 依 LABEL_CLASSES 索引「觀望」；順序若變動此測試先失敗
    assert LABEL_CLASSES == ["跌", "觀望", "漲"]
