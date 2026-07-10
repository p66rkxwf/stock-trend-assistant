import numpy as np
import pandas as pd

from stockta.ml.labeling import LABEL_DOWN, LABEL_HOLD, LABEL_UP, make_labels


def test_labels_match_hand_computed_cases():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100, 100, 100, 100, 100, 103, 97, 101, 100, 100], index=idx, dtype=float)
    labels = make_labels(close, horizon=5)

    assert labels.iloc[0] == LABEL_UP  # 103/100 - 1 = +3% > +2%
    assert labels.iloc[1] == LABEL_DOWN  # 97/100 - 1 = -3% < -2%
    assert labels.iloc[2] == LABEL_HOLD  # +1% 在區間內
    assert labels.iloc[3] == LABEL_HOLD
    assert labels.iloc[4] == LABEL_HOLD


def test_last_horizon_rows_have_no_label():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series(np.linspace(100, 110, 10), index=idx)
    labels = make_labels(close, horizon=5)
    assert labels.iloc[-5:].isna().all()
    assert labels.iloc[:-5].notna().all()


def test_just_inside_threshold_is_hold():
    # 恰好 ±2% 的情況受浮點誤差影響無法穩定測試，改測門檻內側緊鄰值
    idx = pd.bdate_range("2024-01-01", periods=7)
    close = pd.Series([100, 100, 100, 100, 100, 101.9, 98.1], index=idx, dtype=float)
    labels = make_labels(close, horizon=5)
    assert labels.iloc[0] == LABEL_HOLD  # +1.9% 未破上門檻
    assert labels.iloc[1] == LABEL_HOLD  # −1.9% 未破下門檻
