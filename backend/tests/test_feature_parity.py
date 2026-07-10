"""訓練/推論特徵一致（parity）：推論端用較短的近期序列計算特徵，
在 config.INDICATOR_WARMUP_DAYS 暖機後必須與訓練端（全序列）逐值一致。

EMA 類指標受序列起點影響，120 日暖機讓殘差收斂到 1e-4 量級——
若把 INDICATOR_WARMUP_DAYS 調小導致此測試失敗，代表推論結果會偏離訓練分佈。
"""

import numpy as np

from stockta.config import INDICATOR_WARMUP_DAYS, WINDOW_LENGTH_DAYS
from stockta.features.pipeline import FEATURE_COLUMNS, build_features


def test_feature_columns_order_frozen():
    assert FEATURE_COLUMNS[0] == "ret_1d"
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_short_history_features_match_full_history(random_walk_ohlcv):
    df = random_walk_ohlcv
    lookback = INDICATOR_WARMUP_DAYS + WINDOW_LENGTH_DAYS + 10

    full = build_features(df)
    short = build_features(df.iloc[-lookback:])

    # 推論實際使用的是最後 WINDOW_LENGTH_DAYS 列
    window = short.iloc[-WINDOW_LENGTH_DAYS:]
    expected = full.loc[window.index]
    np.testing.assert_allclose(
        expected.to_numpy(), window.to_numpy(), rtol=1e-3, atol=1e-4
    )
