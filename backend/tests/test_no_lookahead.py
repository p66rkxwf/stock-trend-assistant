"""無前視偏差：t 日的特徵只能依賴 t 日（含）以前的資料。

作法：把 t 日之後的資料全部改掉，t 日（含）以前的特徵必須逐值不變。
"""

import numpy as np

from stockta.features.pipeline import build_features


def test_future_data_does_not_change_past_features(random_walk_ohlcv):
    df = random_walk_ohlcv
    cut = 400

    full = build_features(df)
    tampered = df.copy()
    tampered.iloc[cut:, :] = tampered.iloc[cut:, :] * 3.7  # 竄改「未來」
    tampered_feats = build_features(tampered)

    cutoff_date = df.index[cut - 1]
    a = full.loc[:cutoff_date]
    b = tampered_feats.loc[:cutoff_date]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)


def test_truncating_future_keeps_past_features_identical(random_walk_ohlcv):
    df = random_walk_ohlcv
    full = build_features(df)
    truncated = build_features(df.iloc[:450])

    common = truncated.index
    np.testing.assert_allclose(
        full.loc[common].to_numpy(), truncated.to_numpy(), rtol=1e-12, atol=1e-12
    )
