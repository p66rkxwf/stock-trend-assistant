"""無前視偏差：t 日的特徵只能依賴 t 日（含）以前的資料。

作法：把 t 日之後的資料全部改掉，t 日（含）以前的特徵必須逐值不變。
個股資料與市場情境（大盤/寬度）分別竄改驗證。
"""

import numpy as np

from stockta.features.market import build_market_context
from stockta.features.pipeline import build_features


def test_future_data_does_not_change_past_features(random_walk_ohlcv, market_context):
    df = random_walk_ohlcv
    cut = 400

    full = build_features(df, market_context)
    tampered = df.copy()
    tampered.iloc[cut:, :] = tampered.iloc[cut:, :] * 3.7  # 竄改「未來」
    tampered_feats = build_features(tampered, market_context)

    cutoff_date = df.index[cut - 1]
    a = full.loc[:cutoff_date]
    b = tampered_feats.loc[:cutoff_date]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)


def test_truncating_future_keeps_past_features_identical(random_walk_ohlcv, market_context):
    df = random_walk_ohlcv
    full = build_features(df, market_context)
    truncated = build_features(df.iloc[:450], market_context)

    common = truncated.index
    np.testing.assert_allclose(
        full.loc[common].to_numpy(), truncated.to_numpy(), rtol=1e-12, atol=1e-12
    )


def test_future_market_data_does_not_change_past_context(market_ohlcv, random_walk_ohlcv):
    """竄改未來的大盤與池內股價，過去的市場情境特徵必須逐值不變。"""
    pool = {"2330.TW": random_walk_ohlcv}
    cut = 400

    full = build_market_context(market_ohlcv, pool, min_tickers=1)

    tampered_market = market_ohlcv.copy()
    tampered_market.iloc[cut:, :] = tampered_market.iloc[cut:, :] * 2.9
    tampered_stock = random_walk_ohlcv.copy()
    tampered_stock.iloc[cut:, :] = tampered_stock.iloc[cut:, :] * 0.4
    tampered = build_market_context(tampered_market, {"2330.TW": tampered_stock}, min_tickers=1)

    cutoff_date = market_ohlcv.index[cut - 1]
    a = full.loc[:cutoff_date]
    b = tampered.loc[:cutoff_date]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)
