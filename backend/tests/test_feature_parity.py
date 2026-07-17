"""訓練/推論特徵一致（parity）：推論端用較短的近期序列計算特徵，
在 config.INDICATOR_WARMUP_DAYS 暖機後必須與訓練端（全序列）逐值一致。

EMA 類指標受序列起點影響，120 日暖機讓殘差收斂到 1e-4 量級——
若把 INDICATOR_WARMUP_DAYS 調小導致此測試失敗，代表推論結果會偏離訓練分佈。
市場情境 context 只用 rolling（無 EMA），較短的前置歷史即可逐值一致。
"""

import numpy as np

from stockta.config import INDICATOR_WARMUP_DAYS, WINDOW_LENGTH_DAYS
from stockta.features.market import MARKET_CONTEXT_COLUMNS, build_market_context
from stockta.features.pipeline import FEATURE_COLUMNS, build_features


def test_feature_columns_order_frozen():
    assert FEATURE_COLUMNS[0] == "ret_1d"
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    # 市場情境欄位必須整組出現在特徵清單（順序由 MARKET_CONTEXT_COLUMNS 唯一定義）
    for col in MARKET_CONTEXT_COLUMNS:
        assert col in FEATURE_COLUMNS


def test_short_history_features_match_full_history(random_walk_ohlcv, market_context):
    df = random_walk_ohlcv
    lookback = INDICATOR_WARMUP_DAYS + WINDOW_LENGTH_DAYS + 10

    full = build_features(df, market_context)
    short = build_features(df.iloc[-lookback:], market_context)

    # 推論實際使用的是最後 WINDOW_LENGTH_DAYS 列
    window = short.iloc[-WINDOW_LENGTH_DAYS:]
    expected = full.loc[window.index]
    np.testing.assert_allclose(
        expected.to_numpy(), window.to_numpy(), rtol=1e-3, atol=1e-4
    )


def test_short_history_context_matches_full_history(market_ohlcv, random_walk_ohlcv):
    """context 端 parity：推論端只給近期市場資料，rolling 暖機後須與全歷史一致。"""
    pool = {"2330.TW": random_walk_ohlcv}
    full = build_market_context(market_ohlcv, pool, min_tickers=1)

    lookback = 90  # rolling20 + breadth_ma5 暖機（26 列）之後仍留有比對餘裕
    short = build_market_context(
        market_ohlcv.iloc[-lookback:],
        {"2330.TW": random_walk_ohlcv.iloc[-lookback:]},
        min_tickers=1,
    )

    window = short.iloc[-30:]
    expected = full.loc[window.index]
    np.testing.assert_allclose(expected.to_numpy(), window.to_numpy(), rtol=1e-9, atol=1e-12)
