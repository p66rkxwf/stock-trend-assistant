"""無前視偏差：t 日的特徵只能依賴 t 日（含）以前的資料。

作法：把 t 日之後的資料全部改掉，t 日（含）以前的特徵必須逐值不變。
個股資料與市場情境（大盤/寬度）分別竄改驗證。
"""

import numpy as np
import pytest

from stockta.features.market import build_market_context
from stockta.features.pipeline import build_features

# 與 test_label_shuffle.py 同屬洩漏防治的把關，一起進 pre-push 閘門。
# 兩者互補：這裡抓前視特徵，那裡抓標籤資訊回流；打亂標籤測試對前視特徵是盲的。
pytestmark = pytest.mark.leakage


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


def test_gaps_are_not_filled_from_the_future(random_walk_ohlcv, market_context):
    """內部缺口必須被丟棄，不得用後值回填（缺失值填補的方向性）。

    現況本來就乾淨——全專案沒有任何 fillna/bfill/ffill，一律 dropna。這條測試是
    防止未來有人「順手補一下缺值」：缺口刻意緊鄰竄改邊界，一旦改用 bfill，
    缺口會拿邊界之後（已被竄改）的值來填，邊界之前的特徵就會跟著變，測試立刻紅。
    """
    df = random_walk_ohlcv.copy()
    boundary = 300
    df.iloc[boundary - 5 : boundary, :] = np.nan  # 緊貼邊界的內部缺口

    full = build_features(df, market_context)
    tampered = df.copy()
    tampered.iloc[boundary:, :] = tampered.iloc[boundary:, :] * 2.5
    tampered_feats = build_features(tampered, market_context)

    cutoff_date = df.index[boundary - 1]
    a = full.loc[:cutoff_date]
    b = tampered_feats.loc[:cutoff_date]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)

    # 缺口當天本身不該出現在輸出裡——這是「丟棄而非填補」的正面敘述
    gap_dates = df.index[boundary - 5 : boundary]
    assert full.index.intersection(gap_dates).empty


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
