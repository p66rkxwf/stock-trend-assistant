"""實驗 #9 新增機制的守門測試：regime 欄位、折切分、門檻決策規則。

三個必須釘住的性質：
1. regime 欄位仍然是因果的（新特徵最容易偷偷引入前視偏差）
2. 特徵欄位順序由宣告決定，不因 context 帶了哪些欄位而漂移
3. 門檻決策規則只有一份實作（calibrate 與線上 predictor 必須同義）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockta.features.market import (
    MARKET_CONTEXT_BASE_COLUMNS,
    MARKET_REGIME_COLUMNS,
    build_market_context,
)
from stockta.features.pipeline import (
    RELATIVE_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
    build_features,
    feature_columns_for,
)
from stockta.ml.calibrate import HOLD, apply_thresholds, best_thresholds
from stockta.ml.dataset import build_dataset


@pytest.fixture
def regime_context(market_ohlcv, random_walk_ohlcv) -> pd.DataFrame:
    pool = {"2330.TW": random_walk_ohlcv}
    return build_market_context(market_ohlcv, pool, min_tickers=1, include_regime=True)


# --- regime 欄位的因果性 ---


def test_regime_columns_are_present_and_finite(regime_context):
    assert list(regime_context.columns) == [*MARKET_CONTEXT_BASE_COLUMNS, *MARKET_REGIME_COLUMNS]
    assert len(regime_context) > 0
    assert np.isfinite(regime_context[MARKET_REGIME_COLUMNS].to_numpy()).all()


def test_regime_columns_have_expected_ranges(regime_context):
    # 回撤是「距一年高點」，永遠 ≤0；波動百分位是比率，落在 (0, 1]
    assert (regime_context["mkt_drawdown"] <= 1e-12).all()
    pct = regime_context["mkt_vol_pct"]
    assert (pct > 0).all() and (pct <= 1.0).all()


def test_future_market_data_does_not_change_past_regime(market_ohlcv, random_walk_ohlcv):
    """竄改未來的大盤資料，過去的 regime 欄位必須逐值不變。"""
    pool = {"2330.TW": random_walk_ohlcv}
    cut = 500

    full = build_market_context(market_ohlcv, pool, min_tickers=1, include_regime=True)
    tampered_market = market_ohlcv.copy()
    tampered_market.iloc[cut:, :] = tampered_market.iloc[cut:, :] * 5.0
    tampered = build_market_context(tampered_market, pool, min_tickers=1, include_regime=True)

    cutoff = market_ohlcv.index[cut - 1]
    a = full.loc[:cutoff, MARKET_REGIME_COLUMNS]
    b = tampered.loc[:cutoff, MARKET_REGIME_COLUMNS]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)


# --- 特徵欄位順序 ---


def test_feature_columns_follow_context_contents(market_context, regime_context):
    base_cols = feature_columns_for(market_context)
    regime_cols = feature_columns_for(regime_context)

    assert base_cols == [
        *STOCK_FEATURE_COLUMNS,
        *MARKET_CONTEXT_BASE_COLUMNS,
        *RELATIVE_FEATURE_COLUMNS,
    ]
    assert regime_cols == [
        *STOCK_FEATURE_COLUMNS,
        *MARKET_CONTEXT_BASE_COLUMNS,
        *MARKET_REGIME_COLUMNS,
        *RELATIVE_FEATURE_COLUMNS,
    ]
    assert len(regime_cols) == len(base_cols) + len(MARKET_REGIME_COLUMNS)


def test_build_features_works_with_both_context_shapes(
    random_walk_ohlcv, market_context, regime_context
):
    base = build_features(random_walk_ohlcv, market_context)
    regime = build_features(random_walk_ohlcv, regime_context)
    assert list(base.columns) == feature_columns_for(market_context)
    assert list(regime.columns) == feature_columns_for(regime_context)
    assert len(base) > 0 and len(regime) > 0


# --- 折切分 ---


def test_fold_boundaries_and_embargo(random_walk_ohlcv, market_context):
    pool = {"2330.TW": random_walk_ohlcv}
    idx = market_context.index
    train_end, val_end, test_end = idx[200], idx[300], idx[380]

    ds = build_dataset(
        pool,
        market_context,
        window=20,
        stride=1,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )

    assert len(ds.dates_test) == len(ds.y_test)
    if len(ds.y_test):
        dates = pd.to_datetime(ds.dates_test)
        assert dates.min() > val_end
        # embargo：標籤視野（+5 交易日）不得越過測試期末端，故基準日必嚴格早於 test_end
        assert dates.max() < test_end


def test_test_end_shrinks_the_test_window(random_walk_ohlcv, market_context):
    pool = {"2330.TW": random_walk_ohlcv}
    idx = market_context.index
    train_end, val_end = idx[200], idx[300]

    unbounded = build_dataset(
        pool, market_context, window=20, stride=1, train_end=train_end, val_end=val_end
    )
    bounded = build_dataset(
        pool,
        market_context,
        window=20,
        stride=1,
        train_end=train_end,
        val_end=val_end,
        test_end=idx[350],
    )
    assert len(bounded.y_test) < len(unbounded.y_test)
    # 訓練/驗證兩組完全不受 test_end 影響
    assert len(bounded.y_train) == len(unbounded.y_train)
    assert len(bounded.y_val) == len(unbounded.y_val)


# --- 門檻決策規則 ---


def test_apply_thresholds_downgrades_low_confidence_directions():
    proba = np.array(
        [
            [0.60, 0.20, 0.20],  # 跌，信心夠 → 維持
            [0.40, 0.35, 0.25],  # 跌，信心不足 → 降級觀望
            [0.20, 0.20, 0.60],  # 漲，信心夠 → 維持
            [0.25, 0.35, 0.40],  # 漲，信心不足 → 降級觀望
            [0.30, 0.45, 0.25],  # 本來就是觀望 → 不受門檻影響
        ]
    )
    out = apply_thresholds(proba, down=0.45, up=0.45)
    assert out.tolist() == [0, HOLD, 2, HOLD, HOLD]


def test_apply_thresholds_never_creates_new_directions():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=200)
    strict = apply_thresholds(proba, down=0.99, up=0.99)
    assert set(np.unique(strict)) <= {HOLD}  # 門檻拉到極限只剩觀望


def test_best_thresholds_prefers_downgrading_a_wrong_direction():
    # 兩筆「跌」預測：一筆高信心正確、一筆低信心錯誤（實際為觀望）
    proba = np.array([[0.70, 0.20, 0.10], [0.40, 0.35, 0.25]])
    y = np.array([0, 1])

    acc, down, up = best_thresholds(proba, y)
    assert acc == 1.0
    assert 0.40 < down <= 0.70  # 門檻要落在能擋掉錯誤那筆、又保留正確那筆的區間
    assert (apply_thresholds(proba, down, up) == y).all()
