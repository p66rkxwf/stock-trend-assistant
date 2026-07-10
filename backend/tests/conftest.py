from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockta.data.provider import DataProvider
from stockta.inference.predictor import Prediction


class FakeProvider(DataProvider):
    """測試用假 Provider，不打真實網路。"""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        sliced = self._frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return sliced


class FakePredictor:
    """測試用假 Predictor：固定回傳「漲」，不需要真實 artifact。"""

    version = "fake-1.0"
    metadata = {
        "model_name": "fake",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "metrics": {"test": {"macro_auc": 0.61}},
    }

    def predict(self, ohlcv: pd.DataFrame) -> Prediction:
        return Prediction(
            signal="漲",
            confidence=0.71,
            base_date=ohlcv.index[-1].date(),
            proba=[0.10, 0.19, 0.71],
        )


@pytest.fixture
def fake_ohlcv() -> pd.DataFrame:
    # 以「現在」為錨點往回推，確保無論實際執行日期為何都能涵蓋 last_completed_trading_day()
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=600)
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
        },
        index=idx,
    )


@pytest.fixture
def random_walk_ohlcv() -> pd.DataFrame:
    """帶隨機波動的合成 OHLCV（特徵/標籤測試用；平盤資料會讓多數指標退化）。"""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=600)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    spread = np.abs(rng.normal(0, 0.01, len(idx)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.005, len(idx))),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, len(idx)).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def client(fake_ohlcv):
    from fastapi.testclient import TestClient

    from stockta.api.main import app

    with TestClient(app) as test_client:
        app.state.limiter.enabled = False  # 測試逐案累計會誤觸限流
        app.state.data_provider = FakeProvider(fake_ohlcv)
        # 測試不依賴本機是否已訓練出 artifact：預設走 mock 路徑
        app.state.predictor = None
        app.state.prediction_store = None
        yield test_client


@pytest.fixture
def client_with_model(fake_ohlcv):
    from fastapi.testclient import TestClient

    from stockta.api.main import app

    with TestClient(app) as test_client:
        app.state.limiter.enabled = False
        app.state.data_provider = FakeProvider(fake_ohlcv)
        app.state.predictor = FakePredictor()
        app.state.prediction_store = None
        yield test_client
