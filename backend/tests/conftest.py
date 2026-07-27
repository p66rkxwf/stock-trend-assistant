from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockta.data.provider import DataProvider
from stockta.features.market import build_market_context
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

    def predict(self, ohlcv: pd.DataFrame, context: pd.DataFrame) -> Prediction:
        return Prediction(
            signal="漲",
            confidence=0.71,
            base_date=ohlcv.index[-1].date(),
            proba=[0.10, 0.19, 0.71],
        )


class FakeMarketContext:
    """測試用市場情境服務：固定回傳建構時給的 context。"""

    def __init__(self, context: pd.DataFrame):
        self._context = context

    def get(self, end: date) -> pd.DataFrame:
        return self._context


class _FakeCSModel:
    """測試用 cross-sectional 模型：predict_proba 回固定二欄機率。"""

    def predict_proba(self, X):
        return np.tile([0.45, 0.55], (len(X), 1))


class _IdentityScaler:
    def transform(self, X):
        return X


FAKE_CS_MODEL = (_FakeCSModel(), _IdentityScaler(), {"model_name": "fake-cs"})


def _random_walk(seed: int, periods: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=periods)
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
    return _random_walk(42)


@pytest.fixture
def market_ohlcv() -> pd.DataFrame:
    """合成大盤指數 OHLCV（市場情境特徵用，與個股走勢獨立）。"""
    return _random_walk(7)


@pytest.fixture
def market_context(market_ohlcv, random_walk_ohlcv) -> pd.DataFrame:
    """合成市場情境：大盤 + 三檔合成股的寬度（min_tickers 降為 2 以配合小池）。"""
    pool = {
        "2330.TW": random_walk_ohlcv,
        "2317.TW": _random_walk(11),
        "2454.TW": _random_walk(13),
    }
    return build_market_context(market_ohlcv, pool, min_tickers=2)


def _no_local_artifact(*args, **kwargs):
    """測試不得依賴本機 artifact（特徵欄位改版期間會觸發契約錯誤）——一律走 mock 路徑。"""
    raise FileNotFoundError("測試環境不載入 artifact")


@pytest.fixture
def client(fake_ohlcv, monkeypatch):
    from fastapi.testclient import TestClient

    from stockta.api import main as api_main

    import stockta.ml.cross_sectional as cs_mod

    monkeypatch.setattr(api_main.Predictor, "from_registry", _no_local_artifact)
    monkeypatch.setattr(cs_mod, "load_cs_model", _no_local_artifact)  # 測試不載真實 CS 模型
    with TestClient(api_main.app) as test_client:
        state = api_main.app.state
        state.limiter.enabled = False  # 測試逐案累計會誤觸限流
        state.data_provider = FakeProvider(fake_ohlcv)
        fake_ctx = build_market_context(fake_ohlcv, {"2330.TW": fake_ohlcv}, min_tickers=1)
        state.market_context = FakeMarketContext(fake_ctx)
        state.predictor = None
        state.prediction_store = None
        state.cs_model = None
        yield test_client


@pytest.fixture
def client_with_model(fake_ohlcv, monkeypatch):
    from fastapi.testclient import TestClient

    from stockta.api import main as api_main
    import stockta.ml.cross_sectional as cs_mod

    monkeypatch.setattr(api_main.Predictor, "from_registry", _no_local_artifact)
    monkeypatch.setattr(cs_mod, "load_cs_model", lambda *a, **k: FAKE_CS_MODEL)
    with TestClient(api_main.app) as test_client:
        state = api_main.app.state
        state.limiter.enabled = False
        state.data_provider = FakeProvider(fake_ohlcv)
        fake_ctx = build_market_context(fake_ohlcv, {"2330.TW": fake_ohlcv}, min_tickers=1)
        state.market_context = FakeMarketContext(fake_ctx)
        state.predictor = FakePredictor()
        state.prediction_store = None
        state.cs_model = FAKE_CS_MODEL
        yield test_client
