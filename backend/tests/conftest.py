from datetime import date

import pandas as pd
import pytest

from stockta.data.provider import DataProvider


class FakeProvider(DataProvider):
    """測試用假 Provider，不打真實網路。"""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        sliced = self._frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return sliced


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
def client(fake_ohlcv):
    from fastapi.testclient import TestClient

    from stockta.api.main import app

    with TestClient(app) as test_client:
        app.state.data_provider = FakeProvider(fake_ohlcv)
        yield test_client
