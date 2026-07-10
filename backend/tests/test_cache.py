import pandas as pd
import pytest

from stockta.data.cache import InvalidTickerError, ParquetCache, validate_ticker


def test_validate_ticker_accepts_valid_format():
    assert validate_ticker("2330.TW") == "2330.TW"


@pytest.mark.parametrize("bad_ticker", ["../../etc/passwd", "2330", "AAPL", "2330.TW/../x", ""])
def test_validate_ticker_rejects_bad_input(bad_ticker):
    with pytest.raises(InvalidTickerError):
        validate_ticker(bad_ticker)


def test_cache_roundtrip(tmp_path):
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))

    cache.write("2330.TW", df)
    loaded = cache.read("2330.TW")

    assert loaded is not None
    assert list(loaded["close"]) == [1.0, 2.0]


def test_cache_read_missing_returns_none(tmp_path):
    cache = ParquetCache(tmp_path)
    assert cache.read("2330.TW") is None


def test_cache_is_fresh_false_when_missing(tmp_path):
    cache = ParquetCache(tmp_path)
    assert cache.is_fresh("2330.TW", max_age_days=1) is False
