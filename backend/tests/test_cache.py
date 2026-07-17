import pandas as pd
import pytest

from stockta.data.cache import InvalidTickerError, ParquetCache, validate_ticker


def test_validate_ticker_accepts_valid_format():
    assert validate_ticker("2330.TW") == "2330.TW"


@pytest.mark.parametrize("bad_ticker", ["../../etc/passwd", "2330", "AAPL", "2330.TW/../x", ""])
def test_validate_ticker_rejects_bad_input(bad_ticker):
    with pytest.raises(InvalidTickerError):
        validate_ticker(bad_ticker)


def test_validate_ticker_whitelists_market_index():
    assert validate_ticker("^TWII") == "^TWII"


@pytest.mark.parametrize("bad_ticker", ["^GSPC", "^TWII/../x", "^twii"])
def test_validate_ticker_rejects_other_carets(bad_ticker):
    # 白名單只放行精確的 ^TWII，其他含 ^ 的輸入一律拒絕（路徑穿越防線）
    with pytest.raises(InvalidTickerError):
        validate_ticker(bad_ticker)


def test_market_index_cache_filename_strips_caret(tmp_path):
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    cache.write("^TWII", df)
    assert (tmp_path / "TWII.parquet").exists()
    assert cache.read("^TWII") is not None


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
