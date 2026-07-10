from datetime import date

from stockta.inference.store import PredictionStore


def test_same_ticker_base_date_version_recorded_once(tmp_path):
    store = PredictionStore(tmp_path / "p.db")
    for _ in range(3):
        store.record("2330.TW", date(2026, 7, 9), "漲", 0.7, "xgb-2026-07-10")
    assert store.count() == 1

    store.record("2330.TW", date(2026, 7, 8), "跌", 0.6, "xgb-2026-07-10")
    assert store.count() == 2
