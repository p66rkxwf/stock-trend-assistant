from datetime import date

from stockta.inference.store import PredictionStore, RankPredictionStore


def test_same_ticker_base_date_version_recorded_once(tmp_path):
    store = PredictionStore(tmp_path / "p.db")
    for _ in range(3):
        store.record("2330.TW", date(2026, 7, 9), "漲", 0.7, "xgb-2026-07-10")
    assert store.count() == 1

    store.record("2330.TW", date(2026, 7, 8), "跌", 0.6, "xgb-2026-07-10")
    assert store.count() == 2


def test_rank_store_dedups_on_ticker_base_date_version(tmp_path):
    store = RankPredictionStore(tmp_path / "p.db")
    for _ in range(3):
        store.record("2330.TW", date(2026, 7, 9), 0.61, "xgb-cs-2026-07-26")
    assert store.count() == 1  # 同 (ticker, 基準日, 版本) 只記第一筆

    # 不同基準日、不同版本各自成一筆
    store.record("2330.TW", date(2026, 7, 10), 0.55, "xgb-cs-2026-07-26")
    store.record("2330.TW", date(2026, 7, 9), 0.61, "rf-cs-2026-07-26")
    assert store.count() == 3


def test_rank_store_coexists_with_prediction_store(tmp_path):
    # 兩 store 共用同一 DB、各自的表互不干擾
    db = tmp_path / "predictions.db"
    preds = PredictionStore(db)
    ranks = RankPredictionStore(db)
    preds.record("2330.TW", date(2026, 7, 9), "漲", 0.7, "lstm-2026-07-18+cal")
    ranks.record("2330.TW", date(2026, 7, 9), 0.61, "xgb-cs-2026-07-26")
    assert preds.count() == 1
    assert ranks.count() == 1
