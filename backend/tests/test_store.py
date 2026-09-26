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


def test_recorded_pairs_are_per_ticker_not_per_day(tmp_path):
    # 同一天只記到一檔時，其他檔仍須被視為「未記錄」——否則雲端抓取部分失敗的那幾檔永遠補不回來
    from stockta.ml.record_predictions import _recorded_pairs
    from stockta.ml.record_rank_predictions import _recorded_rank_pairs

    db = tmp_path / "predictions.db"
    PredictionStore(db).record("2330.TW", date(2026, 9, 24), "漲", 0.7, "gru-x")
    RankPredictionStore(db).record("2330.TW", date(2026, 9, 24), 0.6, "xgb-cs-x")

    assert _recorded_pairs(db, "gru-x") == {("2330.TW", "2026-09-24")}
    assert ("2317.TW", "2026-09-24") not in _recorded_pairs(db, "gru-x")
    assert _recorded_pairs(db, "other-version") == set()
    assert _recorded_rank_pairs(db, "xgb-cs-x") == {("2330.TW", "2026-09-24")}
