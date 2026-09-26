"""靜態站匯出：檔案結構與「寧可不部署也不發布錯的資料」的關卡。"""

import json
from datetime import date

import pytest

from stockta.export_static import TICKER_ENDPOINTS, ExportError, run_export

# FakePredictor 沒有 _scaler/_model，無法跑 point-in-time 的 /history；其餘端點照常
_ENDPOINTS_NO_HISTORY = tuple(e for e in TICKER_ENDPOINTS if e[1] != "history.json")
_LAST_DAY = date(2026, 9, 25)
_AS_OF = date(2026, 9, 24)


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    # /api/track-record 與 /predictions 讀 predictions.db：指到暫存檔，不碰（也不建立）真實資料庫
    from stockta.api.routers import meta as meta_router
    from stockta.api.routers import stocks as stocks_router

    monkeypatch.setattr(meta_router, "PREDICTIONS_DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(stocks_router, "PREDICTIONS_DB_PATH", tmp_path / "p.db")


def _export(client, out, **kw):
    kw.setdefault("tickers", ["2330.TW"])
    kw.setdefault("ticker_endpoints", _ENDPOINTS_NO_HISTORY)
    return run_export(
        client, out, last_day=_LAST_DAY, data_as_of=_AS_OF, scan_dates=[], log=lambda *_: None, **kw
    )


def test_writes_site_layout_and_meta(client_with_model, tmp_path):
    out = tmp_path / "data"
    meta = _export(client_with_model, out)

    for rel in [
        "model.json", "stocks.json", "market.json", "track-record.json",
        "scan/latest.json", "scan/index.json", "rank/latest.json", "rank/index.json", "rank/summary.json",
        "stocks/2330.TW/prediction.json", "stocks/2330.TW/indicators.json",
        "stocks/2330.TW/predictions.json", "stocks/2330.TW/candles.json", "meta.json",
    ]:
        assert (out / rel).is_file(), rel

    # 檔案內容＝API 回應原文
    pred = json.loads((out / "stocks/2330.TW/prediction.json").read_text(encoding="utf-8"))
    assert pred["ticker"] == "2330.TW" and pred["is_mock"] is False
    assert meta["data_as_of"] == "2026-09-24"
    assert meta["last_trading_day"] == "2026-09-25"
    assert meta["range_days"]["5y"] == 1825
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == meta


def test_refuses_mock_model(client, tmp_path):
    # 模型 artifact 不見時 API 退回 mock：絕不能把假預測發布出去
    with pytest.raises(ExportError, match="mock"):
        _export(client, tmp_path / "data")


def test_refuses_data_date_going_backwards(client_with_model, tmp_path):
    with pytest.raises(ExportError, match="倒退"):
        _export(client_with_model, tmp_path / "data", prev_meta={"data_as_of": "2026-09-30"})


def test_refuses_low_ticker_coverage(client_with_model, tmp_path):
    # 不在股票池的代號全數 404 → 覆蓋率 0%
    with pytest.raises(ExportError, match="覆蓋率"):
        _export(client_with_model, tmp_path / "data", tickers=["9999.TW"])
