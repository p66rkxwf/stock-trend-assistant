def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_stocks_includes_known_ticker(client):
    r = client.get("/api/stocks")
    assert r.status_code == 200
    tickers = [s["ticker"] for s in r.json()["stocks"]]
    assert "2330.TW" in tickers


def test_model_info_marked_as_mock(client):
    r = client.get("/api/model")
    assert r.status_code == 200
    assert r.json()["is_mock"] is True


def test_candles_valid_ticker(client):
    r = client.get("/api/stocks/2330.TW/candles?range=1y")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "2330.TW"
    assert len(body["candles"]) > 0


def test_candles_unknown_ticker_returns_404(client):
    r = client.get("/api/stocks/9999.TW/candles")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "TICKER_NOT_FOUND"


def test_candles_invalid_format_returns_422(client):
    r = client.get("/api/stocks/not-a-ticker/candles")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_TICKER_FORMAT"


def test_candles_invalid_range_returns_422(client):
    r = client.get("/api/stocks/2330.TW/candles?range=bogus")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_RANGE"


def test_prediction_mock_response_shape(client):
    r = client.get("/api/stocks/2330.TW/prediction")
    assert r.status_code == 200
    body = r.json()
    assert body["signal"] in ("漲", "跌", "觀望")
    assert body["risk"] in ("低", "中", "高")
    assert body["is_mock"] is True


def test_prediction_with_loaded_model(client_with_model):
    r = client_with_model.get("/api/stocks/2330.TW/prediction")
    assert r.status_code == 200
    body = r.json()
    assert body["is_mock"] is False
    assert body["model_version"] == "fake-1.0"
    assert body["signal"] == "漲"
    assert body["confidence"] == 0.71
    assert body["risk"] in ("低", "中", "高")


def test_prediction_includes_proba(client_with_model):
    r = client_with_model.get("/api/stocks/2330.TW/prediction")
    assert r.status_code == 200
    assert r.json()["proba"] == {"跌": 0.10, "觀望": 0.19, "漲": 0.71}


def test_prediction_mock_has_no_proba(client):
    r = client.get("/api/stocks/2330.TW/prediction")
    assert r.status_code == 200
    assert r.json()["proba"] is None


def test_indicators_snapshot(client):
    r = client.get("/api/stocks/2330.TW/indicators")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "2330.TW"
    assert 0.0 <= body["rsi14"] <= 1.0
    assert 0.0 <= body["kd_k"] <= 1.0
    for field in ("macd_hist", "bb_pctb", "vol_ratio", "ma_bias_5", "ma_bias_20", "ma_bias_60"):
        assert field in body


def test_scan_mock_mode(client):
    r = client.get("/api/scan")
    assert r.status_code == 200
    body = r.json()
    assert body["is_mock"] is True
    assert body["results"] == []


def test_scan_with_model(client_with_model):
    r = client_with_model.get("/api/scan")
    assert r.status_code == 200
    body = r.json()
    assert body["is_mock"] is False
    assert len(body["results"]) > 0
    # FakePredictor 固定回「漲」，計數與逐筆訊號需一致
    assert body["up"] == len(body["results"])
    assert body["up"] + body["hold"] + body["down"] == len(body["results"])
    first = body["results"][0]
    assert first["signal"] in ("漲", "跌", "觀望")
    assert set(first["proba"]) == {"漲", "跌", "觀望"}


def test_market_snapshot(client):
    r = client.get("/api/market")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["breadth_up"] <= 1.0
    assert "ret_1d" in body and "ma20_bias" in body and "vol20" in body


def test_prediction_history_empty_db(client, monkeypatch, tmp_path):
    from stockta.api.routers import stocks as stocks_router

    monkeypatch.setattr(stocks_router, "PREDICTIONS_DB_PATH", tmp_path / "none.db")
    r = client.get("/api/stocks/2330.TW/predictions")
    assert r.status_code == 200
    assert r.json() == {"ticker": "2330.TW", "records": []}


def test_track_record_empty_db(client, monkeypatch, tmp_path):
    from stockta.api.routers import meta as meta_router

    monkeypatch.setattr(meta_router, "PREDICTIONS_DB_PATH", tmp_path / "none.db")
    r = client.get("/api/track-record")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0 and body["matured"] == 0 and body["hit_rate"] is None


def test_model_info_with_loaded_model(client_with_model):
    r = client_with_model.get("/api/model")
    assert r.status_code == 200
    body = r.json()
    assert body["is_mock"] is False
    assert body["test_auc"] == 0.61


def test_health_reports_model_loaded(client_with_model):
    r = client_with_model.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
