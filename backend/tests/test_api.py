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
