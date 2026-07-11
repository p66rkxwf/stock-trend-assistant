# 系統架構

## 整體架構圖

```mermaid
flowchart LR
    subgraph external["外部"]
        YF[(Yahoo Finance\nyfinance)]
    end

    subgraph backend["backend（stockta package）"]
        direction TB
        PROV["DataProvider 介面\nYFinanceProvider"]
        CACHE[("parquet 快取\ndata_cache/")]
        FEAT["build_features()\n17 個技術指標特徵\n（訓練/推論唯一共用入口）"]

        subgraph ml["訓練管線 ml/"]
            LABEL["labeling.py\n5 日 ±2% 三分類"]
            DS["dataset.py\n時間切割+embargo\nscaler 只 fit 訓練期"]
            MODELS["models/\nrf · xgb · lstm · gru · tcn"]
            REG["registry.py\nmetadata 版本契約"]
        end

        subgraph inference["推論層 inference/"]
            PRED["predictor.py"]
            RISK["risk.py 波動率→低/中/高"]
            STORE[("store.py\npredictions.db")]
        end

        subgraph api["FastAPI api/"]
            EP1["/api/stocks/{t}/candles"]
            EP2["/api/stocks/{t}/prediction"]
            EP3["/api/stocks · /health · /api/model"]
        end
    end

    subgraph frontend["frontend（Next.js）"]
        UI["CandleChart（lightweight-charts）\nPredictionCard · RiskBadge · TickerSearch"]
    end

    YF -->|"快取→過期才連網→失敗退回快取"| PROV
    PROV --> CACHE
    PROV --> FEAT
    FEAT --> DS
    LABEL --> DS
    DS --> MODELS
    MODELS -->|joblib + metadata.json| REG
    REG -->|"啟動比對特徵清單\n不一致拒絕載入"| PRED
    FEAT --> PRED
    PRED --> EP2
    PRED --> STORE
    RISK --> EP2
    PROV --> EP1
    EP1 & EP2 & EP3 -->|"REST JSON\n錯誤格式 {error:{code,message}}"| UI
```

## 五條架構原則的落點

| 原則 | 落點 |
|---|---|
| 單一 Python package | `stockta/` 可安裝（`pip install -e .`），uvicorn 與 pytest import 路徑一致 |
| 外部資料源藏介面後 | `DataProvider`；訓練/推論/測試只認識介面，測試注入 FakeProvider 不打網路 |
| 訓練產物版本契約 | `registry.py` 存檔寫 metadata（特徵清單/視窗/標籤門檻/auto_adjust），載入逐項比對，不一致拒絕啟動 |
| 特徵管線單一入口 | `build_features()`；`test_feature_parity.py` 保證訓練/推論逐值相等 |
| 設定單一事實來源 | `config.py`；前端股票清單吃 `GET /api/stocks`，不自建副本 |

## 訓練與推論的共用與隔離

- **共用**：DataProvider、build_features、config、registry——特徵計算只有一份程式碼，
  杜絕 training/serving skew。
- **隔離**：`ml/`（訓練，含重相依 torch）與 `api/`（服務）互不 import 對方；
  服務端只透過 registry 載入 artifact。
- **推論基準日**：`last_completed_trading_day()`（台北時區，13:30 收盤後才算當日完成），
  避免盤中拿到未完成 K 棒；抓取範圍 = 視窗 60 + 指標暖機 120 + 緩衝。

## 資料流（一次 /prediction 請求）

1. `deps.py` 白名單驗證 ticker（防路徑穿越，ticker 會進快取檔名）
2. Provider 取 OHLCV（快取新鮮→直接用；過期→連網更新；連網失敗→退回快取）
3. `build_features()` → 取最後 60 日視窗 → scaler（artifact 內）→ 模型 predict_proba
4. `risk.py` 以近 60 日年化波動率換算風險等級
5. 預測落地 `predictions.db`（同 ticker+基準日+版本去重）——累積線上實證
6. 回傳 `{signal, confidence, risk, model_version, base_date}`

> 環境備註：專案路徑含中文時 libcurl 讀不到 venv 內 CA 憑證，
> `config.py` 啟動時自動複製 cacert.pem 至 ASCII 路徑並設 `CURL_CA_BUNDLE`。
