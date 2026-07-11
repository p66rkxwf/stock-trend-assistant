# 股價趨勢預測與投資助理系統

彰師大 115 年百萬專題探索（跨域整合類）。詳細開發計畫見 [PLAN.md](./PLAN.md)。

台灣 50 成分股 × 17 個技術指標特徵 × 五模型比較（RF / XGBoost / LSTM / GRU / TCN）
→ 未來 5 個交易日趨勢三分類（漲 / 跌 / 觀望）＋風險等級，FastAPI 後端 + Next.js 前端。

## Backend 開發環境設定（Windows）

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate

# NVIDIA GPU（本專案開發機 RTX 5070 Ti）：先裝 cu128 版 torch，再裝套件本體
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
# 無 GPU 環境改用 CPU wheel：pip install torch --index-url https://download.pytorch.org/whl/cpu
```

技術指標以 pandas 自行實作（pandas-ta 與 numpy≥2 不相容，見 PLAN.md Phase 0 決策），
無需額外安裝指標套件。

## 執行測試

```powershell
cd backend
pytest        # 42 項：標籤/無前視/特徵 parity/API 契約/torch 模型/快取/日曆/registry/store
```

## 訓練與比較

```powershell
cd backend
python -m stockta.data.fetch                    # 抓取股票池日線（已抓過走 parquet 快取）
python -m stockta.ml.train --model rf           # rf | xgb | lstm | gru | tcn
python -m stockta.ml.tune  --model gru          # 深度模型超參數搜尋（val macro AUC 選優）
python -m stockta.ml.compare                    # 產出 docs/model_comparison.md
python -m stockta.ml.report_predictions         # 線上預測實證報告（讀 predictions.db）
```

## 啟動 API（開發模式）

```powershell
cd backend
uvicorn stockta.api.main:app --reload           # http://localhost:8000
```

- `GET /health`、`GET /api/stocks`、`GET /api/model`
- `GET /api/stocks/{ticker}/candles?range=1mo|3mo|6mo|1y|2y|5y`
- `GET /api/stocks/{ticker}/prediction` — 真實模型推論；找不到 artifact 時退回 mock（`is_mock: true`）

## 啟動前端

```powershell
cd frontend
npm install
npm run dev                                     # http://localhost:3000（後端需先啟動）
```

代號搜尋（清單來自 `/api/stocks`）→ K 線圖（lightweight-charts，台股紅漲綠跌、range 切換）
＋預測卡片（訊號/信心/模型版本/測試 AUC/免責聲明）＋風險徽章。

## 目前進度（2026-07-11）

| Phase | 內容 | 狀態 |
|---|---|---|
| 0–3 | package 骨架、API 契約、資料/特徵/標籤、RF+XGBoost 基線 | ✅ |
| 4 | LSTM / GRU / TCN（PyTorch CUDA，causal dilated TCN 自實作） | ✅ |
| 5 | 超參數搜尋 + 五模型比較 → **選型 GRU**（驗證 macro AUC 0.6698） | ✅ |
| 6 | FastAPI 真實推論、風險等級、預測落地 predictions.db | ✅ |
| 7 | Next.js 前端 | ✅ |
| 8 | 端對端驗證（404/422/503 契約、真實預測） | ✅ |
| 9 | 文件（[模型比較](docs/model_comparison.md)、[架構](docs/architecture.md)、[特徵設計](docs/feature_engineering.md)、[線上實證](docs/online_predictions.md)、[資料探索](docs/data_exploration.md)） | ✅ |

> 免責聲明：本系統為學術專題，預測結果不構成投資建議。
