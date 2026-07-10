# 股價趨勢預測與投資助理系統

詳細開發計畫見 [PLAN.md](./PLAN.md)。

## Backend 開發環境設定（Windows）

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# PyTorch CPU 版需指定官方 CPU wheel 索引，Phase 4（深度學習模型）開始才需要：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

⚠️ **pandas-ta 相容性風險**：PyPI 版與 numpy≥2 不相容，Phase 2 特徵工程階段需改安裝
GitHub 開發版 pandas-ta，或自行實作指標（詳見 PLAN.md）。

## 執行測試

```powershell
cd backend
pytest
```

## 啟動 API（開發模式）

```powershell
cd backend
uvicorn stockta.api.main:app --reload
```

- `GET /health`、`GET /api/stocks`、`GET /api/model`
- `GET /api/stocks/{ticker}/candles?range=1y`
- `GET /api/stocks/{ticker}/prediction` — **Phase 6 前為 mock 回應**（`is_mock: true`），供前端提前對接

## 目前進度

Phase 0（專案初始化 + API 契約凍結）已完成骨架：
- `backend/stockta/`：單一 package，`config.py` 為唯一設定來源
- `stockta/data/`：`DataProvider` 介面、yfinance 實作（含快取降級）、交易日曆
- `stockta/api/`：FastAPI 服務、Pydantic 契約（`schemas.py`）、統一錯誤格式（`errors.py`）
- `backend/tests/`：pytest，API 測試以 `FakeProvider` 注入，不打真實網路

Phase 1 起（資料擷取、特徵工程、標籤與模型）尚未實作，見 PLAN.md 各 Phase 說明。
