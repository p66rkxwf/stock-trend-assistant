# 股價趨勢預測與投資助理系統

彰師大 115 年百萬專題探索（跨域整合類）。詳細開發計畫見 [PLAN.md](./PLAN.md)。

台灣 50 成分股 × 25 維特徵（17 個股技術指標 ＋ 6 大盤情境 ＋ 2 相對強弱）
× 五模型比較（RF / XGBoost / LSTM / GRU / TCN），FastAPI 後端 + Next.js 前端。

**公開站：<https://stock.sekinv.com>**（靜態站，每個交易日收盤後自動更新）。部署架構、狀態分支與每日排程見 [docs/deploy.md](docs/deploy.md)。

系統有**兩條預測軌**，兩條都每日落地、都在線上持續驗證：

1. **絕對方向**（原始主線）：未來 5 個交易日的三分類（漲 / 跌 / 觀望）＋風險等級。
2. **相對強弱排序**（2026-07 起的正面主線）：未來 5 日**是否贏過當日全池中位數**。
   絕對方向贏不了多頭 beta 是實測診斷出來的結論（見 `docs/experiment_log.md` #8），
   排序問法在原理上不受大盤漲跌影響。

> 本專案的紀律：**選型與門檻校準只用驗證期，測試期只做最終驗證**，每次動到測試期
> 都記進 [測試集曝光帳本](docs/test_set_ledger.md)；被否定的路線一樣寫成實驗記錄。
> 引用任何歷史測試期數字時，請一併看[線上實證](docs/online_predictions.md)——
> 那是全專案唯一沒有選擇偏差的數字，而它明顯比回測難看。

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
pytest              # 135 項：標籤/無前視/特徵 parity/可用時點/成分股/快照雜湊/
                    #          打亂標籤/API 契約/torch 模型/快取/日曆/registry/store
pytest -m leakage   # 49 項洩漏防治：打亂標籤 + 竄改未來逐值比對（pre-push 閘門跑的就是這組）
```

洩漏防治的兩條測試互補、缺一不可：打亂標籤抓「標籤資訊回流」（跨切分重複樣本、
標籤衍生特徵），竄改未來抓「前視特徵」。安裝 pre-push 閘門（測不過不准 push）：

```powershell
powershell -File scripts/install_hooks.ps1
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

- `GET /health`、`GET /api/stocks`、`GET /api/model`、`GET /api/market`
- `GET /api/stocks/{ticker}/candles?range=1mo|3mo|6mo|1y|2y|5y`、`/indicators`、`/history`
- `GET /api/stocks/{ticker}/prediction` — 真實模型推論；找不到 artifact 時退回 mock（`is_mock: true`）
- `GET /api/stocks/{ticker}/predictions` — 該檔的歷史預測與到期結果
- `GET /api/scan` — 全池掃描（絕對方向）
- `GET /api/rank`、`GET /api/rank/summary` — 相對強弱排序
- `GET /api/track-record` — 線上實證的即時戰績（不是回測數字）

## 啟動前端

```powershell
cd frontend
npm install
npm run dev                                     # http://localhost:3000（後端需先啟動）
```

代號搜尋（清單來自 `/api/stocks`）→ K 線圖（lightweight-charts，台股紅漲綠跌、range 切換）
＋預測卡片（訊號/信心/模型版本/測試 AUC/免責聲明）＋風險徽章。

## 目前進度

### 第一輪（2026-07-11）系統建置

| Phase | 內容 | 狀態 |
|---|---|---|
| 0–3 | package 骨架、API 契約、資料/特徵/標籤、RF+XGBoost 基線 | ✅ |
| 4 | LSTM / GRU / TCN（PyTorch CUDA，causal dilated TCN 自實作） | ✅ |
| 5 | 超參數搜尋 + 五模型比較 | ✅ |
| 6 | FastAPI 真實推論、風險等級、預測落地 predictions.db | ✅ |
| 7 | Next.js 前端 | ✅ |
| 8 | 端對端驗證（404/422/503 契約、真實預測） | ✅ |
| 9 | 技術文件 | ✅ |

### 第二輪（2026-07~08）模型與方法論

完整裁決記錄見 [實驗記錄](docs/experiment_log.md) #1–#9。

| 主題 | 結果 |
|---|---|
| 信心門檻校準（#1） | 採用；只用驗證期網格搜尋，`resolve_signal()` 為唯一決策規則 |
| 五模型集成（#2）、籌碼面（#6）、triple-barrier（#7b）、regime 特徵（#9） | **不採用**，四個負結果都寫成記錄 |
| walk-forward 週期性重訓（#3、#9） | 採用；**production = `gru-2026-08-13+cal`**（驗證 macro AUC 0.6712），每半年重訓 |
| **cross-sectional 相對強弱（#8）** | 採用，成為正面主線；回測測試期 Rank IC **+0.0442（t=2.3）** |
| 每日排程 + 自癒 catch-up | 收盤後自動落地預測；當日 K 線未到位時冪等補回（以「標的×日」為單位）。2026-09 起由 GitHub Actions 執行（台北 15:20、16:50 補跑），取代 Windows 工作排程器 |

### 第三輪（2026-08-16）資料誠信與洩漏防治

這一輪不追分數，只追「數字站不站得住」。

| 主題 | 產出 |
|---|---|
| 打亂標籤測試 | [leakage_report.md](docs/leakage_report.md)：5 個置換種子的虛無分佈；兩種刻意注入的洩漏都被抓到 |
| 倖存者偏誤 | [survivorship_study.md](docs/survivorship_study.md)：問題已量化（池與官方名單 15 處不一致），**PIT 資料尚未補齊** |
| 資料快照雜湊 | `data_cache/MANIFEST.json` + artifact metadata 的 `data_manifest_sha` |
| 資料源修訂實測 | [data_revision_study.md](docs/data_revision_study.md)：實測 yfinance 會回頭改寫歷史 |
| 特徵可用時點 | [feature_availability.md](docs/feature_availability.md)：新增特徵忘了標註就紅燈 |
| 測試集曝光帳本 | [test_set_ledger.md](docs/test_set_ledger.md)：9 次動到測試期，2 次影響決策 |

### 文件索引

- 方法論：[實驗記錄](docs/experiment_log.md)、[測試集曝光帳本](docs/test_set_ledger.md)、[洩漏報告](docs/leakage_report.md)、[倖存者偏誤](docs/survivorship_study.md)、[資料源修訂](docs/data_revision_study.md)、[特徵可用時點](docs/feature_availability.md)
- 模型與回測：[模型比較](docs/model_comparison.md)、[回測報告](docs/backtest_report.md)、[相對強弱](docs/cross_sectional_report.md)、[walk-forward](docs/experiment_walkforward.md)
- 線上實證：[絕對方向](docs/online_predictions.md)、[線上 Rank IC](docs/online_rank_ic.md)
- 系統：[架構](docs/architecture.md)、[特徵設計](docs/feature_engineering.md)、[資料探索](docs/data_exploration.md)

> 免責聲明：本系統為學術專題，預測結果不構成投資建議。
