# 基於深度學習之股價趨勢預測與投資助理系統 — 完整開發計畫

## Context（背景）

彰師大 115 年百萬專題探索申請案（跨域整合類，資工×財金 5 人團隊，指導老師林逸程），目標參加資訊應用服務創新競賽／全國大專院校產學創新實作競賽。申請表已定義技術棧與 A–I 九項學習進度，本計畫將其落實為可執行的開發藍圖，從零建置整個系統。

**已確認需求：**
- 目標市場：**台股**（yfinance 代號格式 `2330.TW`，前端顯示新台幣）
- 預測任務：**未來 5 個交易日趨勢三分類** — 漲（累積報酬 > +2%）／跌（< −2%）／觀望（區間內），附信心機率
- 風險等級：依歷史波動率（年化）換算低／中／高
- 訓練環境：**本機 Windows 11 + PyTorch（CPU）**，Python 3.13.9、Node v24.13.1
- 技術棧（依申請表）：yfinance、pandas-ta、RF/XGBoost 基線、LSTM/GRU/TCN、FastAPI、Next.js + lightweight-charts

## 架構原則（本次修訂新增）

以下五條原則貫穿整份計畫，是 5 人分工不互相踩腳、模型與服務不脫節的地基：

1. **單一 Python package**：`ml/` 與 `app/` 不是兩個平行頂層目錄，而是同一個可安裝 package（`pip install -e .`）內的子模組，根治 import 路徑問題（uvicorn 與 pytest 啟動路徑一致）。
2. **外部資料源藏在介面後**：所有 yfinance 存取只透過 `DataProvider` 介面，內建「先查快取→過期才連網→連網失敗退回快取」的降級策略。訓練、推論、測試都只認識這個介面（測試注入假 Provider，不打網路）。
3. **訓練產物有版本契約**：模型存檔時一併寫入 `metadata.json`（特徵清單與順序、視窗長度、標籤門檻、`auto_adjust` 設定、訓練日期、測試指標）。API 啟動時比對當前特徵產出，不一致就拒絕啟動——殺掉「舊模型配新特徵默默出錯」這一整類 bug。
4. **特徵管線單一入口**：`build_features()` 是訓練與推論唯一共用進入點，並以測試保證兩條路徑逐值相等，杜絕 training/serving skew。
5. **設定單一事實來源**：股票池、視窗長度、標籤門檻、風險分位、`auto_adjust` 全部集中在 `config.py`；前端股票清單改由 `GET /api/stocks` 取得，不自己內建。

## 專案結構

核心改動：`ml/` 與 `app/` 收進**同一個 package `stockta/`**，並把「訓練與推論都需要」的東西（資料存取、特徵、設定）提升為共用層。

```
stock-trend-assistant/
├── backend/
│   ├── pyproject.toml              # 以 package 安裝（pip install -e .），根治 import 問題
│   ├── requirements.txt / 依賴鎖定
│   ├── stockta/                    # 單一 package（訓練與服務共用）
│   │   ├── config.py               # 唯一設定來源：股票池、視窗、標籤門檻、風險分位、auto_adjust
│   │   ├── data/
│   │   │   ├── provider.py         # DataProvider 介面 + YFinanceProvider（含快取與失敗降級）
│   │   │   ├── cache.py            # parquet 讀寫（ticker 驗證後才進得來）
│   │   │   └── calendar.py         # last_completed_trading_day()、暖機期長度計算（台北時區）
│   │   ├── features/
│   │   │   └── pipeline.py         # build_features()：訓練/推論唯一共用入口（進度 B）
│   │   ├── ml/                     # 訓練管線（進度 A–E）
│   │   │   ├── labeling.py         # 5 日趨勢三分類標籤
│   │   │   ├── dataset.py          # 滑動視窗切割、時間序列 train/val/test 分割、標準化
│   │   │   ├── models/
│   │   │   │   ├── base.py         # 統一 predict_proba 介面
│   │   │   │   ├── baselines.py    # RandomForest、XGBoost
│   │   │   │   └── lstm.py gru.py tcn.py   # PyTorch（TCN 以 causal dilated Conv1d 自行實作）
│   │   │   ├── train.py            # 統一訓練入口（--model lstm|gru|tcn|rf|xgb）
│   │   │   ├── evaluate.py         # Accuracy、macro AUC-ROC（one-vs-rest）、混淆矩陣
│   │   │   ├── compare.py          # 產出模型比較表（Markdown + 圖）
│   │   │   └── registry.py         # artifact 存取 + metadata + 啟動一致性檢查
│   │   ├── inference/
│   │   │   ├── predictor.py        # 載入 artifact → predict(ticker)（吃 DataProvider）
│   │   │   ├── risk.py             # 波動率 → 低/中/高（純函式，門檻讀 config）
│   │   │   └── store.py            # 預測結果落地 SQLite（append-only）
│   │   └── api/                    # FastAPI 服務（進度 F）
│   │       ├── main.py             # lifespan 載入模型、CORS、限流
│   │       ├── schemas.py          # Pydantic 回應模型（Phase 0 凍結）
│   │       ├── errors.py           # 統一錯誤回應格式與 status code
│   │       ├── deps.py             # ticker 白名單驗證（FastAPI Depends，兩支 API 共用）
│   │       └── routers/
│   │           ├── stocks.py       # /candles、/prediction
│   │           └── meta.py         # /health、/api/model、/api/stocks
│   ├── artifacts/                  # 模型權重（gitignore 大檔）；metadata.json 進版控
│   ├── data_cache/                 # parquet 快取（gitignore）
│   ├── predictions.db              # SQLite 線上預測紀錄（gitignore）
│   └── tests/                      # pytest
│       ├── test_labeling.py        # 標籤計算正確
│       ├── test_no_lookahead.py    # 無前視偏差
│       ├── test_feature_parity.py  # 訓練/推論特徵逐值一致
│       └── test_api.py             # 注入 FakeProvider，不打網路
├── frontend/                       # Next.js（進度 G，create-next-app + TypeScript + Tailwind）
│   ├── app/page.tsx                # 主儀表板
│   ├── components/
│   │   ├── CandleChart.tsx         # lightweight-charts K 線 + 預測標示
│   │   ├── PredictionCard.tsx      # 漲/跌/觀望 + 信心機率 + 模型版本/AUC + 免責聲明
│   │   ├── RiskBadge.tsx           # 低/中/高風險
│   │   └── TickerSearch.tsx        # 股票代號輸入（清單改吃 GET /api/stocks）
│   └── lib/api.ts                  # 後端 API client
├── docs/                           # 進度 I：系統架構圖、模型比較報告、特徵設計說明
└── README.md
```

另外執行 `git init` 建立版本控制（目前非 git repo），`.gitignore` 排除 `.venv`、`node_modules`、`artifacts/*.pt`、`data_cache/`、`predictions.db`。

## 開發階段（對應申請表進度 A–I）

### Phase 0：專案初始化 + API 契約凍結

- `git init`、建立上述 `stockta/` package 骨架、`backend/.venv`（Python 3.13）、`pyproject.toml`（可 `pip install -e .`）、`requirements.txt`
- 關鍵相依：`torch`（CPU wheel）、`yfinance`、`pandas`、`scikit-learn`、`xgboost`、`fastapi`、`uvicorn`、`pyarrow`、`slowapi`（限流）
- ⚠️ **pandas-ta 相容性風險**：PyPI 版 0.3.14b0 與 numpy≥2 不相容（`numpy.NaN` import 錯誤），Python 3.13 無法退回 numpy 1.x。對策：安裝 GitHub 開發版 pandas-ta；若仍失敗則以 pandas 自行實作指標（MA/RSI/MACD/BBands 公式皆簡單），申請表精神不變
- **凍結 API 契約（解鎖 5 人並行開發的關鍵）**：先寫定 `schemas.py` 與 `errors.py`，後端出**假資料 mock endpoint**，前端從第一週即可對接，不必等到 Phase 6。錯誤格式統一為 `{"error": {"code": ..., "message": ...}}`：404 = 代號不存在、422 = 格式錯誤、503 = 資料源失敗

### Phase 1（A）：資料擷取與探索

- **先建 `DataProvider` 介面**（約 30 行）：`get_ohlcv(ticker, start, end) -> DataFrame`。`YFinanceProvider` 實作並內建「先查 parquet 快取→過期才連網→連網失敗退回快取」；訓練與 API 都只認識這個介面
- 股票池：台灣 50 成分股（約 50 檔，`config.py` 維護清單，作為唯一事實來源），透過 Provider 抓取近 10 年日線 OHLCV，快取成 parquet。固定 `auto_adjust`（台股除權息頻繁，訓練/推論必須一致，寫入 config）
- 定義**快取更新策略**（增量或整包重抓擇一並文件化），確保團隊成員快取內容一致，模型比較數字可重現
- 產出 `docs/data_exploration.md`：價格分佈、缺值情況、基本統計量（申請表要求的資料探索報告）

### Phase 2（B）：特徵工程

- 指標：MA(5/10/20/60)、RSI(14)、MACD(12,26,9)、Bollinger Bands(20,2)、KD 隨機指標、成交量變化率、日報酬率
- **特徵管線單一入口 `build_features(ohlcv) -> DataFrame`**：訓練與推論唯一共用；`test_feature_parity.py` 保證兩條路徑逐值相等
- 價格類特徵轉為相對值（如收盤價/MA − 1）避免尺度問題；StandardScaler **只 fit 訓練集**（防資料洩漏）

### Phase 3（C）：標籤與基線模型 + artifact 版本契約

- 標籤：未來 5 日累積報酬 >+2% 漲／<−2% 跌／其餘觀望；先統計三類分佈，若嚴重不平衡則微調門檻或用 class weight
- 分割：**依時間切割**（如 2016–2022 訓練、2023 驗證、2024–2025 測試），絕不隨機打亂
- **建立 `registry.py`（在第一個模型存檔前就位）**：存 artifact 時一併寫 `metadata.json`（特徵名稱清單含順序、視窗長度、標籤門檻、`auto_adjust`、訓練日期、測試指標、git commit hash）。載入時比對當前 `build_features` 產出的特徵名稱，不一致就拒絕啟動
- RF、XGBoost 以攤平的 60 日視窗特徵訓練，記錄 Accuracy 與 macro AUC-ROC 作為基準

### Phase 4（D）：深度學習模型

- LSTM、GRU、TCN 三模型統一介面（`models/base.py`）：輸入 `(batch, 60, n_features)`，輸出 3 類 logits
- 訓練：Adam、early stopping（驗證集 loss）、class weight 處理不平衡；CPU 可行（日線資料量不大），單一模型預估數分鐘～數十分鐘
- 產出五模型效能比較表（`compare.py`）

### Phase 5（E）：模型選定與調參

- 對最佳模型做超參數搜尋（隱藏維度、層數、learning rate、視窗長度；用驗證集），確認測試集表現穩定優於隨機基線與多數類基線
- 最終模型 + scaler + `metadata.json` 經 `registry.py` 存入 `artifacts/`，供 API 載入

### Phase 6（F）：FastAPI 後端 + 推論層

- **推論基準日**：以 `last_completed_trading_day()`（台北時區，13:30 收盤後才算當日完成）為錨點，避免盤中拿到未完成 K 棒。抓取範圍 = 視窗 60 + 指標暖機 60 + 緩衝，避免暖機不足產生 NaN
- `GET /api/stocks/{ticker}/candles?range=1y`：K 線資料（lightweight-charts 格式）
- `GET /api/stocks/{ticker}/prediction`：基準日資料→`build_features`→`predictor`→回傳 `{signal, confidence, risk, model_version, base_date}`；風險等級以近 60 日年化波動率按台股經驗分位切檔（分位門檻讀 `config.py`）
- **兩支 API 共用同一 DataProvider 與快取**（同一檔股票不重複抓）；預測以「ticker + 基準日」為快取 key，每檔每天只算一次
- **預測落地**：每次推論經 `store.py` append 到 SQLite（ticker、base_date、signal、confidence、model_version、created_at）——競賽前累積數週「線上預測 vs 實際走勢」實證
- 補三支便宜 endpoint：`GET /api/stocks`（股票池，前端清單來源）、`GET /health`（模型是否載入成功，demo 前自檢）、`GET /api/model`（版本、訓練日、測試 AUC）
- ticker 白名單驗證放 `deps.py`（`Depends`），一處驗證兩支 API 共用（同時防路徑穿越）；`slowapi` 每 IP 限流保護 Yahoo 抓取；CORS 明確 origin 清單，勿用 `*`

### Phase 7（G）：Next.js 前端

- `create-next-app`（TypeScript + Tailwind）+ `lightweight-charts`
- 主頁：代號搜尋（清單來自 `GET /api/stocks`）→ K 線圖（縮放、時間範圍切換）＋預測卡片（漲紅/跌綠依台股慣例、信心機率、模型版本與測試 AUC、投資免責聲明）＋風險徽章

### Phase 8（H）：整合與端對端驗證

- 驗證 Phase 0 凍結的錯誤契約：無效代號（404）、資料不足（422）、資料源失敗（503）、載入狀態
- 端對端流程：輸入 2330.TW → K 線與預測正確對應顯示；確認前後端錯誤處理一致

### Phase 9（I）：文件

- `docs/`：系統架構圖（mermaid）、模型比較報告、特徵工程設計說明、**線上預測實證數據**（來自 SQLite）、README 安裝執行指南 — 直接可用於專題報告與競賽文件

## 關鍵技術決策與風險

| 項目 | 決策 | 理由 |
|---|---|---|
| 標準化 | scaler 只 fit 訓練期 | 防止前視偏差，這是財金 ML 最常見錯誤 |
| 資料分割 | 純時間切割 | 隨機切割會讓模型「偷看未來」，比較表失真 |
| 類別不平衡 | class weight + 門檻調整 | ±2% 門檻下「觀望」可能過半 |
| pandas-ta | GitHub 版，備案自行實作 | numpy 2 相容性問題 |
| 準確率預期 | 略優於基線即達標 | 申請表自評標準即「優於隨機猜測 0.50」；股價預測本質困難，報告誠實呈現 |
| **外部資料源** | **藏在 DataProvider 介面後 + 快取降級** | **yfinance 是非官方爬蟲，會限流/改版；直接同步依賴會讓 demo 白屏，且無介面難以測試** |
| **訓練/推論一致** | **artifact metadata + 啟動比對特徵清單** | **改特徵後舊模型會默默吃錯特徵、輸出無錯誤訊息的垃圾預測；5 人分工幾乎必然踩到** |
| **特徵重用** | **build_features 單一入口 + parity 測試** | **training/serving skew 是產線 ML 最隱蔽的 bug** |
| **推論基準日** | **last_completed_trading_day（收盤後才算）** | **盤中拿到未完成日 K，同股同日給出不同預測；暖機不足產生 NaN 偏差** |
| **除權息調整** | **固定 auto_adjust 並寫入 config** | **台股除權息頻繁，訓練/推論調整方式不一致會讓除息日前後特徵與標籤全失真** |
| **股票清單** | **只存後端，前端由 API 取得** | **兩份清單必然漂移，使用者會點到模型沒訓練過的股票** |
| **預測落地** | **每次推論 append SQLite** | **線上實證比測試集 AUC 更有競賽說服力；也是未來 watchlist/回測的資料基礎** |
| **範圍克制** | **不做 Docker/MQ/正式 DB/使用者系統/MLflow** | **競賽專題規模，SQLite 檔案即足夠；過度基礎設施是負債** |

## 安全性與合規（本機 demo 範圍）

不做登入/授權/角色系統（正確的範圍克制），但以下四點即使 demo 也處理：

1. **ticker 輸入驗證**：`^\d{4,6}\.TW$` 或直接對股票池白名單驗證，防路徑穿越（ticker 會進快取檔名）
2. **抓取限流**：`slowapi` 每 IP 限流，避免被人透過 `/prediction` 放大攻擊 Yahoo 導致 demo 機 IP 被封
3. **CORS**：明確 origin 清單，勿 `allow_origins=["*"]`
4. **投資免責聲明**：UI 與 API 回應皆含免責聲明（台灣《投信投顧法》規範投資建議，競賽展示時有此行為加分項）

## 驗證方式

1. `pytest backend/tests/`：標籤計算正確、特徵無前視、**訓練/推論特徵逐值一致（parity）**、API 回應格式（注入 FakeProvider 不打網路）
2. 訓練管線煙霧測試：以 3 檔股票 × 2 年資料跑通 `train.py` 全部五種模型
3. **artifact 契約測試**：故意改動特徵清單，確認 API 啟動時如期拒絕載入
4. `uvicorn stockta.api.main:app` 啟動後端，curl 驗證 `/candles`、`/prediction`、`/health`、`/api/stocks`、`/api/model` 回應
5. `npm run dev` 啟動前端，瀏覽器實測輸入 2330.TW 端對端顯示 K 線＋預測＋風險
6. 完整訓練跑完後檢查 `compare.py` 比較表數據合理（AUC 介於 0.5–0.7 屬正常範圍）
7. 確認 SQLite 有逐次累積預測紀錄，供競賽實證使用

## 執行順序建議

**第一週的地基（做對成本最低、事後改成本最高）：**

1. **單一 package + `pyproject.toml`** — 目錄骨架是所有程式碼的地基，事後搬目錄要改遍 import
2. **DataProvider 介面 + 快取降級** — Phase 1 第一天就要抓資料，現在放介面後幾乎零成本，同時解掉可測試性
3. **API 契約凍結 + mock endpoint** — 5 人分工的解鎖鍵，晚一週定前端就閒置一週
4. **artifact metadata + 啟動檢查** — 必須在 Phase 3 第一個模型存檔前就位，否則已存 artifacts 全缺 metadata 得重訓
5. **預測落地 SQLite** — 工作量最小但越早累積越有價值，Phase 6 API 上線第一天就開始記

其中 1、2、4 相互增強：package 結構讓 provider 與 registry 有地方住，registry 讓「重用 ml/ 模組」從口號變成有機制保證的契約。

**批次順序：** Phase 0–3 為第一批（地基＋資料＋基線，約佔工作量 40%），Phase 4–5 第二批（深度模型），Phase 6–8 第三批（系統），Phase 9 收尾。每個 Phase 完成即 git commit，方便團隊 5 人分工接手（資工 3 人：模型＋後端＋前端；財金 2 人：特徵選型依據、風險定義、報告）。因 API 契約在 Phase 0 已凍結，前端可在第一批期間即以 mock 並行開發，Phase 8 整合風險從「大爆炸」降為持續驗證。
