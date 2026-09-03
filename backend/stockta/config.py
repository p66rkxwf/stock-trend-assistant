"""唯一設定來源：股票池、視窗長度、標籤門檻、風險分位、auto_adjust 等。

不要在 features/、api/、frontend 各自維護一份設定副本——一律從這裡讀取。
"""

import os
import shutil
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo


def _fix_curl_ca_bundle() -> None:
    """專案路徑含非 ASCII 字元（如「百萬專題」）時，libcurl 讀不到 venv 內
    certifi 的 CA 檔（curl error 77），yfinance/curl_cffi 的所有 HTTPS 都會失敗。
    對策：把 cacert.pem 複製到 ASCII 路徑並以 CURL_CA_BUNDLE 指定（已設定者不動）。
    """
    if os.environ.get("CURL_CA_BUNDLE"):
        return
    try:
        import certifi

        src = certifi.where()
        if src.isascii():
            return  # 路徑本來就沒問題
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        if not base.isascii():
            return  # 找不到 ASCII 落腳處，維持原行為（快取降級仍可運作）
        target = Path(base) / "stockta" / "cacert.pem"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, target)
        os.environ["CURL_CA_BUNDLE"] = str(target)
    except Exception:
        pass  # 盡力而為，不因此擋住啟動


_fix_curl_ca_bundle()

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = BACKEND_ROOT / "data_cache"
ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
ARTIFACTS_CS_DIR = BACKEND_ROOT / "artifacts_cs"  # cross-sectional 排序模型（與 3 類分開）
PREDICTIONS_DB_PATH = BACKEND_ROOT / "predictions.db"

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
MARKET_CLOSE_HOUR = 13
MARKET_CLOSE_MINUTE = 30

# yfinance 調整價設定：訓練與推論必須一致，否則除權息日前後特徵與標籤全部失真
AUTO_ADJUST = True

# 市場情境特徵的大盤指數（台股加權指數）；快取檔名映射見 data/cache.py
MARKET_INDEX_TICKER = "^TWII"
# 市場寬度（上漲家數比）當日有資料的成分股少於此數即視為不可信（NaN）
BREADTH_MIN_TICKERS = 30

# 市場狀態（regime）特徵開關（實驗 #9）：開啟後 context 由 6 欄增為 10 欄、
# 總特徵由 25 欄增為 29 欄。**切換此旗標必須重訓**——既有 artifact 的特徵契約
# 記錄的是舊欄位清單，不重訓就會在載入時被 registry 擋下（這是刻意的）。
MARKET_INCLUDE_REGIME = False

WINDOW_LENGTH_DAYS = 60
# EMA 類指標（RSI/MACD/KD）的值受序列起點影響，需足夠長的暖機期收斂後
# 推論（短序列）與訓練（全序列）的特徵才會一致；120 個交易日可讓 EMA(26)
# 殘差降到 1e-4 量級，test_feature_parity.py 依此容忍度把關。
INDICATOR_WARMUP_DAYS = 120
LABEL_HORIZON_DAYS = 5
LABEL_UP_THRESHOLD = 0.02
LABEL_DOWN_THRESHOLD = -0.02

# 標籤類別的唯一順序（模型 predict_proba 輸出欄位順序），與 artifact metadata 比對
LABEL_CLASSES = ["跌", "觀望", "漲"]

# 時間序列切分（絕不隨機打亂）：訓練 ≤ TRAIN_END、驗證 ≤ VAL_END、其後為測試
# 2026-07-17 walk-forward 重訓：原切分（train≤2022、val=2023）的模型到 2026 年
# 命中率明顯下滑（市場漂移），故將切分前移兩年重訓全部模型；舊 artifact 備份於
# backend/artifacts_backup_2022split/
# 2026-08-14 實驗 #9 再次前移：4 折 walk-forward 證實「凍結不重訓」的代價會隨時間放大——
# 最新半年（F4）重訓 45.2% vs 凍結 39.1%、喊跌精度 45.3% vs 28.4%。切分改採 F4 設定，
# 舊 artifact 備份於 backend/artifacts_backup_2024split/。**此後需定期重訓**（見
# docs/experiment_walkforward.md 的重訓節奏建議）。
SPLIT_TRAIN_END = "2025-06-30"
SPLIT_VAL_END = "2026-06-30"

# 訓練資料回溯年數。2026-07-18 由 10 年延長為 20 年：#5 延伸實驗顯示 19 年資料
# 在兩種子×兩架構共 6 組測試中，驗證期校準後命中率一致優於 8.5 年（+1.4~1.9pp）；
# 2015-06 漲跌幅 7%→10% 的結構斷點實測無害，涵蓋 2008 海嘯等罕見情境反而有益
# （詳 docs/experiment_log.md #5）。個別股票上市較晚者以實際可得資料訓練。
HISTORY_YEARS = 20

# API 載入的正式模型名稱（artifacts/<名稱>/），訓練比較後由 compare.py 結果決定
# 2026-07-11 首輪比較（train≤2022/val=2023）以驗證 Macro AUC 選 gru（0.6698）。
# 2026-07-17 walk-forward 重訓（train≤2024/val=2025）：lstm 0.6604 ≈ gru 0.6603，
#   以驗證期校準後命中率決勝改選 lstm（測試期 45.5% > 舊 gru 43.5%）。
# 2026-07-18 市場情境特徵（17→25 欄）重訓：以驗證 Macro AUC 選 gru
#   （0.6709 > lstm 0.6673；亦優於 17 欄基準 0.6604），驗證期原始命中率 47.0%
#   （17 欄基準 45.6%）；校準後 52.58% 與基準 52.6% 在 ±0.46% 噪音內持平——
#   新特徵使 argmax 天生平衡、不再依賴門檻救援（完整裁決見 docs/experiment_log.md）
# 2026-07-18(b) 資料區間 10→20 年重訓：驗證期 lstm ≈ gru 平手
#   （AUC 0.6706/0.6699、校準後 53.23%/53.25%，皆過 52.6% 門檻）；
#   測試期最終驗證 gru 退步（45.1%）、lstm 持平（47.6%）→ 依「不部署最終驗證
#   退步的模型」選 lstm。此為測試結果的防守性使用，已於 experiment_log #5 揭露
# 2026-08-14(#9) 切分前移重訓五模型：驗證期 Macro AUC gru 0.6712 > lstm 0.6684
#   > tcn 0.6680 > xgb 0.6629 > rf 0.6567，領先幅度與 #4 選 gru 時同量級，
#   且測試期最終驗證 gru 亦居首（0.6334）→ 選 gru，無需動用平手裁決規則
PRODUCTION_MODEL = "gru"

# === Cross-sectional 相對強弱排序（experiment_log #8；與上方 3 類絕對方向管線並存）===
# 標籤＝未來 LABEL_HORIZON_DAYS 日報酬是否贏過「當日全池中位數」（二分類，設計上約 50%）。
# 模型分數 = P(贏過中位數)，用於全池排序。sklearn RF/XGB（自動二分類）。
CS_PRODUCTION_MODEL = "xgb"  # 由 cross_sectional report 以驗證期 Rank IC 選出（rf≈xgb 皆 +0.0427，取 xgb）
CS_TOP_FRACTION = 0.2  # 組合回測做多分數前 1/5
CS_HOLDING_DAYS_CANDIDATES = (5, 10, 20)  # 掃描換股週期，以驗證期淨報酬選最佳
# 台股來回交易成本（bp）：手續費 0.1425%×2 + 賣出證交稅 0.3% ≈ 58.5bp
CS_COST_BPS = 58.5

# 方向訊號信心門檻：信心低於門檻的方向訊號一律降級為「觀望」——少喊、喊得準。
# 門檻僅以驗證期整體命中率網格搜尋選出（2026-08-14 對切分前移後的 gru 重新校準，
# 驗證期 2025-06-30~2026-06-30 共 11,656 筆：原始 48.3% → 校準後 50.8%）。
# 重新訓練或更換模型後，需執行 python -m stockta.ml.calibrate 重新校準本設定。
SIGNAL_CONFIDENCE_THRESHOLDS: dict[str, float] = {"跌": 0.42, "漲": 0.38}

# 風險等級以近 N 日報酬的年化波動率計算
RISK_WINDOW_DAYS = 60
TRADING_DAYS_PER_YEAR = 252

# 年化波動率分位門檻（Phase 6 risk.py 使用；台股經驗值，Phase 5 後可依實際資料校準）
RISK_VOL_LOW_MAX = 0.20
RISK_VOL_HIGH_MIN = 0.40

# 股票池最後一次「與指數官方名單核對過」的日期。**不是**檔案最後修改日。
# 台灣 50 每年 3/6/9/12 月第三個星期五收盤後定期審核，所以這個日期一過就開始腐壞；
# python -m stockta.data.pool_audit events 會依此算出錯過了幾次審核。
#
# 2026-08-16 實測：本清單與 2025-12-19、2026-06-18 兩次已查證的審核有 15 處不一致
# （詳 docs/survivorship_study.md）。下方日期記的是「最後一次真正核對」的時點，
# 在補上官方名單之前不得往後改——把它改成今天不會讓池子變新，只會讓警訊消失。
STOCK_POOL_ASOF = "2025-07-01"

# 唯一事實來源：股票池清單。前端 GET /api/stocks 由此提供，不得在前端另外維護清單
# 台灣 50 成分股參考清單（成分股每季調整；**目前已知過期**，見 STOCK_POOL_ASOF）
# 個別股票上市未滿十年者以實際可得資料訓練
STOCK_POOL: dict[str, str] = {
    "1101.TW": "台泥",
    "1216.TW": "統一",
    "1301.TW": "台塑",
    "1303.TW": "南亞",
    "1326.TW": "台化",
    "2002.TW": "中鋼",
    "2207.TW": "和泰車",
    "2301.TW": "光寶科",
    "2303.TW": "聯電",
    "2308.TW": "台達電",
    "2317.TW": "鴻海",
    "2327.TW": "國巨",
    "2330.TW": "台積電",
    "2345.TW": "智邦",
    "2357.TW": "華碩",
    "2379.TW": "瑞昱",
    "2382.TW": "廣達",
    "2395.TW": "研華",
    "2412.TW": "中華電",
    "2454.TW": "聯發科",
    "2603.TW": "長榮",
    "2618.TW": "長榮航",
    "2880.TW": "華南金",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "2883.TW": "凱基金",
    "2884.TW": "玉山金",
    "2885.TW": "元大金",
    "2886.TW": "兆豐金",
    "2887.TW": "台新金",
    "2890.TW": "永豐金",
    "2891.TW": "中信金",
    "2892.TW": "第一金",
    "2912.TW": "統一超",
    "3008.TW": "大立光",
    "3017.TW": "奇鋐",
    "3034.TW": "聯詠",
    "3037.TW": "欣興",
    "3045.TW": "台灣大",
    "3231.TW": "緯創",
    "3661.TW": "世芯-KY",
    "3711.TW": "日月光投控",
    "4904.TW": "遠傳",
    "4938.TW": "和碩",
    "5871.TW": "中租-KY",
    "5876.TW": "上海商銀",
    "5880.TW": "合庫金",
    "6505.TW": "台塑化",
    "6669.TW": "緯穎",
}
