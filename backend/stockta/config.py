"""唯一設定來源：股票池、視窗長度、標籤門檻、風險分位、auto_adjust 等。

不要在 features/、api/、frontend 各自維護一份設定副本——一律從這裡讀取。
"""

from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = BACKEND_ROOT / "data_cache"
ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
PREDICTIONS_DB_PATH = BACKEND_ROOT / "predictions.db"

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
MARKET_CLOSE_HOUR = 13
MARKET_CLOSE_MINUTE = 30

# yfinance 調整價設定：訓練與推論必須一致，否則除權息日前後特徵與標籤全部失真
AUTO_ADJUST = True

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
SPLIT_TRAIN_END = "2022-12-31"
SPLIT_VAL_END = "2023-12-31"

# 訓練資料回溯年數
HISTORY_YEARS = 10

# API 載入的正式模型名稱（artifacts/<名稱>/），訓練比較後由 compare.py 結果決定
# 2026-07-10 比較：rf 測試 Macro AUC 0.6496 > xgb 0.6303（見 docs/model_comparison.md）
PRODUCTION_MODEL = "rf"

# 風險等級以近 N 日報酬的年化波動率計算
RISK_WINDOW_DAYS = 60
TRADING_DAYS_PER_YEAR = 252

# 年化波動率分位門檻（Phase 6 risk.py 使用；台股經驗值，Phase 5 後可依實際資料校準）
RISK_VOL_LOW_MAX = 0.20
RISK_VOL_HIGH_MIN = 0.40

# 唯一事實來源：股票池清單。前端 GET /api/stocks 由此提供，不得在前端另外維護清單
# 台灣 50 成分股參考清單（2026-07 快照；成分股每季調整，變動時更新此處即可，
# 個別股票上市未滿十年者以實際可得資料訓練）
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
