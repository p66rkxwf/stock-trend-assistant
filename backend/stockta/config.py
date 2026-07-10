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
INDICATOR_WARMUP_DAYS = 60
LABEL_HORIZON_DAYS = 5
LABEL_UP_THRESHOLD = 0.02
LABEL_DOWN_THRESHOLD = -0.02

# 年化波動率分位門檻（Phase 6 risk.py 使用；台股經驗值，Phase 5 後可依實際資料校準）
RISK_VOL_LOW_MAX = 0.20
RISK_VOL_HIGH_MIN = 0.40

# 唯一事實來源：股票池清單。前端 GET /api/stocks 由此提供，不得在前端另外維護清單
STOCK_POOL: dict[str, str] = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2308.TW": "台達電",
    "2382.TW": "廣達",
    "2412.TW": "中華電",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "1301.TW": "台塑",
    "1303.TW": "南亞",
    # TODO(Phase 1): 擴充為完整台灣 50 成分股清單
}
