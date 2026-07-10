"""特徵管線 —— build_features() 是訓練與推論唯一共用進入點。

技術指標以 pandas 自行實作（pandas-ta 與 numpy>=2 不相容，見 PLAN.md Phase 0）。
所有特徵皆為因果計算（只用當日與更早的資料），並轉為相對值避免價格尺度問題；
test_no_lookahead.py 與 test_feature_parity.py 對這兩點把關。

注意：EMA 類指標（RSI/MACD/KD）受序列起點影響，推論端必須依
config.INDICATOR_WARMUP_DAYS 提供足夠暖機資料，特徵值才與訓練期一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 特徵欄位的唯一順序 —— artifact metadata 記錄此清單，API 啟動時比對，不一致拒絕載入
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "close_ma5",
    "close_ma10",
    "close_ma20",
    "close_ma60",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pctb",
    "bb_width",
    "kd_k",
    "kd_d",
    "vol_ratio",
    "vol_chg",
    "hl_range",
]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """輸入 open/high/low/close/volume（日期索引遞增），輸出 FEATURE_COLUMNS 特徵。

    暖機期不足的前段列（rolling/EMA 尚無值）會被丟棄，回傳列數少於輸入列數。
    """
    close = ohlcv["close"].astype("float64")
    high = ohlcv["high"].astype("float64")
    low = ohlcv["low"].astype("float64")
    volume = ohlcv["volume"].astype("float64")

    out = pd.DataFrame(index=ohlcv.index)

    out["ret_1d"] = close.pct_change()
    out["ret_5d"] = close.pct_change(5)

    for n in (5, 10, 20, 60):
        out[f"close_ma{n}"] = close / close.rolling(n).mean() - 1.0

    out["rsi14"] = _rsi(close, 14)

    # MACD 先除以收盤價轉為尺度無關，再取訊號線與柱狀圖
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    band_range = 4.0 * std20  # 上下軌相距 4σ
    # 平盤期間 σ=0，%B 無定義，取中性值 0.5
    out["bb_pctb"] = ((close - (ma20 - 2.0 * std20)) / band_range).where(band_range > 0, 0.5)
    out["bb_width"] = band_range / ma20

    k, d = _stochastic_kd(close, high, low, 9)
    out["kd_k"] = k
    out["kd_d"] = d

    vol_ma20 = volume.rolling(20).mean()
    out["vol_ratio"] = (volume / vol_ma20 - 1.0).where(vol_ma20 > 0, 0.0)
    out["vol_chg"] = volume.pct_change().replace([np.inf, -np.inf], 0.0)
    out["hl_range"] = (high - low) / close

    return out[FEATURE_COLUMNS].dropna()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI，輸出縮放為 0–1（而非 0–100），與其他特徵尺度一致。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    total = gain + loss
    # 全平盤（漲跌皆 0）時 RSI 無定義，取中性值 0.5
    rsi = (gain / total).where(total > 0, 0.5)
    rsi.iloc[: period] = np.nan  # 前 period 日視為暖機，不輸出
    return rsi


def _stochastic_kd(
    close: pd.Series, high: pd.Series, low: pd.Series, period: int
) -> tuple[pd.Series, pd.Series]:
    """KD 隨機指標（台股慣例：K/D 以 1/3 平滑），輸出縮放為 0–1。"""
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    span = highest - lowest
    rsv = ((close - lowest) / span).where(span > 0, 0.5)
    k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    return k, d
