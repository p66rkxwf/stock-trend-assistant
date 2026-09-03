"""市場情境特徵：大盤指數（^TWII）與市場寬度（股票池上漲家數比）。

build_market_context() 產出的 context DataFrame 是 build_features() 的必要輸入，
訓練與推論共用同一條路徑。所有欄位皆為因果計算（只用當日與更早的資料），
與個股特徵同受 test_no_lookahead.py 把關。

寬度（breadth）的分母是「當日有資料的成分股數」——個別股票停牌或上市較晚不會
汙染比值；但可用檔數低於 config.BREADTH_MIN_TICKERS 時該日視為不可信（NaN，
隨下游 dropna 丟棄），避免池子資料殘缺時算出誤導性的寬度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockta.config import BREADTH_MIN_TICKERS, MARKET_INCLUDE_REGIME

# 短期情境欄位（2026-07-18 起既有）
MARKET_CONTEXT_BASE_COLUMNS = [
    "mkt_ret_1d",
    "mkt_ret_5d",
    "mkt_ma20",
    "mkt_vol20",
    "breadth_up",
    "breadth_ma5",
]

# 市場狀態（regime）欄位（實驗 #9）：全部只用 ^TWII 自身、皆為因果計算。
# 動機：模型訓練資料到某個時點就凍結，之後市場狀態改變（多頭↔空頭、低波↔高波）
# 時預測會系統性偏一邊——2026 多頭段「喊跌」只有 23.5% 命中即是此症狀。
# 給模型看得見的狀態變數，讓它有機會條件化，而不是把所有年份混成一個平均。
MARKET_REGIME_COLUMNS = [
    "mkt_ma60",       # 中期趨勢：收盤相對 60 日均線
    "mkt_ma200",      # 多空分界：收盤相對 200 日均線（>0 概念上為多頭）
    "mkt_drawdown",   # 距一年高點的回撤深度（≤0）
    "mkt_vol_pct",    # 20 日波動在過去一年中的百分位（0~1，低波/高波狀態）
]

# 所有可能的 context 欄位（含尚未採用者）——特徵順序的唯一宣告來源
ALL_CONTEXT_COLUMNS = [*MARKET_CONTEXT_BASE_COLUMNS, *MARKET_REGIME_COLUMNS]

# **正式管線實際使用**的 context 欄位；pipeline.FEATURE_COLUMNS 引用此清單。
# 由 config.MARKET_INCLUDE_REGIME 決定——實驗 #9 未給出採用結論前維持 6 欄，
# 否則既有 artifact（25 欄）會因特徵契約不符而拒絕載入，線上直接壞掉。
MARKET_CONTEXT_COLUMNS = [
    *MARKET_CONTEXT_BASE_COLUMNS,
    *(MARKET_REGIME_COLUMNS if MARKET_INCLUDE_REGIME else []),
]

# regime 欄位所需的最長回溯（交易日）——推論端必須提供至少這麼長的大盤資料，
# 否則 rolling 尚無值、整段 context 會被 dropna 清空。inference/market_context.py 依此加長回溯。
MARKET_REGIME_WINDOW_DAYS = 200

# 每個 context 欄位的「最早可用時間」。新增欄位而忘了標註，
# test_feature_availability.py 會紅燈——這是刻意的。
# 定義與檢查見 features/availability.py。
CONTEXT_AVAILABILITY = {
    # 只需要 ^TWII 自己的日 K
    "mkt_ret_1d": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_ret_5d": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_ma20": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_vol20": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_ma60": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_ma200": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_drawdown": (0, "^TWII 的 t 日收盤 K 線到位"),
    "mkt_vol_pct": (0, "^TWII 的 t 日收盤 K 線到位"),
    # 寬度要等**全池**——實務上這是當日最晚到位的特徵，決策時點由它決定
    "breadth_up": (0, f"全池至少 {BREADTH_MIN_TICKERS} 檔的 t 日收盤 K 線到位"),
    "breadth_ma5": (0, f"全池至少 {BREADTH_MIN_TICKERS} 檔的 t 日收盤 K 線到位"),
}


def build_market_context(
    market_ohlcv: pd.DataFrame,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    min_tickers: int = BREADTH_MIN_TICKERS,
    include_regime: bool | None = None,
) -> pd.DataFrame:
    """由大盤 OHLCV 與股票池 OHLCV 建出日期索引的市場情境特徵。

    暖機期不足的前段列（rolling 尚無值）會被丟棄。include_regime 預設跟隨
    config.MARKET_INCLUDE_REGIME；顯式傳 True/False 可強制切換，供實驗 #9
    做「有無 regime 欄位」的成對消融。
    """
    if include_regime is None:
        include_regime = MARKET_INCLUDE_REGIME
    close = market_ohlcv["close"].astype("float64")

    out = pd.DataFrame(index=market_ohlcv.index)
    out["mkt_ret_1d"] = close.pct_change()
    out["mkt_ret_5d"] = close.pct_change(5)
    out["mkt_ma20"] = close / close.rolling(20).mean() - 1.0
    out["mkt_vol20"] = out["mkt_ret_1d"].rolling(20).std()

    # 各股日報酬並排成寬表（聯集索引；無資料處為 NaN，不參與當日分母）
    rets = pd.DataFrame(
        {t: df["close"].astype("float64").pct_change() for t, df in ohlcv_by_ticker.items()}
    )
    available = rets.notna().sum(axis=1)
    breadth = (rets > 0).sum(axis=1) / available.replace(0, np.nan)
    breadth[available < min_tickers] = np.nan
    out["breadth_up"] = breadth.reindex(out.index)
    out["breadth_ma5"] = out["breadth_up"].rolling(5).mean()

    columns = list(MARKET_CONTEXT_BASE_COLUMNS)
    if include_regime:
        w = MARKET_REGIME_WINDOW_DAYS
        out["mkt_ma60"] = close / close.rolling(60).mean() - 1.0
        out["mkt_ma200"] = close / close.rolling(w).mean() - 1.0
        out["mkt_drawdown"] = close / close.rolling(252, min_periods=w).max() - 1.0
        # 百分位用 rank(pct=True) 對「過去 252 日（含當日）」計算，不看未來
        out["mkt_vol_pct"] = (
            out["mkt_vol20"]
            .rolling(252, min_periods=w)
            .apply(lambda s: float((s <= s[-1]).mean()), raw=True)
        )
        columns += MARKET_REGIME_COLUMNS

    return out[columns].dropna()
