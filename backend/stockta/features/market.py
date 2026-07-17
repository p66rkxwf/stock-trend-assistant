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

from stockta.config import BREADTH_MIN_TICKERS

# context 欄位的唯一順序；pipeline.FEATURE_COLUMNS 引用此清單
MARKET_CONTEXT_COLUMNS = [
    "mkt_ret_1d",
    "mkt_ret_5d",
    "mkt_ma20",
    "mkt_vol20",
    "breadth_up",
    "breadth_ma5",
]


def build_market_context(
    market_ohlcv: pd.DataFrame,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    min_tickers: int = BREADTH_MIN_TICKERS,
) -> pd.DataFrame:
    """由大盤 OHLCV 與股票池 OHLCV 建出日期索引的市場情境特徵。

    暖機期不足的前段列（rolling 尚無值）會被丟棄。
    """
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

    return out[MARKET_CONTEXT_COLUMNS].dropna()
