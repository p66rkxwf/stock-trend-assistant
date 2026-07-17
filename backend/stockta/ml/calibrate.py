"""信心門檻校準：python -m stockta.ml.calibrate

對驗證期（SPLIT_TRAIN_END ~ SPLIT_VAL_END）收集 production 模型的逐日機率，
以整體命中率網格搜尋「跌/漲」信心門檻，印出建議值。

門檻只准用驗證期挑選（測試期留給 backtest 驗證，避免樂觀偏差）。
重新訓練或更換 PRODUCTION_MODEL 後執行本指令，把結果回填
config.SIGNAL_CONFIDENCE_THRESHOLDS 並重跑 python -m stockta.ml.backtest。
"""

from __future__ import annotations

from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd

from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    LABEL_CLASSES,
    MARKET_INDEX_TICKER,
    PRODUCTION_MODEL,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.inference.predictor import Predictor
from stockta.ml.backtest import collect_probas

HOLD = 1  # LABEL_CLASSES.index("觀望")


def main() -> int:
    provider = YFinanceProvider(cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST)
    predictor = Predictor.from_registry(PRODUCTION_MODEL)
    train_end = pd.Timestamp(SPLIT_TRAIN_END)
    val_end = np.datetime64(pd.Timestamp(SPLIT_VAL_END))
    fetch_start = train_end.date() - timedelta(days=730)
    end = last_completed_trading_day()
    print(f"模型：{predictor.version}；驗證期 {SPLIT_TRAIN_END} ~ {SPLIT_VAL_END}")

    pool_ohlcv = {}
    for ticker in STOCK_POOL:
        try:
            pool_ohlcv[ticker] = provider.get_ohlcv(ticker, fetch_start, end)
        except DataProviderError as exc:
            print(f"[略過] {ticker}: {exc}")
    market = provider.get_ohlcv(MARKET_INDEX_TICKER, fetch_start, end)
    context = build_market_context(market, pool_ohlcv)

    probas, ys = [], []
    for df in pool_ohlcv.values():
        p, a, d = collect_probas(predictor, df, context, train_end)
        m = d <= val_end
        if m.any():
            probas.append(p[m])
            ys.append(a[m])
    proba = np.concatenate(probas)
    y = np.concatenate(ys)
    print(f"驗證期樣本 {len(y):,} 筆")

    grid = np.arange(0.34, 0.72, 0.01)
    best = (-1.0, 0.0, 0.0)
    for td, tu in product(grid, grid):
        pred = proba.argmax(axis=1)
        pred[(pred == 0) & (proba[:, 0] < td)] = HOLD
        pred[(pred == 2) & (proba[:, 2] < tu)] = HOLD
        acc = float((pred == y).mean())
        if acc > best[0]:
            best = (acc, td, tu)

    acc, td, tu = best
    raw_acc = float((proba.argmax(axis=1) == y).mean())
    print(f"原始 argmax 驗證期命中率 {raw_acc:.1%} → 校準後 {acc:.1%}")
    print(f'建議回填 config.py：SIGNAL_CONFIDENCE_THRESHOLDS = {{"跌": {td:.2f}, "漲": {tu:.2f}}}')
    print("回填後請重跑 python -m stockta.ml.backtest 以測試期驗證。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
