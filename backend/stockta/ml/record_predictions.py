"""每日批次記錄線上預測：python -m stockta.ml.record_predictions

對股票池全部（或 --tickers 指定的）標的抓最新股價、以 production 模型推論，
並落地 predictions.db——與 API `/prediction` 同一條路徑與去重規則
（同 ticker+基準日+版本只記第一筆），不需啟動後端伺服器。

每日（收盤後）跑一次即可累積「線上預測 vs 實際走勢」實證資料，
之後以 python -m stockta.ml.report_predictions 產出報告。
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    PREDICTIONS_DB_PATH,
    PRODUCTION_MODEL,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.inference.predictor import InsufficientDataError, Predictor
from stockta.inference.store import PredictionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="批次記錄線上預測")
    parser.add_argument(
        "--tickers", nargs="+", default=list(STOCK_POOL), help="預設為股票池全部標的"
    )
    args = parser.parse_args()

    provider = YFinanceProvider(cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST)
    predictor = Predictor.from_registry(PRODUCTION_MODEL)
    store = PredictionStore(PREDICTIONS_DB_PATH)
    print(f"模型：{predictor.version}；標的 {len(args.tickers)} 檔")

    end = last_completed_trading_day()
    start = end - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS))

    n_ok = 0
    for ticker in args.tickers:
        try:
            df = provider.get_ohlcv(ticker, start, end)
            pred = predictor.predict(df)
        except (DataProviderError, InsufficientDataError) as exc:
            print(f"[略過] {ticker}: {exc}")
            continue
        store.record(ticker, pred.base_date, pred.signal, pred.confidence, predictor.version)
        print(f"  {ticker} {STOCK_POOL.get(ticker, '')}: {pred.base_date} {pred.signal} ({pred.confidence:.2f})")
        n_ok += 1

    print(f"完成 {n_ok}/{len(args.tickers)} 檔，累計 {store.count()} 筆（predictions.db）")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
