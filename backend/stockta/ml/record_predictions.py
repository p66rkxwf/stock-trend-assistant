"""每日批次記錄線上預測：python -m stockta.ml.record_predictions

先抓全股票池與大盤（^TWII）資料、建市場情境 context，再對全部（或 --tickers
指定的）標的以 production 模型推論並落地 predictions.db——與 API `/prediction`
同一條路徑與去重規則（同 ticker+基準日+版本只記第一筆），不需啟動後端伺服器。
「先抓全池」同時保證了市場寬度（breadth）的資料時效。

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
    MARKET_INDEX_TICKER,
    PREDICTIONS_DB_PATH,
    PRODUCTION_MODEL,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.inference.predictor import InsufficientDataError, Predictor
from stockta.inference.store import PredictionStore

# context 的 rolling 暖機（20+5 日）餘裕
_CONTEXT_EXTRA_DAYS = 60


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
    lookback = calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS)
    start = end - timedelta(days=lookback)
    ctx_start = end - timedelta(days=lookback + _CONTEXT_EXTRA_DAYS)

    # 先抓全池（含未指定 --tickers 的成分股）——市場寬度需要全池資料
    pool_ohlcv = {}
    for ticker in STOCK_POOL:
        try:
            pool_ohlcv[ticker] = provider.get_ohlcv(ticker, ctx_start, end)
        except DataProviderError as exc:
            print(f"[略過] {ticker}: {exc}")

    try:
        market = provider.get_ohlcv(MARKET_INDEX_TICKER, ctx_start, end)
    except DataProviderError as exc:
        print(f"無法取得大盤指數 {MARKET_INDEX_TICKER}: {exc}")
        return 1
    context = build_market_context(market, pool_ohlcv)

    n_ok = 0
    for ticker in args.tickers:
        df = pool_ohlcv.get(ticker)
        if df is None:
            continue
        try:
            pred = predictor.predict(df.loc[df.index >= str(start)], context)
        except InsufficientDataError as exc:
            print(f"[略過] {ticker}: {exc}")
            continue
        store.record(ticker, pred.base_date, pred.signal, pred.confidence, predictor.version)
        print(f"  {ticker} {STOCK_POOL.get(ticker, '')}: {pred.base_date} {pred.signal} ({pred.confidence:.2f})")
        n_ok += 1

    print(f"完成 {n_ok}/{len(args.tickers)} 檔，累計 {store.count()} 筆（predictions.db）")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
