"""每日批次記錄線上相對強弱分數：python -m stockta.ml.record_rank_predictions

對全股票池以 cross-sectional production 模型（CS_PRODUCTION_MODEL）評分
（P(未來 LABEL_HORIZON_DAYS 日贏過當日全池中位數)），落地 predictions.db 的
rank_predictions 表——與絕對方向 record_predictions 並存、互不干擾。累積後由
report_rank_predictions 產出**線上 Rank IC**，補上「絕對方向線上實證只驗證已淘汰
路線」的缺口（相對排序才是 experiment_log #8 的正面主線）。

子命令：
    live      （預設）以最後一個已完成交易日為基準日即時評分並記錄（source=live）
    backfill  以固定權重、point-in-time 重建既有絕對預測相同的歷史基準日
              （source=pit，無前視、與 /api/rank?date= 同一路徑），使線上 Rank IC
              立即有可到期樣本，之後由每日 live 累積延續
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta

from stockta.config import (
    AUTO_ADJUST,
    CS_PRODUCTION_MODEL,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    MARKET_INDEX_TICKER,
    PREDICTIONS_DB_PATH,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.features.market import build_market_context
from stockta.inference.replay import context_fetch_start, load_pool_context
from stockta.inference.store import RankPredictionStore
from stockta.ml.cross_sectional import load_cs_model, score_asof
from stockta.ml.report_rank_predictions import cs_production_version

# context 的 rolling 暖機餘裕（與 record_predictions 一致）
_CONTEXT_EXTRA_DAYS = 60


def run_live(store: RankPredictionStore, model, scaler, version: str) -> int:
    """以最後一個已完成交易日為基準日，即時評分全池並記錄（source=live）。"""
    provider = YFinanceProvider(cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST)
    end = last_completed_trading_day()
    lookback = calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS)
    ctx_start = end - timedelta(days=lookback + _CONTEXT_EXTRA_DAYS)

    pool_ohlcv: dict = {}
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

    n = 0
    for ticker, df in pool_ohlcv.items():
        r = score_asof(model, scaler, df, context, None)
        if r is None:
            continue
        score, base_date = r
        store.record(ticker, base_date, score, version, source="live")
        n += 1
    print(f"live：記錄 {n} 檔，累計 {store.count()} 筆 rank_predictions（{version}）")
    return 0 if n else 1


def run_backfill(store: RankPredictionStore, model, scaler, version: str) -> int:
    """對既有絕對方向預測的歷史基準日，以固定權重 point-in-time 重建 CS 分數
    （source=pit，無前視，與 /api/rank?date= 同一重算路徑）。"""
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    base_dates = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT base_date FROM predictions ORDER BY base_date"
        ).fetchall()
    ]
    conn.close()

    provider = YFinanceProvider(
        cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST, max_cache_age_days=9999
    )
    total = 0
    for bd in base_dates:
        as_of = date.fromisoformat(bd)
        try:
            pool_ohlcv, context = load_pool_context(provider, context_fetch_start(as_of), as_of)
        except DataProviderError as exc:
            print(f"[略過] {bd}: {exc}")
            continue
        n = 0
        for ticker, df in pool_ohlcv.items():
            r = score_asof(model, scaler, df, context, as_of)
            if r is None:
                continue
            score, base_date = r
            store.record(ticker, base_date, score, version, source="pit")
            n += 1
        print(f"  {bd}: 重建 {n} 檔")
        total += n
    print(
        f"backfill：{len(base_dates)} 個基準日、共 {total} 筆（point-in-time），"
        f"累計 {store.count()} 筆 rank_predictions（{version}）"
    )
    return 0 if total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="批次記錄線上相對強弱分數")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("live", help="以最後一個已完成交易日即時評分並記錄（預設）")
    sub.add_parser("backfill", help="point-in-time 重建既有絕對預測相同的歷史基準日")
    args = parser.parse_args()

    model, scaler, _meta = load_cs_model(CS_PRODUCTION_MODEL)
    version = cs_production_version()
    store = RankPredictionStore(PREDICTIONS_DB_PATH)
    if args.cmd == "backfill":
        return run_backfill(store, model, scaler, version)
    return run_live(store, model, scaler, version)


if __name__ == "__main__":
    raise SystemExit(main())
