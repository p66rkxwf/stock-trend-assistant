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
import sqlite3
from datetime import timedelta

from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    LABEL_HORIZON_DAYS,
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


def _recorded_pairs(db_path, version: str) -> set[tuple[str, str]]:
    """該版本已記錄過的 (ticker, 基準日) 集合（供冪等自癒：跳過已補上的、省去重算）。

    以「標的×日」而非「日」為單位：雲端排程抓資料時個別標的失敗很常見（Yahoo 對
    資料中心 IP 限流），若以日為單位，當天只要記到一檔就算「已記錄」，失敗的那幾檔
    之後永遠不會被補上。
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ticker, base_date FROM predictions WHERE model_version=?", (version,)
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {(r[0], r[1]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="批次記錄線上預測")
    parser.add_argument(
        "--tickers", nargs="+", default=list(STOCK_POOL), help="預設為股票池全部標的"
    )
    parser.add_argument(
        "--days", type=int, default=LABEL_HORIZON_DAYS,
        help="回補最近 N 個交易日（冪等自癒；預設=標籤天數）。資料商當日 K 線未及時到位時，"
        "當次會記到較舊日、隔日跑再補回；視窗 ≤ 標籤天數確保回補日的 N 日結果尚未實現，"
        "故無前視、fill 規則與結果無關亦無選擇偏誤。",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="有標的缺當日資料時以 exit 2 結束（雲端排程據此觸發補跑）；已取得的照常記錄",
    )
    args = parser.parse_args()

    provider = YFinanceProvider(cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST)
    predictor = Predictor.from_registry(PRODUCTION_MODEL)
    store = PredictionStore(PREDICTIONS_DB_PATH)
    print(f"模型：{predictor.version}；標的 {len(args.tickers)} 檔；回補視窗 {args.days} 交易日")

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

    # 最近 N 個交易日（以市場情境索引為交易日曆），跳過已記錄者→自癒漏記、避免每次重算
    recent_days = list(context.index[-args.days:])
    already = _recorded_pairs(PREDICTIONS_DB_PATH, predictor.version)
    covered: list[str] = []
    missing: list[str] = []
    n_records = 0
    for as_of in recent_days:
        day = as_of.date().isoformat()
        day_ok = 0
        for ticker in args.tickers:
            if (ticker, day) in already:
                continue
            df = pool_ohlcv.get(ticker)
            if df is None:
                missing.append(f"{ticker}@{day}")
                continue
            sub = df.loc[(df.index >= str(start)) & (df.index <= as_of)]
            # 當日 K 棒不在（下載失敗退回舊快取）→ 不記，留給下次補；否則會記成較舊的基準日
            if sub.empty or sub.index[-1] != as_of:
                missing.append(f"{ticker}@{day}")
                continue
            try:
                pred = predictor.predict(sub, context)
            except InsufficientDataError:
                continue
            store.record(ticker, pred.base_date, pred.signal, pred.confidence, predictor.version)
            day_ok += 1
        if day_ok:
            covered.append(as_of.date().isoformat())
            n_records += day_ok
            print(f"  {as_of.date()}: 記錄 {day_ok} 檔")

    if covered:
        print(f"完成：新記錄基準日 {'、'.join(covered)}（{n_records} 筆），累計 {store.count()} 筆（predictions.db）")
    else:
        print(f"完成：最近 {args.days} 交易日皆已記錄、無新增；累計 {store.count()} 筆（predictions.db）")
    if missing:
        print(f"[未完成] {len(missing)} 筆缺當日資料、待下次補：{'、'.join(missing[:10])}")
        return 2 if args.require_complete else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
