"""統一訓練入口：python -m stockta.ml.train --model rf|xgb|lstm|gru|tcn

資料一律經 DataProvider 取得（已抓過的股票走 parquet 快取，離線可重跑）。
訓練完成後以 registry.save() 存 artifact + metadata，並印出驗證/測試指標。
深度模型（lstm/gru/tcn）延遲 import torch：無 torch 的環境仍可訓練基線。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import pandas as pd

from stockta.config import (
    AUTO_ADJUST,
    DATA_CACHE_DIR,
    HISTORY_YEARS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    STOCK_POOL,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import last_completed_trading_day
from stockta.data.provider import DataProviderError, YFinanceProvider
from stockta.ml.dataset import build_dataset
from stockta.ml.evaluate import evaluate
from stockta.ml.models.baselines import MODEL_FACTORIES
from stockta.ml import registry

DEEP_MODEL_NAMES = ("lstm", "gru", "tcn")
ALL_MODEL_NAMES = sorted([*MODEL_FACTORIES, *DEEP_MODEL_NAMES])


def create_model(name: str, **hyperparams):
    """建立模型實例；深度模型延遲 import torch。"""
    if name in DEEP_MODEL_NAMES:
        from stockta.ml.models.deep import DEEP_FACTORIES

        return DEEP_FACTORIES[name](**hyperparams)
    return MODEL_FACTORIES[name](**hyperparams)


def load_pool_ohlcv(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    provider = YFinanceProvider(
        cache=ParquetCache(DATA_CACHE_DIR), auto_adjust=AUTO_ADJUST, max_cache_age_days=3.0
    )
    end = last_completed_trading_day()
    start = end - timedelta(days=HISTORY_YEARS * 365)
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers or list(STOCK_POOL):
        try:
            out[ticker] = provider.get_ohlcv(ticker, start, end)
        except DataProviderError as exc:
            print(f"[略過] {ticker}: {exc}")
    if not out:
        raise SystemExit("沒有任何可用資料，請先執行 python -m stockta.data.fetch")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練趨勢分類模型")
    parser.add_argument("--model", required=True, choices=ALL_MODEL_NAMES)
    parser.add_argument("--stride", type=int, default=2, help="視窗取樣間隔（交易日）")
    parser.add_argument("--tickers", nargs="*", default=None, help="只用指定股票（煙霧測試用）")
    args = parser.parse_args()

    print(f"載入 {len(args.tickers or STOCK_POOL)} 檔股票資料…")
    ohlcv = load_pool_ohlcv(args.tickers)

    print(f"建立資料集（stride={args.stride}）…")
    ds = build_dataset(ohlcv, stride=args.stride)
    print(
        f"train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}"
        f"（切分：訓練 ≤{SPLIT_TRAIN_END}、驗證 ≤{SPLIT_VAL_END}、其後測試）"
    )

    model = create_model(args.model)
    print(f"訓練 {args.model}…")
    t0 = time.perf_counter()
    model.fit(ds.X_train, ds.y_train, X_val=ds.X_val, y_val=ds.y_val)
    train_seconds = time.perf_counter() - t0
    print(f"訓練完成，耗時 {train_seconds:.1f}s")

    metrics = {
        "val": evaluate(model, ds.X_val, ds.y_val),
        "test": evaluate(model, ds.X_test, ds.y_test),
        "train_seconds": round(train_seconds, 1),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    extra = {"stride": args.stride, "n_train": len(ds.y_train), "tickers": sorted(ohlcv)}
    if hasattr(model, "hyperparams"):
        extra["hyperparams"] = model.hyperparams()
    out_dir = registry.save(args.model, model, ds.scaler, metrics, extra=extra)
    print(f"artifact 已存入 {out_dir}")


if __name__ == "__main__":
    main()
