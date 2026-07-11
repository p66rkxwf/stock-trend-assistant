"""深度模型超參數搜尋：python -m stockta.ml.tune --model lstm|gru|tcn

小型 grid search（PLAN.md Phase 5）：以驗證集 macro AUC 選優，
最佳組合的模型連同超參數與指標經 registry.save 存回 artifacts/<model>/
（覆蓋 train.py 的預設超參數版本）。資料與切分沿用 train.py 的同一條路徑，
所有組合都在同一份資料上比較。基線模型（rf/xgb）不在此調參範圍。
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

from stockta.ml.dataset import build_dataset
from stockta.ml.evaluate import evaluate
from stockta.ml.train import DEEP_MODEL_NAMES, create_model, load_pool_ohlcv
from stockta.ml import registry

# 小而有代表性的網格：容量（hidden/layers）、學習率、正則化（dropout）各兩三檔
GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "lr": [1e-3, 3e-4],
    "dropout": [0.2, 0.4],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="深度模型超參數搜尋")
    parser.add_argument("--model", required=True, choices=DEEP_MODEL_NAMES)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--tickers", nargs="*", default=None, help="只用指定股票（煙霧測試用）")
    args = parser.parse_args()

    ohlcv = load_pool_ohlcv(args.tickers)
    ds = build_dataset(ohlcv, stride=args.stride)
    print(f"train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}")

    keys = list(GRID)
    results: list[dict] = []
    best = None
    for combo in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        t0 = time.perf_counter()
        model = create_model(args.model, **params)
        model.fit(ds.X_train, ds.y_train, X_val=ds.X_val, y_val=ds.y_val)
        val = evaluate(model, ds.X_val, ds.y_val)
        seconds = time.perf_counter() - t0
        row = {**params, "val_macro_auc": val["macro_auc"], "epochs": model.epochs_run_,
               "seconds": round(seconds, 1)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best is None or val["macro_auc"] > best["val"]["macro_auc"]:
            best = {"params": params, "model": model, "val": val, "seconds": seconds}

    assert best is not None
    print(f"\n最佳組合（驗證 macro AUC {best['val']['macro_auc']:.4f}）: {best['params']}")

    metrics = {
        "val": best["val"],
        "test": evaluate(best["model"], ds.X_test, ds.y_test),
        "train_seconds": round(best["seconds"], 1),
    }
    print(json.dumps(metrics["test"], ensure_ascii=False, indent=2))

    extra = {
        "stride": args.stride,
        "n_train": len(ds.y_train),
        "tickers": sorted(ohlcv),
        "hyperparams": best["model"].hyperparams(),
        "tuned": True,
        "tuning_grid": GRID,
        "tuning_results": results,
    }
    out_dir = registry.save(args.model, best["model"], ds.scaler, metrics, extra=extra)
    print(f"最佳模型已存入 {out_dir}")


if __name__ == "__main__":
    main()
