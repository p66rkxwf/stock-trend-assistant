"""滑動視窗切割、時間序列切分與標準化。

切分原則（PLAN.md）：依日期純時間切割，絕不隨機打亂；StandardScaler 只 fit
訓練期的列。訓練/驗證樣本若其標籤視野（t+horizon）跨入下一個切分期，一律剔除
（embargo），避免切分邊界的前視洩漏。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from stockta.config import (
    LABEL_HORIZON_DAYS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    WINDOW_LENGTH_DAYS,
)
from stockta.features.pipeline import build_features
from stockta.ml.labeling import make_labels


@dataclass
class Dataset:
    """攤平視窗後的三組資料；X 形狀 (n, window * n_features)，float32。"""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    # 測試樣本的基準日（視窗終點）；walk-forward 實驗要把同一個模型的測試期
    # 再切成數個時間窗分別評估，沒有日期就辦不到。順序與 X_test 對齊。
    dates_test: np.ndarray | None = None


def build_dataset(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    window: int = WINDOW_LENGTH_DAYS,
    stride: int = 1,
    train_end: str | pd.Timestamp | None = None,
    val_end: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
) -> Dataset:
    """由多檔股票的 OHLCV 與市場情境 context 建出訓練/驗證/測試集。

    context 由 features.market.build_market_context() 產生（大盤 + 寬度），
    全部股票共用同一份。stride > 1 時每隔 stride 個交易日取一個樣本
    （相鄰視窗重疊 59/60，子取樣可大幅降低訓練時間而幾乎不損失資訊量）。

    train_end/val_end 預設取 config 的正式切分；walk-forward 實驗（#9）需要逐折
    改變切分點，故開放覆寫。test_end 可再把測試期截在某日之前（折與折之間不重疊）。
    """
    train_end = pd.Timestamp(train_end or SPLIT_TRAIN_END)
    val_end = pd.Timestamp(val_end or SPLIT_VAL_END)
    test_end_ts = pd.Timestamp(test_end) if test_end else None

    feats: list[tuple[pd.DataFrame, pd.Series]] = []
    for ohlcv in ohlcv_by_ticker.values():
        f = build_features(ohlcv, context)
        labels = make_labels(ohlcv["close"]).reindex(f.index)
        feats.append((f, labels))

    feat_cols = feats[0][0].columns
    # scaler 只 fit 訓練期的列，驗證/測試期的分佈對 scaler 不可見
    train_rows = pd.concat([f.loc[f.index <= train_end] for f, _ in feats])
    scaler = StandardScaler().fit(train_rows.to_numpy(dtype=np.float64))

    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte", "dte")
    }
    for f, labels in feats:
        if len(f) < window:
            continue
        scaled = scaler.transform(f.to_numpy(dtype=np.float64)).astype(np.float32)
        # windows[i] 是以第 i+window-1 列為終點、往回看 window 天的攤平視窗
        windows = sliding_window_view(scaled, window, axis=0)
        windows = windows.transpose(0, 2, 1).reshape(windows.shape[0], -1)

        end_dates = f.index[window - 1 :]
        y = labels.iloc[window - 1 :].to_numpy(dtype=np.float64)
        valid = ~np.isnan(y)

        # 標籤日（t+horizon 的實際交易日）；超出資料末端者其標籤本來就是 NaN
        pos = np.arange(window - 1, len(f))
        label_pos = pos + LABEL_HORIZON_DAYS
        label_dates = np.full(len(pos), np.datetime64(pd.Timestamp.max, "ns"))
        in_range = label_pos < len(f)
        label_dates[in_range] = f.index.values[label_pos[in_range]]

        stride_mask = np.zeros(len(pos), dtype=bool)
        stride_mask[::stride] = True

        is_train = (end_dates <= train_end) & (label_dates <= np.datetime64(train_end, "ns"))
        is_val = (
            (end_dates > train_end)
            & (end_dates <= val_end)
            & (label_dates <= np.datetime64(val_end, "ns"))
        )
        is_test = end_dates > val_end
        if test_end_ts is not None:
            # 測試期同樣套 embargo：標籤日超出測試期末端者剔除，折與折之間不重疊
            is_test = (
                is_test
                & (end_dates <= test_end_ts)
                & (label_dates <= np.datetime64(test_end_ts, "ns"))
            )

        for x_key, y_key, mask in (
            ("Xtr", "ytr", np.asarray(is_train)),
            ("Xva", "yva", np.asarray(is_val)),
            ("Xte", "yte", np.asarray(is_test)),
        ):
            m = mask & valid & stride_mask
            if m.any():
                parts[x_key].append(windows[m])
                parts[y_key].append(y[m])
                if x_key == "Xte":
                    parts["dte"].append(np.asarray(end_dates)[m])

    def stack(xs: list[np.ndarray], ys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not xs:
            return np.empty((0, window * len(feat_cols)), dtype=np.float32), np.empty((0,))
        return np.concatenate(xs), np.concatenate(ys).astype(np.int64)

    X_train, y_train = stack(parts["Xtr"], parts["ytr"])
    X_val, y_val = stack(parts["Xva"], parts["yva"])
    X_test, y_test = stack(parts["Xte"], parts["yte"])
    dates_test = (
        np.concatenate(parts["dte"]) if parts["dte"] else np.empty((0,), dtype="datetime64[ns]")
    )
    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test, scaler, dates_test)
