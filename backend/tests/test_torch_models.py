"""深度模型（LSTM/GRU/TCN）單元測試：介面形狀、攤平佈局還原、early stopping。

以小型合成資料快速驗證（CPU 數秒內跑完），不做真實訓練。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stockta.config import WINDOW_LENGTH_DAYS  # noqa: E402
from stockta.features.pipeline import build_features  # noqa: E402
from stockta.ml.models.deep import DEEP_FACTORIES  # noqa: E402

N_FEATURES = 17
N_FLAT = WINDOW_LENGTH_DAYS * N_FEATURES


def _toy_data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FLAT)).astype(np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    return X, y


@pytest.mark.parametrize("name", sorted(DEEP_FACTORIES))
def test_fit_predict_shapes_and_probability(name):
    X, y = _toy_data()
    model = DEEP_FACTORIES[name](hidden_size=8, max_epochs=2, batch_size=64)
    model.fit(X, y)

    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()

    pred = model.predict(X[:10])
    assert pred.shape == (10,)
    assert set(np.unique(pred)) <= {0, 1, 2}


def test_flattened_layout_roundtrip(random_walk_ohlcv):
    """dataset.py 的攤平佈局必須能 reshape 回 (n, window, n_features) 且逐值等於原特徵。

    深度模型內部依賴此還原；若 dataset.py 改動攤平方式，此測試會先失敗。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    feats = build_features(random_walk_ohlcv)
    arr = feats.to_numpy(dtype=np.float32)
    windows = sliding_window_view(arr, WINDOW_LENGTH_DAYS, axis=0)
    flat = windows.transpose(0, 2, 1).reshape(windows.shape[0], -1)  # dataset.py 的攤平

    restored = flat.reshape(-1, WINDOW_LENGTH_DAYS, arr.shape[1])
    # 第 i 個視窗以第 i+window-1 列為終點：restored[i, t] 應等於 arr[i + t]
    np.testing.assert_array_equal(restored[0], arr[:WINDOW_LENGTH_DAYS])
    np.testing.assert_array_equal(restored[5], arr[5 : 5 + WINDOW_LENGTH_DAYS])


def test_early_stopping_uses_validation():
    """有驗證集時應在 patience 次無改善後停止（不跑滿 max_epochs）。"""
    X, y = _toy_data(n=200)
    X_val, y_val = _toy_data(n=60, seed=1)

    model = DEEP_FACTORIES["gru"](hidden_size=4, max_epochs=50, patience=2, batch_size=64)
    model.fit(X, y, X_val=X_val, y_val=y_val)
    # 隨機標籤上驗證 loss 很快停止改善；patience=2 應遠早於 50 epochs 停止
    assert model.epochs_run_ < 50

    # 無驗證集：跑滿 max_epochs（煙霧測試路徑）
    model_no_val = DEEP_FACTORIES["gru"](hidden_size=4, max_epochs=3, batch_size=64)
    model_no_val.fit(X, y)
    assert model_no_val.epochs_run_ == 3


def test_joblib_roundtrip(tmp_path):
    """joblib 序列化後載回，預測結果逐值一致（API 載入路徑的前提）。"""
    import joblib

    X, y = _toy_data()
    model = DEEP_FACTORIES["lstm"](hidden_size=8, max_epochs=2, batch_size=64)
    model.fit(X, y)
    before = model.predict_proba(X[:5])

    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    after = loaded.predict_proba(X[:5])
    np.testing.assert_allclose(before, after, atol=1e-7)
