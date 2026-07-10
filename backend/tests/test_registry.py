"""artifact 版本契約：metadata 與程式碼不一致時必須拒絕載入。"""

import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from stockta.features.pipeline import FEATURE_COLUMNS
from stockta.ml import registry
from stockta.ml.models.baselines import RandomForestModel


@pytest.fixture
def saved_artifact(tmp_path):
    rng = np.random.default_rng(0)
    n_dim = 60 * len(FEATURE_COLUMNS)
    X = rng.normal(size=(90, n_dim)).astype(np.float32)
    y = np.array([0, 1, 2] * 30)
    model = RandomForestModel(n_estimators=5, min_samples_leaf=1).fit(X, y)
    scaler = StandardScaler().fit(rng.normal(size=(90, len(FEATURE_COLUMNS))))
    registry.save("rf", model, scaler, metrics={"test": {"macro_auc": 0.5}}, artifacts_dir=tmp_path)
    return tmp_path


def test_roundtrip_load(saved_artifact):
    model, scaler, metadata = registry.load("rf", artifacts_dir=saved_artifact)
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    proba = model.predict_proba(np.zeros((1, 60 * len(FEATURE_COLUMNS)), dtype=np.float32))
    assert proba.shape == (1, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_feature_mismatch_refuses_to_load(saved_artifact):
    meta_path = saved_artifact / "rf" / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["feature_columns"] = list(reversed(metadata["feature_columns"]))
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(registry.ArtifactContractError):
        registry.load("rf", artifacts_dir=saved_artifact)


def test_window_mismatch_refuses_to_load(saved_artifact):
    meta_path = saved_artifact / "rf" / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["window_length_days"] = 999
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(registry.ArtifactContractError):
        registry.load("rf", artifacts_dir=saved_artifact)


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent", artifacts_dir=tmp_path)
