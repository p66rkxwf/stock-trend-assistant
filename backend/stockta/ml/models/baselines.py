"""RandomForest 與 XGBoost 基線（PLAN.md Phase 3）。

兩者皆以攤平的 60 日視窗特徵訓練，class weight 處理「觀望」過半的類別不平衡。
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from stockta.ml.models.base import TrendModel, align_proba


class RandomForestModel(TrendModel):
    name = "rf"

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 20, random_state: int = 42):
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return align_proba(self._clf.predict_proba(X), self._clf.classes_)


class XGBoostModel(TrendModel):
    name = "xgb"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self._clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        # XGBoost 無 class_weight 參數，改以逐樣本權重達成 balanced 效果
        counts = np.bincount(y.astype(int), minlength=3).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / (np.count_nonzero(counts) * counts), 0.0)
        self._clf.fit(X, y, sample_weight=weights[y.astype(int)])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return align_proba(self._clf.predict_proba(X), self._clf.classes_)


MODEL_FACTORIES = {
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
}
