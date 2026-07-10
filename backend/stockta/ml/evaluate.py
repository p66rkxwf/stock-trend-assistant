"""評估指標：Accuracy、Macro F1、macro AUC-ROC（one-vs-rest）、混淆矩陣。

多數類基線（全猜「觀望」）一併回報——申請表自評標準是「優於隨機猜測」，
但真正該贏的對手是多數類基線，Accuracy 單獨看會被類別不平衡灌水。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from stockta.config import LABEL_CLASSES
from stockta.ml.models.base import TrendModel


def evaluate(model: TrendModel, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)

    present = np.unique(y).astype(int)
    if len(present) == 3:
        macro_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    else:  # 極端情況（子集資料缺類別）AUC 無定義
        macro_auc = float("nan")

    majority = np.bincount(y.astype(int), minlength=3).argmax()
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_auc": macro_auc,
        "majority_class": LABEL_CLASSES[int(majority)],
        "majority_baseline_accuracy": float((y == majority).mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "label_distribution": {
            LABEL_CLASSES[i]: int((y == i).sum()) for i in range(3)
        },
    }
