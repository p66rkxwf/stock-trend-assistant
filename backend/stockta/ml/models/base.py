"""統一模型介面：predict_proba 輸出欄位順序固定為 config.LABEL_CLASSES。

sklearn/xgboost 的 classes_ 順序取決於訓練資料中出現的類別，不保證是 0/1/2
完整三類；此介面負責把機率欄位對齊到固定順序，缺席類別補 0 機率——
下游（評估、API 信心分數）永遠拿到 (n, 3) 且欄位語義固定的矩陣。
"""

from __future__ import annotations

import abc

import numpy as np

from stockta.config import LABEL_CLASSES

N_CLASSES = len(LABEL_CLASSES)


class TrendModel(abc.ABC):
    """三分類趨勢模型的統一介面。"""

    name: str = "base"

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrendModel":
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳 (n, 3) 機率矩陣，欄位順序 = LABEL_CLASSES（跌/觀望/漲）。"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """把底層模型輸出的機率欄位對齊到 0..N_CLASSES-1 的固定順序。"""
    aligned = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
    for col, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, col]
    return aligned
