"""深度模型工廠（此模組 import torch，呼叫端需延遲載入）。

train.py / tune.py 只有在 --model lstm|gru|tcn 時才 import 此模組，
無 torch 的環境仍可正常訓練 rf/xgb 基線。
"""

from __future__ import annotations

from stockta.ml.models.gru import GRUModel
from stockta.ml.models.lstm import LSTMModel
from stockta.ml.models.tcn import TCNModel

DEEP_FACTORIES = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "tcn": TCNModel,
}
