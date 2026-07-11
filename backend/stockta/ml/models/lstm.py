"""LSTM 趨勢分類模型（PLAN.md Phase 4）。"""

from __future__ import annotations

from torch import nn

from stockta.ml.models.torch_common import RecurrentNet, TorchTrendModel


class LSTMModel(TorchTrendModel):
    name = "lstm"

    def _build_net(self, n_features: int) -> nn.Module:
        return RecurrentNet(nn.LSTM, n_features, self.hidden_size, self.num_layers, self.dropout)
