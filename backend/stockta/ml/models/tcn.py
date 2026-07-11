"""TCN（Temporal Convolutional Network）趨勢分類模型（PLAN.md Phase 4）。

以 causal dilated Conv1d 自行實作（PLAN 要求，不引入第三方 TCN 套件）：
- causal：對時間軸只向左 padding 再裁掉尾端，輸出第 t 步只看得到 ≤ t 的輸入
- dilated：膨脹係數 (1, 2, 4, 8)、kernel 3，感受野 1 + 2*(3-1)*(1+2+4+8) = 61 > 60，
  最後一個 time step 覆蓋整個 60 日視窗
- residual：每個 TemporalBlock 兩層卷積 + 1x1 殘差捷徑（通道數不符時投影）
"""

from __future__ import annotations

import torch
from torch import nn

from stockta.ml.models.base import N_CLASSES
from stockta.ml.models.torch_common import TorchTrendModel


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, ch, time)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNNet(nn.Module):
    DILATIONS = (1, 2, 4, 8)
    KERNEL_SIZE = 3

    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        blocks = []
        in_ch = n_features
        for d in self.DILATIONS:
            blocks.append(TemporalBlock(in_ch, hidden_size, self.KERNEL_SIZE, d, dropout))
            in_ch = hidden_size
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, window, n_features)
        out = self.tcn(x.transpose(1, 2))  # Conv1d 吃 (batch, ch, time)
        return self.head(out[:, :, -1])    # causal：最後一步已覆蓋整個視窗


class TCNModel(TorchTrendModel):
    name = "tcn"

    def _build_net(self, n_features: int) -> nn.Module:
        # num_layers 對 TCN 無意義（層數由 DILATIONS 決定），沿用其餘超參數
        return TCNNet(n_features, self.hidden_size, self.dropout)
