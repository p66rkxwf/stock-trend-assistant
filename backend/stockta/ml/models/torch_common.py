"""PyTorch 深度模型共用訓練邏輯（LSTM / GRU / TCN 繼承此類）。

輸入沿用 dataset.py 的攤平視窗 (n, window * n_features)：sliding_window_view
後 transpose(0, 2, 1) 再 reshape 是 row-major 的 (window, n_features) 攤平，
可無損 reshape 回 (n, window, n_features)——因此 dataset/predictor/registry/API
全部零改動，序列化走既有 joblib 路徑（fit 完成後權重固定放 CPU）。

訓練細節：Adam、CrossEntropyLoss 帶 balanced class weight（「觀望」過半）、
early stopping 監控驗證集 loss；有 CUDA 用 CUDA，推論一律 CPU（API 部署場景）。
"""

from __future__ import annotations

import abc
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stockta.config import WINDOW_LENGTH_DAYS
from stockta.ml.models.base import N_CLASSES, TrendModel


class TorchTrendModel(TrendModel):
    """三個深度模型的共用骨架；子類只需實作 _build_net()。"""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 5,
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self._net: nn.Module | None = None

    @abc.abstractmethod
    def _build_net(self, n_features: int) -> nn.Module:
        """輸入 (batch, window, n_features)，輸出 (batch, 3) logits。"""
        raise NotImplementedError

    def hyperparams(self) -> dict:
        """存入 artifact metadata，讓比較表與調參結果可追溯。"""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }

    def _to_sequences(self, X: np.ndarray) -> torch.Tensor:
        n_features = X.shape[1] // WINDOW_LENGTH_DAYS
        return torch.from_numpy(
            np.ascontiguousarray(X, dtype=np.float32).reshape(-1, WINDOW_LENGTH_DAYS, n_features)
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TorchTrendModel":
        torch.manual_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_features = X.shape[1] // WINDOW_LENGTH_DAYS
        self._net = self._build_net(n_features).to(device)

        # balanced class weight：n_samples / (n_classes * count)，缺席類別權重 0
        counts = np.bincount(y.astype(int), minlength=N_CLASSES).astype(np.float64)
        weights = np.where(
            counts > 0, counts.sum() / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(self._to_sequences(X), torch.from_numpy(y.astype(np.int64))),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            val_x = self._to_sequences(X_val).to(device)
            val_y = torch.from_numpy(y_val.astype(np.int64)).to(device)

        best_val = math.inf
        best_state: dict | None = None
        bad_epochs = 0
        self.epochs_run_ = 0
        for _epoch in range(self.max_epochs):
            self.epochs_run_ = _epoch + 1
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

            if not has_val:
                continue  # 無驗證集（煙霧測試）：跑滿 max_epochs
            self._net.eval()
            with torch.no_grad():
                val_loss = float(criterion(self._net(val_x), val_y))
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        # 固定放 CPU：joblib 序列化與 API 推論（部署機不保證有 GPU）皆以 CPU 為準
        self._net.cpu().eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        self._net.eval()
        seqs = self._to_sequences(X)
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(seqs), 4096):
                logits = self._net(seqs[i : i + 4096])
                outs.append(torch.softmax(logits, dim=1).numpy())
        if not outs:
            return np.zeros((0, N_CLASSES), dtype=np.float64)
        return np.concatenate(outs).astype(np.float64)


class RecurrentNet(nn.Module):
    """LSTM/GRU 共用網路：取最後一個 time step 的隱狀態接分類頭。"""

    def __init__(
        self,
        rnn_cls: type[nn.RNNBase],
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch 規定單層 RNN 的 inter-layer dropout 無意義，需為 0
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, window, n_features)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])
