"""打亂標籤測試（label-permutation null test）——洩漏防治的回歸防線。

把標籤隨機打亂、整條管線原封不動重跑，表現必須崩塌到基線水準。沒崩塌，
就代表管線裡有洩漏，而且是嚴重的那種。

**它抓得到什麼、抓不到什麼**（不寫清楚，綠燈只是自我安慰）：

- 抓得到：標籤資訊回流管線的洩漏——訓練/測試樣本重複或高度重疊、預處理在
  全樣本上 fit、任何由標籤衍生出來的特徵。
- 抓不到：純前視特徵（特徵含 t 之後的資訊）。標籤一打亂，前視特徵一樣派不上
  用場，這條測試對它是盲的——那由 test_no_lookahead.py 的「竄改未來、逐值比對」
  負責。**兩者互補，缺一不可。**

關鍵設計一：置換綁在 (ticker, 日期) 鍵上、**在切分之前**完成，train/val/test 共用
同一份假標籤。只打亂訓練標籤就抓不到「跨切分重複樣本」——而重疊視窗（相鄰樣本
共用 59/60 歷史）正是本專案最可能的洩漏形態。

關鍵設計二：置換必須**跨全池匯總後再洗**（scope="pooled"），不能逐檔各洗各的。
逐檔置換只打斷「同一檔內特徵與標籤的時間對應」，卻保留了**各檔之間標籤分佈的
差異**：實測全池 49 檔的「觀望」占比從 0.222（世芯-KY，高波動）到 0.852（中華電，
牛皮）差近 4 倍，而波動率、通道寬度、量能這些特徵本來就認得出是哪一類股票。
模型於是學得到「這種特徵的股票，標籤分佈長這樣」——2026-08-16 實測逐檔置換下
macro AUC 停在 0.577±0.004（5 個種子、n_test=4,954）而非 0.5。

**那不是洩漏，是虛無假設沒設乾淨。** 全域置換把跨檔差異也一併打散，才得到真正的
虛無分佈。這個坑值得記著：虛無假設設錯，測出來的紅燈會讓人去追一個不存在的 bug；
而若當初只用同質的合成資料驗證，這個坑根本不會浮現。
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd

from stockta.ml.evaluate import evaluate
from stockta.ml.models.base import TrendModel

if TYPE_CHECKING:  # 只為型別標註，執行期不匯入（避免與 dataset 循環）
    from stockta.ml.dataset import Dataset

# 打亂標籤後允許的殘餘優勢，超過即判定洩漏。
# 測試樣本 n≈2,000 時 macro AUC 的標準誤約 0.01，0.03 約等於三個標準誤——
# 純噪音碰不到，真有洩漏時的訊號又遠大於此。
AUC_TOLERANCE = 0.03
ACCURACY_TOLERANCE = 0.03


def ticker_seed(ticker: str, seed: int) -> int:
    """由 (ticker, seed) 導出穩定的置換種子。

    不能用內建 hash()——str 的 hash 受 PYTHONHASHSEED 影響，跨行程會變，
    測試就不可重現。blake2b 在任何機器、任何 Python 行程都給同一個值。
    """
    digest = hashlib.blake2b(ticker.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") ^ (seed & 0xFFFF_FFFF_FFFF_FFFF)) % (2**63)


def permute_labels(labels: pd.Series, seed: int) -> pd.Series:
    """隨機置換標籤值，NaN 留在原位。

    NaN 不動是刻意的：build_dataset 以 ~isnan(y) 決定哪些樣本有效，NaN 若被洗到
    別的位置，打亂版與真實版的樣本數與切分邊界就會不同，兩者的差異也就不再只
    來自標籤。對照組必須只差一個變因。
    """
    values = labels.to_numpy(dtype=np.float64, copy=True)
    mask = ~np.isnan(values)
    drawn = values[mask]
    np.random.default_rng(seed).shuffle(drawn)
    values[mask] = drawn
    return pd.Series(values, index=labels.index, name=labels.name)


def permute_labels_pooled(
    label_series: Sequence[pd.Series], seed: int
) -> list[pd.Series]:
    """把所有標的的標籤匯進同一個池全域置換後發回，NaN 各自留在原位。

    這是**預設且正確**的虛無假設（理由見模組 docstring 的「關鍵設計二」）：
    逐檔置換會留下各檔標籤分佈的差異，那是特徵認得出來的，於是虛無分佈不會落在
    0.5，紅燈變成假警報。全域置換後所有樣本可交換，才是乾淨的虛無。
    """
    arrays = [s.to_numpy(dtype=np.float64, copy=True) for s in label_series]
    masks = [~np.isnan(a) for a in arrays]

    pool = np.concatenate([a[m] for a, m in zip(arrays, masks)]) if arrays else np.empty(0)
    np.random.default_rng(seed).shuffle(pool)

    out: list[pd.Series] = []
    offset = 0
    for series, array, mask in zip(label_series, arrays, masks):
        count = int(mask.sum())
        array[mask] = pool[offset : offset + count]
        offset += count
        out.append(pd.Series(array, index=series.index, name=series.name))
    return out


@dataclasses.dataclass(frozen=True)
class PipelineScore:
    """一次「訓練 → 測試」的成績，附帶與基線的距離。"""

    macro_auc: float
    accuracy: float
    majority_baseline: float
    n_train: int
    n_test: int

    @property
    def auc_excess(self) -> float:
        """超出隨機猜測（AUC 0.5）的幅度。"""
        return self.macro_auc - 0.5

    @property
    def accuracy_excess(self) -> float:
        """超出多數類基線的幅度。"""
        return self.accuracy - self.majority_baseline

    def leaked(
        self, auc_tol: float = AUC_TOLERANCE, acc_tol: float = ACCURACY_TOLERANCE
    ) -> bool:
        """單尾判定：打亂標籤後仍明顯**優於**基線才算洩漏。

        低於基線只是噪音或模型無能，不構成洩漏的證據，故不做雙尾。
        """
        return self.auc_excess > auc_tol or self.accuracy_excess > acc_tol

    def summary(self) -> str:
        return (
            f"macro AUC {self.macro_auc:.4f}（超出隨機 {self.auc_excess:+.4f}）、"
            f"準確率 {self.accuracy:.4f} vs 多數類基線 {self.majority_baseline:.4f}"
            f"（超出 {self.accuracy_excess:+.4f}）、"
            f"訓練 {self.n_train} 筆／測試 {self.n_test} 筆"
        )


def score_dataset(
    dataset: "Dataset", model_factory: Callable[[], TrendModel]
) -> PipelineScore:
    """在既有 Dataset 上訓練並評估測試集。"""
    model = model_factory().fit(
        dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val
    )
    metrics = evaluate(model, dataset.X_test, dataset.y_test)
    return PipelineScore(
        macro_auc=float(metrics["macro_auc"]),
        accuracy=float(metrics["accuracy"]),
        majority_baseline=float(metrics["majority_baseline_accuracy"]),
        n_train=int(len(dataset.y_train)),
        n_test=int(len(dataset.y_test)),
    )


def score_pipeline(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    *,
    model_factory: Callable[[], TrendModel],
    label_permutation_seed: int | None = None,
    **dataset_kwargs,
) -> PipelineScore:
    """走完整條管線（建特徵 → 切分 → 標準化 → 訓練 → 評估）。

    label_permutation_seed 給值即為打亂標籤的對照跑法；其餘參數與真實跑法完全相同。
    """
    # 延後匯入：dataset 在模組頂層匯入本模組的 permute_labels，頂層互相匯入會循環
    from stockta.ml.dataset import build_dataset

    dataset = build_dataset(
        ohlcv_by_ticker,
        context,
        label_permutation_seed=label_permutation_seed,
        **dataset_kwargs,
    )
    return score_dataset(dataset, model_factory)


# === 刻意注入的洩漏（對照組）=================================================
# 用途只有一個：證明上面那條檢查真的抓得到東西。一條永遠綠的檢查等於沒有檢查，
# 所以測試裡必須有「已知會紅」的對照，否則管線整條壞掉時打亂標籤測試也會過。


def inject_duplicate_rows(dataset: "Dataset", n: int = 400) -> "Dataset":
    """把測試集前 n 列原樣塞進訓練集——製造「跨切分重複樣本」。

    這是最常見也最難察覺的洩漏形態之一（去重沒做乾淨就會發生，news 專案的
    dedup_indices 正是為此而存在）。模型只要背下這些列，測試分數就會虛高，
    且**與標籤是真是假無關**——所以打亂標籤後它依然抓得到。
    """
    k = min(n, len(dataset.y_test))
    if k == 0:
        return dataset
    return dataclasses.replace(
        dataset,
        X_train=np.concatenate([dataset.X_train, dataset.X_test[:k]]),
        y_train=np.concatenate([dataset.y_train, dataset.y_test[:k]]),
    )


def inject_label_derived_feature(dataset: "Dataset", noise: float = 0.3) -> "Dataset":
    """在特徵尾端接上一欄由標籤直接算出的值——製造「目標洩漏進特徵」。

    對應清單上的「標準化用的 mean／std 是不是用了全樣本統計」那一類問題的極端版：
    只要有任何一欄帶著標籤的影子，打亂標籤後它會跟著假標籤走，分數就不會崩塌。
    """
    rng = np.random.default_rng(0)

    def with_column(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.empty((0, X.shape[1] + 1), dtype=np.float32)
        col = y.astype(np.float32) + rng.normal(0, noise, len(y)).astype(np.float32)
        return np.column_stack([X, col]).astype(np.float32)

    return dataclasses.replace(
        dataset,
        X_train=with_column(dataset.X_train, dataset.y_train),
        X_val=with_column(dataset.X_val, dataset.y_val),
        X_test=with_column(dataset.X_test, dataset.y_test),
    )
