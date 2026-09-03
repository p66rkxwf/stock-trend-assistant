"""模型 artifact 存取與版本契約（PLAN.md 架構原則 3）。

存檔時一併寫入 metadata.json（特徵清單與順序、視窗長度、標籤門檻、auto_adjust、
訓練日期、測試指標）；載入時逐項比對當前程式碼的設定，不一致就拒絕載入——
殺掉「舊模型配新特徵默默輸出垃圾預測」這一整類無錯誤訊息的 bug。

metadata.json 進版控（模型權重 model.joblib 由 .gitignore 排除）。
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import StandardScaler

from stockta.config import (
    ARTIFACTS_DIR,
    AUTO_ADJUST,
    LABEL_CLASSES,
    LABEL_DOWN_THRESHOLD,
    LABEL_HORIZON_DAYS,
    LABEL_UP_THRESHOLD,
    WINDOW_LENGTH_DAYS,
)
from stockta.features.pipeline import FEATURE_COLUMNS
from stockta.ml.models.base import TrendModel


class ArtifactContractError(RuntimeError):
    """artifact metadata 與當前程式碼設定不一致，拒絕載入。"""


def save(
    name: str,
    model: TrendModel,
    scaler: StandardScaler,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    out_dir = artifacts_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, out_dir / "model.joblib")

    metadata = {
        "model_name": name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
        "window_length_days": WINDOW_LENGTH_DAYS,
        "label_classes": LABEL_CLASSES,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "label_up_threshold": LABEL_UP_THRESHOLD,
        "label_down_threshold": LABEL_DOWN_THRESHOLD,
        "auto_adjust": AUTO_ADJUST,
        "git_commit": _git_commit(),
        # 訓練當下的資料快照雜湊——git_commit 記的是「哪一版程式碼」，這個記的是
        # 「哪一份資料」。yfinance 的還原股價會回頭改寫歷史，沒有這一欄就無法回答
        # 「這個數字能不能重跑得到」。快取沒有 MANIFEST.json 時為 None（不擋訓練）。
        "data_manifest_sha": _data_manifest_sha(),
        "metrics": metrics,
        **(extra or {}),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def load(
    name: str, artifacts_dir: Path = ARTIFACTS_DIR
) -> tuple[TrendModel, StandardScaler, dict[str, Any]]:
    """載入 artifact 並驗證版本契約；不一致拋 ArtifactContractError。"""
    out_dir = artifacts_dir / name
    meta_path = out_dir / "metadata.json"
    model_path = out_dir / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"artifact 不存在: {out_dir}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _check(metadata, "feature_columns", FEATURE_COLUMNS)
    _check(metadata, "window_length_days", WINDOW_LENGTH_DAYS)
    _check(metadata, "label_classes", LABEL_CLASSES)
    _check(metadata, "label_horizon_days", LABEL_HORIZON_DAYS)
    _check(metadata, "auto_adjust", AUTO_ADJUST)

    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"], metadata


def _check(metadata: dict[str, Any], key: str, expected: Any) -> None:
    actual = metadata.get(key)
    if actual != expected:
        raise ArtifactContractError(
            f"artifact metadata 的 {key} 與當前程式碼不一致：\n"
            f"  artifact: {actual!r}\n  程式碼:   {expected!r}\n"
            f"模型是用舊版特徵/設定訓練的，直接載入會默默輸出錯誤預測，請重新訓練。"
        )


def _data_manifest_sha() -> str | None:
    """訓練當下的資料快照雜湊；沒有 manifest 或讀取失敗時回 None。

    刻意不因此擋下訓練：manifest 是可追溯性的加分項，不是訓練的前置條件，
    在還沒 build 過 manifest 的機器上也應該訓練得起來。
    """
    try:
        from stockta.data.manifest import current_sha

        return current_sha()
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return None
