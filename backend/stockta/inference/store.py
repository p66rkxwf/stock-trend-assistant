"""預測結果落地 SQLite（append-only）。

每次線上推論都記一筆——競賽前累積數週「線上預測 vs 實際走勢」的實證資料，
比測試集 AUC 更有說服力（PLAN.md Phase 6）。同一 (ticker, 基準日, 模型版本)
只記第一筆，重複呼叫不會灌水。
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    base_date TEXT NOT NULL,
    signal TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (ticker, base_date, model_version)
);
"""


class PredictionStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(
        self, ticker: str, base_date: date, signal: str, confidence: float, model_version: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO predictions "
                "(ticker, base_date, signal, confidence, model_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    base_date.isoformat(),
                    signal,
                    confidence,
                    model_version,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
