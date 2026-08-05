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


_RANK_SCHEMA = """
CREATE TABLE IF NOT EXISTS rank_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    base_date TEXT NOT NULL,
    score REAL NOT NULL,
    model_version TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'live',
    created_at TEXT NOT NULL,
    UNIQUE (ticker, base_date, model_version)
);
"""


class RankPredictionStore:
    """Cross-sectional 相對強弱分數落地（append-only，與 PredictionStore 並存同一 DB）。

    分數＝P(未來 LABEL_HORIZON_DAYS 日贏過當日全池中位數)，用於全池排序。與 3 類絕對
    方向的 predictions 表互不干擾。累積後由 report_rank_predictions 算**線上 Rank IC**
    ——選股技能的活證據，且不受多頭 beta 影響（問「贏過中位數」而非「漲」）。

    同 (ticker, 基準日, 模型版本) 只記第一筆，重複呼叫不灌水。source：'live'＝當日收盤後
    即時記錄；'pit'＝以固定權重點對點重建的歷史基準日（point-in-time、無前視，與
    /api/rank?date= 同一路徑）。
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_RANK_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(
        self, ticker: str, base_date: date, score: float, model_version: str, source: str = "live"
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO rank_predictions "
                "(ticker, base_date, score, model_version, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    base_date.isoformat(),
                    float(score),
                    model_version,
                    source,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM rank_predictions").fetchone()[0]
