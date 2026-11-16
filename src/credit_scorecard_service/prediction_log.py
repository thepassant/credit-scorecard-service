"""Append-only prediction log in SQLite.

Each scored applicant becomes one row keyed by (request_id, row_index), so
real outcomes can later be joined back for monitoring and retraining.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from credit_scorecard_service.scorecard import ScoreResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_json TEXT NOT NULL,
    probability_of_default REAL NOT NULL,
    score INTEGER NOT NULL,
    risk_band TEXT NOT NULL,
    decision TEXT NOT NULL,
    PRIMARY KEY (request_id, row_index)
)
"""


class PredictionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per write keeps this safe across worker threads.
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def write(
        self,
        request_id: str,
        model_version: str,
        inputs: Sequence[dict[str, Any]],
        results: Sequence[ScoreResult],
    ) -> None:
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        rows = [
            (
                request_id,
                i,
                now,
                model_version,
                json.dumps(inp),
                res.probability_of_default,
                res.score,
                res.risk_band,
                res.decision,
            )
            for i, (inp, res) in enumerate(zip(inputs, results, strict=True))
        ]
        with self._connect() as conn:
            conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
