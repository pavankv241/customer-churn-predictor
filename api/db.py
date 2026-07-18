"""SQLite helpers for prediction audit history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import ROOT

DB_PATH = ROOT / "data" / "predictions.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                customer_id TEXT,
                churn_probability REAL NOT NULL,
                predicted_churn TEXT NOT NULL,
                risk_band TEXT NOT NULL,
                threshold REAL NOT NULL,
                payload_json TEXT
            )
            """
        )


def insert_prediction(
    *,
    source: str,
    customer_id: str | None,
    churn_probability: float,
    predicted_churn: str,
    risk_band: str,
    threshold: float,
    payload_json: str | None = None,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                created_at, source, customer_id, churn_probability,
                predicted_churn, risk_band, threshold, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                source,
                customer_id,
                churn_probability,
                predicted_churn,
                risk_band,
                threshold,
                payload_json,
            ),
        )
        return int(cur.lastrowid)


def list_predictions(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, source, customer_id, churn_probability,
                   predicted_churn, risk_band, threshold
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
