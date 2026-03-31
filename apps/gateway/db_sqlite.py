"""
db_sqlite.py — SQLite-backed receipt storage for Maestro Gateway.

Used as the default persistence backend when DATABASE_URL is not set.
Zero configuration: creates a SQLite database file at the configured path.

Environment:
    SQLITE_DB_PATH  Path to SQLite database file
                    (default: data/maestro.db relative to repo root)
"""

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default: data/maestro.db in the repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
SQLITE_DB_PATH: str = os.environ.get(
    "SQLITE_DB_PATH",
    str(_REPO_ROOT / "data" / "maestro.db"),
)

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = Path(SQLITE_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db() -> None:
    """Create the participation_receipts table if it doesn't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS participation_receipts (
            receipt_id  TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            merkle_root TEXT NOT NULL UNIQUE,
            payload     TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_receipts_session_id
            ON participation_receipts (session_id);
        CREATE INDEX IF NOT EXISTS idx_receipts_merkle_root
            ON participation_receipts (merkle_root);
    """)
    logger.info("SQLite participation_receipts table ready at %s", SQLITE_DB_PATH)


def store_receipt(
    receipt_id: str,
    session_id: str,
    merkle_root: str,
    payload_dict: dict,
) -> None:
    """Insert a receipt row; silently skips if merkle_root already exists."""
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR IGNORE INTO participation_receipts
            (receipt_id, session_id, merkle_root, payload)
        VALUES (?, ?, ?, ?)
        """,
        (receipt_id, session_id, merkle_root, json.dumps(payload_dict, ensure_ascii=False)),
    )
    conn.commit()


def get_receipt(merkle_root: str) -> Optional[dict]:
    """Return the deserialized receipt payload, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT payload FROM participation_receipts WHERE merkle_root = ?",
        (merkle_root,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["payload"])


def list_receipts(since: Optional[str] = None) -> list[dict]:
    """Return all receipt rows as dicts."""
    conn = _get_conn()
    if since:
        rows = conn.execute(
            """
            SELECT receipt_id, session_id, merkle_root, payload, created_at
            FROM participation_receipts
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT receipt_id, session_id, merkle_root, payload, created_at
            FROM participation_receipts
            ORDER BY created_at ASC
            """
        ).fetchall()

    return [
        {
            "receipt_id":  row["receipt_id"],
            "session_id":  row["session_id"],
            "merkle_root": row["merkle_root"],
            "payload":     json.loads(row["payload"]),
            "created_at":  row["created_at"],
        }
        for row in rows
    ]
