"""
db.py — Postgres-backed receipt storage for Maestro Gateway.

Replaces the in-memory dict in participation.py with a connection-pooled
psycopg2 store.  Table is created on startup if it doesn't exist.

Environment:
    DATABASE_URL  postgresql://user:pass@host:5432/db
                  (default: postgresql://postgres:postgres@localhost:5432/maestro)
"""

import json
import logging
import os
from typing import Optional

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

# DATABASE_URL must be set explicitly in production. The fallback below uses
# default Postgres credentials and is only safe for local development without
# a .env file. Never rely on this fallback outside of a dev environment.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/maestro",
)

_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
    return _pool


def init_db() -> None:
    """Create the participation_receipts table if it doesn't exist."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS participation_receipts (
                        receipt_id  TEXT PRIMARY KEY,
                        session_id  TEXT NOT NULL,
                        merkle_root TEXT NOT NULL UNIQUE,
                        payload     JSONB NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_receipts_session_id
                    ON participation_receipts (session_id)
                """)
        logger.info("participation_receipts table ready")
    except Exception as e:
        logger.error(f"db.init_db failed: {e}")
        raise
    finally:
        pool.putconn(conn)


def store_receipt(
    receipt_id: str,
    session_id: str,
    merkle_root: str,
    payload_dict: dict,
) -> None:
    """Insert a receipt row; silently skips if merkle_root already exists."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO participation_receipts
                        (receipt_id, session_id, merkle_root, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (merkle_root) DO NOTHING
                    """,
                    (receipt_id, session_id, merkle_root, json.dumps(payload_dict)),
                )
    finally:
        pool.putconn(conn)


def get_receipt(merkle_root: str) -> Optional[dict]:
    """Return the deserialized receipt payload, or None if not found."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM participation_receipts WHERE merkle_root = %s",
                    (merkle_root,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                # psycopg2 deserializes JSONB to a Python dict automatically
                payload = row[0]
                return payload if isinstance(payload, dict) else json.loads(payload)
    finally:
        pool.putconn(conn)


def list_receipts(since: Optional[str] = None) -> list[dict]:
    """
    Return all receipt rows as dicts with keys:
        receipt_id, session_id, merkle_root, payload, created_at (ISO string).

    Optionally filters to rows where created_at >= `since` (ISO 8601 string).
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                if since:
                    cur.execute(
                        """
                        SELECT receipt_id, session_id, merkle_root, payload,
                               created_at AT TIME ZONE 'UTC' AS created_at
                        FROM participation_receipts
                        WHERE created_at >= %s::timestamptz
                        ORDER BY created_at ASC
                        """,
                        (since,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT receipt_id, session_id, merkle_root, payload,
                               created_at AT TIME ZONE 'UTC' AS created_at
                        FROM participation_receipts
                        ORDER BY created_at ASC
                        """
                    )
                rows = cur.fetchall()
                result = []
                for receipt_id, session_id, merkle_root, payload, created_at in rows:
                    result.append({
                        "receipt_id":  receipt_id,
                        "session_id":  session_id,
                        "merkle_root": merkle_root,
                        "payload":     payload if isinstance(payload, dict) else json.loads(payload),
                        "created_at":  created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                    })
                return result
    finally:
        pool.putconn(conn)
