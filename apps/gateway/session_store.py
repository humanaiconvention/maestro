"""
session_store.py — Lightweight server-side session cache for the Maestro gateway.

Purpose
───────
The interview flow is stateless by design: the frontend owns the full
conversation history and replays it on every request.  This module adds a
thin server-side record of *session metadata* — turn counts, last-activity
timestamps, consent state — without duplicating message content.

What is stored per session
──────────────────────────
SessionMeta fields:
  session_id   str         JWT session_id claim (UUID)
  created_at   float       Unix timestamp of first chat message
  last_active  float       Unix timestamp of most recent chat message
  turn_count   int         Number of completed assistant turns
  kernel_type  str|None    "transcript" | "concept" (set after consent)
  consented    bool        True once consent JWT has been verified
  closed       bool        True after terminal signal or explicit close

Storage
───────
In-memory dict keyed by session_id, with a background TTL sweep.
When Redis is available (REDIS_URL env var), the store serialises to JSON
and uses Redis hashes with EXPIRE — giving session persistence across
gateway restarts and horizontal scaling.

TTL is SESSION_CACHE_TTL seconds (default 10 800 s = 3 h).  Sessions are
evicted lazily on read and actively on the periodic sweep.

Concurrency
───────────
In-memory store uses threading.Lock for read-modify-write safety.
Redis operations are atomic (HSET + EXPIRE in a pipeline).

Usage
─────
    from apps.gateway.session_store import session_store

    # On each chat completion:
    session_store.touch(session_id, kernel_type=..., consented=...)
    session_store.increment_turn(session_id)

    # To inspect a session:
    meta = session_store.get(session_id)

    # Cleanup (called by gateway lifespan):
    session_store.sweep()
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_CACHE_TTL = int(os.environ.get("SESSION_CACHE_TTL", str(3 * 3600)))  # 3h default


# ─────────────────────────────────────────────────────────────────────────────
# Session metadata dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionMeta:
    session_id:  str
    created_at:  float          = field(default_factory=time.time)
    last_active: float          = field(default_factory=time.time)
    turn_count:  int            = 0
    kernel_type: Optional[str]  = None
    consented:   bool           = False
    closed:      bool           = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_active


# ─────────────────────────────────────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────────────────────────────────────

class _InMemorySessionStore:
    def __init__(self, ttl: int = SESSION_CACHE_TTL) -> None:
        self._store: dict[str, SessionMeta] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def _is_expired(self, meta: SessionMeta) -> bool:
        return meta.idle_seconds > self._ttl

    def get(self, session_id: str) -> Optional[SessionMeta]:
        with self._lock:
            meta = self._store.get(session_id)
            if meta is None:
                return None
            if self._is_expired(meta):
                del self._store[session_id]
                return None
            return meta

    def touch(
        self,
        session_id: str,
        kernel_type: Optional[str] = None,
        consented: bool = False,
    ) -> SessionMeta:
        """
        Record or update a session.

        Creates a new SessionMeta on first call; updates last_active,
        kernel_type, and consented on subsequent calls.  Never decrements
        consented from True to False (one-way gate).
        """
        with self._lock:
            meta = self._store.get(session_id)
            if meta is None:
                meta = SessionMeta(session_id=session_id)
                self._store[session_id] = meta
                logger.debug("session_created", extra={"session_id": session_id})
            meta.last_active = time.time()
            if kernel_type:
                meta.kernel_type = kernel_type
            if consented and not meta.consented:
                meta.consented = True
            return meta

    def increment_turn(self, session_id: str) -> int:
        """Increment the turn counter and return the new value."""
        with self._lock:
            meta = self._store.get(session_id)
            if meta is None:
                meta = SessionMeta(session_id=session_id)
                self._store[session_id] = meta
            meta.turn_count += 1
            meta.last_active = time.time()
            return meta.turn_count

    def close_session(self, session_id: str) -> None:
        """Mark a session as closed (terminal signal received)."""
        with self._lock:
            meta = self._store.get(session_id)
            if meta:
                meta.closed = True

    def sweep(self) -> int:
        """Evict expired sessions.  Returns number of sessions removed."""
        with self._lock:
            expired = [sid for sid, m in self._store.items() if self._is_expired(m)]
            for sid in expired:
                del self._store[sid]
        if expired:
            logger.info(
                "session_sweep",
                extra={"evicted": len(expired), "remaining": len(self._store)},
            )
        return len(expired)

    def stats(self) -> dict:
        with self._lock:
            total     = len(self._store)
            active    = sum(1 for m in self._store.values() if not m.closed and not self._is_expired(m))
            consented = sum(1 for m in self._store.values() if m.consented)
        return {"total": total, "active": active, "consented": consented}


# ─────────────────────────────────────────────────────────────────────────────
# Redis-backed store (optional)
# ─────────────────────────────────────────────────────────────────────────────

class _RedisSessionStore:
    """
    Redis-backed session store using hashes: key = ``session:{session_id}``.

    Falls back gracefully to a no-op if Redis is unavailable.
    """
    _KEY_PREFIX = "maestro:session:"

    def __init__(self, redis_url: str, ttl: int = SESSION_CACHE_TTL) -> None:
        self._ttl = ttl
        self._r = None
        self._fallback = _InMemorySessionStore(ttl=ttl)
        try:
            import redis
            self._r = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            self._r.ping()
            logger.info("session_store: connected to Redis")
        except Exception as exc:
            logger.warning(
                "session_store: Redis unavailable, using in-memory fallback",
                extra={"error": str(exc)},
            )
            self._r = None

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    def _load(self, session_id: str) -> Optional[SessionMeta]:
        if self._r is None:
            return self._fallback.get(session_id)
        try:
            data = self._r.hgetall(self._key(session_id))
            if not data:
                return None
            # Coerce stored strings back to correct types
            data["created_at"]  = float(data.get("created_at", 0))
            data["last_active"] = float(data.get("last_active", 0))
            data["turn_count"]  = int(data.get("turn_count", 0))
            data["consented"]   = data.get("consented") == "True"
            data["closed"]      = data.get("closed") == "True"
            return SessionMeta.from_dict(data)
        except Exception as exc:
            logger.warning("session_store: Redis read error", extra={"error": str(exc)})
            return self._fallback.get(session_id)

    def _save(self, meta: SessionMeta) -> None:
        if self._r is None:
            return
        try:
            d = {k: str(v) for k, v in meta.to_dict().items()}
            pipe = self._r.pipeline()
            pipe.hset(self._key(meta.session_id), mapping=d)
            pipe.expire(self._key(meta.session_id), self._ttl)
            pipe.execute()
        except Exception as exc:
            logger.warning("session_store: Redis write error", extra={"error": str(exc)})

    def get(self, session_id: str) -> Optional[SessionMeta]:
        return self._load(session_id)

    def touch(
        self,
        session_id: str,
        kernel_type: Optional[str] = None,
        consented: bool = False,
    ) -> SessionMeta:
        meta = self._load(session_id)
        if meta is None:
            meta = SessionMeta(session_id=session_id)
            if self._r is None:
                return self._fallback.touch(session_id, kernel_type=kernel_type, consented=consented)
        meta.last_active = time.time()
        if kernel_type:
            meta.kernel_type = kernel_type
        if consented and not meta.consented:
            meta.consented = True
        self._save(meta)
        return meta

    def increment_turn(self, session_id: str) -> int:
        if self._r is None:
            return self._fallback.increment_turn(session_id)
        try:
            key = self._key(session_id)
            turn_count = self._r.hincrby(key, "turn_count", 1)
            self._r.hset(key, "last_active", str(time.time()))
            self._r.expire(key, self._ttl)
            return int(turn_count)
        except Exception as exc:
            logger.warning("session_store: Redis incr error", extra={"error": str(exc)})
            return self._fallback.increment_turn(session_id)

    def close_session(self, session_id: str) -> None:
        if self._r is None:
            self._fallback.close_session(session_id)
            return
        try:
            self._r.hset(self._key(session_id), "closed", "True")
        except Exception:
            self._fallback.close_session(session_id)

    def sweep(self) -> int:
        # Redis handles TTL natively; sweep is a no-op here
        return 0

    def stats(self) -> dict:
        if self._r is None:
            return self._fallback.stats()
        try:
            keys = self._r.keys(f"{self._KEY_PREFIX}*")
            return {"total": len(keys), "active": None, "consented": None}
        except Exception:
            return {"total": None, "active": None, "consented": None}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

def _build_store() -> "_InMemorySessionStore | _RedisSessionStore":
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        return _RedisSessionStore(redis_url)
    return _InMemorySessionStore()


session_store: "_InMemorySessionStore | _RedisSessionStore" = _build_store()
