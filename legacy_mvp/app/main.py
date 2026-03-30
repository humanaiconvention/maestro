from __future__ import annotations

import os
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import router
from app.db import Base, engine
import app.models.domain  # Ensure models are registered

# Create tables for local MVP
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Project Maestro API",
    description="A public Maestro for bounded human-to-AI grounding",
    version="2.0.0",
)

app.include_router(router, prefix="/api")


# ── Sliding-window rate limiter ──────────────────────────────────────
# Applied to all /api/* routes.  Configurable via env-vars:
#   RATE_LIMIT_REQUESTS  — max requests per window (default 100)
#   RATE_LIMIT_WINDOW_S  — sliding window width in seconds (default 60)
#
# Keyed on (client IP, first 8 chars of API key) so that distinct agents
# sharing an IP still get independent quota, while unauthenticated requests
# all share the IP bucket.
#
# Note: this is an in-process store — not suitable for multi-worker
# deployments without a shared cache (Redis).  Adequate for single-worker
# and development use.

_RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
_RATE_LIMIT_WINDOW_S = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
_request_log: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only enforce on /api/ routes
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    api_key_prefix = request.headers.get("X-Maestro-API-Key", "")[:8]
    client_id = f"{client_ip}:{api_key_prefix}"

    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_S

    # Evict timestamps outside the current window
    _request_log[client_id] = [t for t in _request_log[client_id] if t > window_start]

    if len(_request_log[client_id]) >= _RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded.",
                "limit": _RATE_LIMIT_REQUESTS,
                "window_seconds": _RATE_LIMIT_WINDOW_S,
            },
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW_S)},
        )

    _request_log[client_id].append(now)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}
