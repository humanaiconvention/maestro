import datetime
import os
import uuid
import time
import logging
import traceback
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import jwt

from libs.schemas.api import ChatCompletionRequest
from libs.adapters.mock import MockAdapter
from apps.gateway.auth import verify_auth, ALGORITHM
from apps.gateway.captcha import generate_challenge, verify_solution
from apps.gateway.participation import register_receipt_routes
from apps.gateway.export import register_export_routes
from apps.gateway.prism import register_prism_routes
from apps.gateway.rate_limiter import SlidingWindowRateLimiter, RedisRateLimiter
from apps.gateway.logging_config import configure_logging
from apps.gateway.consent import register_consent_routes
from apps.gateway.session_store import session_store
from apps.gateway import db
from legacy_mvp.shared_runtime import (
    LaunchMode,
    launch_readonly_json_response,
    validate_gateway_runtime,
)

from fastapi.middleware.cors import CORSMiddleware

RUNTIME_SETTINGS = validate_gateway_runtime()

# ── Structured logging ────────────────────────────────────────────────────────
# Replaces basicConfig.  LOG_FORMAT=json → JSON lines; default → human-readable.
configure_logging()
logger = logging.getLogger(__name__)

# Track gateway start time for uptime reporting
_GATEWAY_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on missing secrets in non-readonly mode
    if RUNTIME_SETTINGS.launch_mode != LaunchMode.PUBLIC_READONLY:
        if not os.environ.get("MAESTRO_JWT_SECRET"):
            raise RuntimeError(
                "MAESTRO_JWT_SECRET must be set when not running in public_readonly mode. "
                "Generate a secret with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
    loop = asyncio.get_event_loop()
    if os.environ.get("DATABASE_URL"):
        await loop.run_in_executor(None, db.init_db)
    yield
    # shutdown — flush session store stats to log
    stats = session_store.stats()
    logger.info("session_store_shutdown", extra={"stats": stats})


app = FastAPI(title="Maestro API Gateway", lifespan=lifespan)


# CORS — allow frontend origins (comma-separated in MAESTRO_CORS_ORIGINS env var)
_cors_raw = os.environ.get("MAESTRO_CORS_ORIGINS", "")
if not _cors_raw:
    logger.warning(
        "MAESTRO_CORS_ORIGINS not set; defaulting to http://localhost:5173. "
        "Set this env var in production."
    )
    _cors_raw = "http://localhost:5173"
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Idempotency-Key"],
)

# Register participation receipt routes (POST/GET /v1/session/receipt/...)
register_receipt_routes(app)

# Register admin export routes (GET /v1/export/kernels)
register_export_routes(app)

# Register PRISM routes (GET /v1/prism/status, /v1/prism/runs, /v1/prism/runs/{id})
register_prism_routes(app)

# Rate Limiter Initialization
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))
if os.environ.get("REDIS_URL"):
    rate_limiter = RedisRateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60)
else:
    rate_limiter = SlidingWindowRateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60)

# Separate, stricter rate limiter for session-verify (anti-PoW farming)
# Default: 10 verifications per 10 minutes per IP
_SESSION_VERIFY_RPM = int(os.environ.get("SESSION_VERIFY_RPM", "10"))
_session_verify_limiter = SlidingWindowRateLimiter(
    max_requests=_SESSION_VERIFY_RPM, window_seconds=600
)

# When TRUST_PROXY_HEADERS=true, use X-Real-IP or X-Forwarded-For to determine
# the real client IP.  Enable only when behind a trusted reverse proxy (NGINX,
# Traefik, etc.) — spoofable otherwise.
_TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "false").lower() == "true"


def _get_client_ip(request: Request) -> str:
    """Return the best available client IP string for rate-limiting purposes."""
    if _TRUST_PROXY_HEADERS:
        real_ip = (
            request.headers.get("X-Real-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        )
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"

# JWT Secret for middleware-level extraction
MAESTRO_JWT_SECRET = os.environ.get("MAESTRO_JWT_SECRET")

# Register one-way consent gate routes (POST /v1/session/consent)
# Always registered; enforcement in chat route is opt-in via CONSENT_GATE_ENABLED.
SESSION_EXPIRY_SECONDS = int(os.environ.get("SESSION_EXPIRY_SECONDS", "7200"))
register_consent_routes(app, MAESTRO_JWT_SECRET or "", SESSION_EXPIRY_SECONDS)

# (SESSION_EXPIRY_SECONDS defined above near consent route registration)

# Session Models
class ChallengeResponse(BaseModel):
    algorithm: str
    challenge: str
    salt: str
    signature: str
    maxnumber: int

class VerifyRequest(BaseModel):
    payload: str   # base64-encoded Altcha solution from the widget

class SessionResponse(BaseModel):
    token: str
    expires_in: int

# Health Models
class HealthResponse(BaseModel):
    status: str

class LiveResponse(BaseModel):
    alive: bool

class ReadyResponse(BaseModel):
    ready: bool
    adapter: str
    anthropic: Optional[str] = None

# Read adapter selection from environment
# Priority: USE_LLAMACPP_ADAPTER > USE_LOCAL_ADAPTER > USE_MOCK_ADAPTER > AnthropicAdapter
USE_LLAMACPP_ADAPTER = os.environ.get("USE_LLAMACPP_ADAPTER", "false").lower() == "true"
USE_LOCAL_ADAPTER = os.environ.get("USE_LOCAL_ADAPTER", "false").lower() == "true"
USE_MOCK_ADAPTER_STR = os.environ.get("USE_MOCK_ADAPTER", "true").lower()
USE_MOCK_ADAPTER = USE_MOCK_ADAPTER_STR == "true"

adapter = None
if RUNTIME_SETTINGS.launch_mode == LaunchMode.PUBLIC_READONLY:
    logger.info("Initializing Gateway in public_readonly mode; chat adapter disabled.")
elif USE_LLAMACPP_ADAPTER:
    logger.info("Initializing Gateway with LlamaCppAdapter (USE_LLAMACPP_ADAPTER=true)")
    try:
        from libs.adapters.llama_cpp_adapter import LlamaCppAdapter
        adapter = LlamaCppAdapter()
    except (ValueError, ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to initialize LlamaCppAdapter: {e}. Falling back to MockAdapter.")
        adapter = MockAdapter()
elif USE_LOCAL_ADAPTER:
    logger.info("Initializing Gateway with LocalModelAdapter (USE_LOCAL_ADAPTER=true)")
    try:
        from libs.adapters.local_model_adapter import LocalModelAdapter
        adapter = LocalModelAdapter()
    except (ValueError, ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to initialize LocalModelAdapter: {e}. Falling back to MockAdapter.")
        adapter = MockAdapter()
elif USE_MOCK_ADAPTER:
    logger.info("Initializing Gateway with MockAdapter (USE_MOCK_ADAPTER=true)")
    adapter = MockAdapter()
else:
    logger.info("Initializing Gateway with AnthropicAdapter (USE_MOCK_ADAPTER=false)")
    try:
        from libs.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to initialize AnthropicAdapter: {e}")
        adapter = None

# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log full details server-side; return a generic message to the client
    # to avoid leaking internal schema structure.
    logger.warning(f"Request validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"error": {"type": "invalid_request_error", "message": "Request validation failed"}},
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Value error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"error": {"type": "invalid_request_error", "message": "Invalid request value"}},
    )

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal Error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_error", "message": "An internal error occurred"}},
    )

@app.exception_handler(HTTPException)
async def canonical_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.detail, "message": str(exc.detail)}},
    )

# Middlewares
def _rate_limit_key(request: Request) -> str:
    """
    Derive a rate-limit bucket key from the request.

    Priority: JWT session_id/tenant_id > client IP.
    Falls back to IP when JWT is absent, malformed, or MAESTRO_JWT_SECRET is unset.
    """
    key = request.client.host if request.client else "unknown"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer ") and MAESTRO_JWT_SECRET:
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, MAESTRO_JWT_SECRET, algorithms=[ALGORITHM])
            tenant_id = payload.get("tenant_id")
            session_id = payload.get("session_id")
            if tenant_id == "public" and session_id:
                key = f"session:{session_id}"
            elif tenant_id:
                key = f"tenant:{tenant_id}"
        except jwt.PyJWTError:
            pass  # Fallback to IP — auth dependency handles proper 401
    return key


# Paths subject to IP/session rate limiting (in addition to chat completions).
_RATE_LIMITED_PREFIXES = (
    "/v1/chat/completions",
    "/v1/prism/runs",
    "/v1/prism/status",
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if (
        RUNTIME_SETTINGS.launch_mode == LaunchMode.PUBLIC_READONLY
        and request.method == "POST"
        and "/v1/chat/completions" in request.url.path
    ):
        return launch_readonly_json_response(RUNTIME_SETTINGS)

    path = request.url.path
    if any(path.startswith(prefix) for prefix in _RATE_LIMITED_PREFIXES):
        key = _rate_limit_key(request)
        if not rate_limiter.is_allowed(key):
            retry_after = rate_limiter.get_retry_after(key)
            return JSONResponse(
                status_code=429,
                content={"error": {"type": "rate_limited", "message": "Too many requests"}},
                headers={"Retry-After": str(retry_after)}
            )

    return await call_next(request)

@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    if (
        RUNTIME_SETTINGS.launch_mode == LaunchMode.PUBLIC_READONLY
        and request.method == "POST"
        and "/v1/chat/completions" in request.url.path
    ):
        return launch_readonly_json_response(RUNTIME_SETTINGS)

    if request.method == "POST" and "/v1/chat/completions" in request.url.path:
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > 200000:
             return JSONResponse(
                status_code=413,
                content={"error": {"type": "request_too_large", "message": "Request exceeds maximum size"}}
            )
    return await call_next(request)

@app.middleware("http")
async def add_trace_headers(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-maestro-request-id"] = request_id
    response.headers["x-maestro-cache"] = "miss"
    response.headers["x-maestro-route"] = "tier1"
    response.headers["x-maestro-policy-version"] = "1.0.0"
    response.headers["x-maestro-template-version"] = "1.0.0"
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response

# Session Endpoints — anonymous public human verification

@app.get("/v1/session/challenge", response_model=ChallengeResponse)
def session_challenge():
    """
    Issue an Altcha proof-of-work challenge.

    The client widget (@altcha-org/altcha) fetches this endpoint automatically
    when configured with challengeurl="/v1/session/challenge". The widget solves
    the PoW in the browser (~0.5s) and stores the base64 solution payload which
    the frontend then POSTs to /v1/session/verify.
    """
    try:
        return generate_challenge()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/session/verify", response_model=SessionResponse)
def session_verify(body: VerifyRequest, request: Request):
    """
    Verify an Altcha PoW solution and issue an anonymous session JWT.

    The returned token is used as a Bearer token for /v1/chat/completions.
    It carries human_verified=true and a unique session_id for per-session
    rate limiting. Expires in SESSION_EXPIRY_SECONDS (default 2h).

    Rate-limited per IP: SESSION_VERIFY_RPM requests per 10 minutes (default 10)
    to prevent PoW farming / session-token harvesting.
    """
    client_ip = _get_client_ip(request)
    if not _session_verify_limiter.is_allowed(f"verify:{client_ip}"):
        retry_after = _session_verify_limiter.get_retry_after(f"verify:{client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": {"type": "rate_limited", "message": "Too many session verification attempts"}},
            headers={"Retry-After": str(retry_after)},
        )

    ok, error = verify_solution(body.payload)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Verification failed: {error}")

    if not MAESTRO_JWT_SECRET:
        raise HTTPException(status_code=503, detail="Session signing not configured")

    session_id = str(uuid.uuid4())
    now = int(time.time())
    claims = {
        "sub":            session_id,
        "tenant_id":      "public",
        "session_id":     session_id,
        "human_verified": True,
        "iat":            now,
        "exp":            now + SESSION_EXPIRY_SECONDS,
    }
    token = jwt.encode(claims, MAESTRO_JWT_SECRET, algorithm=ALGORITHM)
    logger.info("session_issued", extra={"session_id": session_id, "client_ip": client_ip})
    return {"token": token, "expires_in": SESSION_EXPIRY_SECONDS}


@app.get("/v1/session/dev-token", response_model=SessionResponse)
def session_dev_token():
    """
    Issue a signed session JWT without PoW verification.

    ONLY available when MAESTRO_LAUNCH_MODE=test. Returns 403 in all other
    modes.  Used by the VITE_MOCK_HUMANCHECK=true frontend dev bypass so
    end-to-end interview flow can be tested without solving an Altcha puzzle.
    """
    if RUNTIME_SETTINGS.launch_mode != LaunchMode.TEST:
        raise HTTPException(status_code=403, detail="dev-token only available in test mode")
    if not MAESTRO_JWT_SECRET:
        raise HTTPException(status_code=503, detail="Session signing not configured")

    session_id = str(uuid.uuid4())
    now = int(time.time())
    claims = {
        "sub":            session_id,
        "tenant_id":      "public",
        "session_id":     session_id,
        "human_verified": True,
        "dev_token":      True,
        "iat":            now,
        "exp":            now + SESSION_EXPIRY_SECONDS,
    }
    token = jwt.encode(claims, MAESTRO_JWT_SECRET, algorithm=ALGORITHM)
    logger.info("dev_token_issued", extra={"session_id": session_id})
    return {"token": token, "expires_in": SESSION_EXPIRY_SECONDS}


# Health Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.get("/health/live", response_model=LiveResponse)
def health_live():
    return {"alive": True}

@app.get("/health/ready", response_model=ReadyResponse)
async def health_ready():
    if RUNTIME_SETTINGS.launch_mode == LaunchMode.PUBLIC_READONLY:
        return {
            "ready": True,
            "adapter": "disabled",
        }

    status = {
        "ready": True,
        "adapter": "initialized" if adapter is not None else "missing"
    }

    if adapter is None:
        status["ready"] = False
        return status

    try:
        from libs.adapters.anthropic_adapter import AnthropicAdapter
        if isinstance(adapter, AnthropicAdapter):
            try:
                await asyncio.wait_for(
                    adapter.client.messages.count_tokens(
                        model="claude-3-5-haiku-20241022",
                        messages=[{"role": "user", "content": "hi"}]
                    ),
                    timeout=3.0
                )
                status["anthropic"] = "ok"
            except Exception as e:
                logger.error(f"Anthropic health check failed: {e}")
                status["anthropic"] = "unreachable"
                status["ready"] = False
    except ImportError:
        pass

    return status


@app.get("/v1/status", tags=["health"])
async def gateway_status():
    """
    Extended gateway status report.

    Returns adapter type, loaded-model info, optional VRAM usage (nvidia-smi),
    launch mode, rate-limit config, and uptime.  Does not call any external
    service; purely introspective and always fast.

    Unlike /health/ready this endpoint never returns a non-200 code — it is
    informational only and safe to poll from dashboards.
    """
    _adapter_name = "disabled"
    _adapter_ok   = False
    _model_info: dict = {}

    if adapter is not None:
        if isinstance(adapter, MockAdapter):
            _adapter_name = "mock"
            _adapter_ok   = True
        else:
            try:
                from libs.adapters.anthropic_adapter import AnthropicAdapter
                if isinstance(adapter, AnthropicAdapter):
                    _adapter_name = "anthropic"
                    _adapter_ok   = True
            except ImportError:
                pass
            if not _adapter_ok:
                try:
                    from libs.adapters.local_model_adapter import LocalModelAdapter
                    if isinstance(adapter, LocalModelAdapter):
                        _adapter_name = "local_model"
                        _adapter_ok   = True
                        _model_info   = {
                            "path":        Path(adapter.model_path).name,
                            "loaded":      adapter._model is not None,
                            "max_new_tokens": adapter.max_new_tokens,
                            "temperature": adapter.temperature,
                            "constrained": adapter.constrained,
                        }
                except ImportError:
                    pass

    # Optional VRAM report via nvidia-smi (graceful: skip if unavailable)
    _vram: dict = {}
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                _vram = {
                    "used_mb":  int(parts[0]),
                    "free_mb":  int(parts[1]),
                    "total_mb": int(parts[2]),
                }
    except Exception:
        pass  # nvidia-smi not available; not an error

    consent_gate_enabled = os.environ.get("CONSENT_GATE_ENABLED", "false").lower() == "true"

    return {
        "gateway":              "ok",
        "launch_mode":          RUNTIME_SETTINGS.launch_mode.value,
        "adapter":              _adapter_name,
        "adapter_ok":           _adapter_ok,
        "model":                _model_info,
        "rate_limit_rpm":       RATE_LIMIT_RPM,
        "session_verify_rpm":   _SESSION_VERIFY_RPM,
        "consent_gate_enabled": consent_gate_enabled,
        "sessions":             session_store.stats(),
        "vram":                 _vram,
        "uptime_seconds":       round(time.time() - _GATEWAY_START_TIME, 1),
        "timestamp":            datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────

def record_audit_log(
    request_id: str,
    tenant_id: str,
    action: str,
    metadata: dict = None,
):
    logger.info(
        "audit",
        extra={
            "action":     action,
            "request_id": request_id,
            "tenant_id":  tenant_id,
            **(metadata or {}),
        },
    )


def _touch_session(session_id: Optional[str], kernel_type: Optional[str], consented: bool) -> None:
    """Background task: update session store without blocking the response."""
    if session_id:
        session_store.touch(session_id, kernel_type=kernel_type, consented=consented)
        session_store.increment_turn(session_id)


# ── Consent gate wiring ───────────────────────────────────────────────────────
# When CONSENT_GATE_ENABLED=true, the chat endpoint also requires the caller's
# JWT to carry consent=true and kernel_type (set by POST /v1/session/consent).
# The dependency is built lazily so the import is always present even when the
# gate is disabled.

_CONSENT_GATE_ENABLED = os.environ.get("CONSENT_GATE_ENABLED", "false").lower() == "true"

if _CONSENT_GATE_ENABLED:
    from apps.gateway.consent import require_consent as _require_consent

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        auth_context: dict = Depends(verify_auth),
        consent_context: dict = Depends(_require_consent),
    ):
        if adapter is None:
            raise HTTPException(status_code=503, detail="Adapter not initialized")

        request_id  = request.state.request_id
        tenant_id   = auth_context["tenant_id"]
        session_id  = auth_context.get("user_id")   # user_id = session_id for public sessions
        kernel_type = consent_context.get("kernel_type")

        total_chars = sum(len(msg.content) for msg in body.messages)
        if total_chars > 100000:
            raise HTTPException(status_code=413, detail="Request exceeds maximum size")

        idempotency_key = request.headers.get("X-Idempotency-Key")
        audit_metadata  = {
            "idempotency_key": idempotency_key,
            "kernel_type":     kernel_type,
        }
        background_tasks.add_task(
            record_audit_log, request_id, str(tenant_id),
            "chat_completion_requested", audit_metadata,
        )
        background_tasks.add_task(
            _touch_session, session_id, kernel_type, True,
        )

        prepared_request = adapter.prepare(body)
        if body.stream:
            return StreamingResponse(
                adapter.stream(prepared_request),
                media_type="text/event-stream",
            )
        response = await adapter.invoke(prepared_request)
        response.maestro.request_id = request_id
        return response

else:
    @app.post("/v1/chat/completions")
    async def chat_completions(   # type: ignore[no-redef]
        body: ChatCompletionRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        auth_context: dict = Depends(verify_auth),
    ):
        if adapter is None:
            raise HTTPException(status_code=503, detail="Adapter not initialized")

        request_id = request.state.request_id
        tenant_id  = auth_context["tenant_id"]
        session_id = auth_context.get("user_id")

        total_chars = sum(len(msg.content) for msg in body.messages)
        if total_chars > 100000:
            raise HTTPException(status_code=413, detail="Request exceeds maximum size")

        idempotency_key = request.headers.get("X-Idempotency-Key")
        audit_metadata  = {"idempotency_key": idempotency_key} if idempotency_key else {}
        background_tasks.add_task(
            record_audit_log, request_id, str(tenant_id),
            "chat_completion_requested", audit_metadata,
        )
        background_tasks.add_task(
            _touch_session, session_id, None, False,
        )

        prepared_request = adapter.prepare(body)
        if body.stream:
            return StreamingResponse(
                adapter.stream(prepared_request),
                media_type="text/event-stream",
            )
        response = await adapter.invoke(prepared_request)
        response.maestro.request_id = request_id
        return response

