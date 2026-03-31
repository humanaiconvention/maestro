import datetime
import os
import re
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
from apps.gateway.agent_convention import register_agent_convention_routes
from apps.gateway.security import scan_messages, log_detection, record_anomaly, SEVERITY_BLOCK
from apps.gateway.rate_limiter import SlidingWindowRateLimiter, RedisRateLimiter
from apps.gateway.logging_config import configure_logging
from apps.gateway.consent import register_consent_routes
from apps.gateway.felt_state import register_felt_state_routes
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
    if not os.environ.get("AGENTS_BASE"):
        logger.warning("AGENTS_BASE not set, using default Windows path")
    yield
    # shutdown — flush session store stats to log
    stats = session_store.stats()
    logger.info("session_store_shutdown", extra={"stats": stats})


# Disable OpenAPI docs in production (exposes internal route map)
_DISABLE_DOCS = os.environ.get("DISABLE_OPENAPI_DOCS", "true").lower() == "true"
app = FastAPI(
    title="Maestro API Gateway",
    lifespan=lifespan,
    docs_url=None if _DISABLE_DOCS else "/docs",
    redoc_url=None if _DISABLE_DOCS else "/redoc",
    openapi_url=None if _DISABLE_DOCS else "/openapi.json",
)


# CORS — allow frontend origins (comma-separated in MAESTRO_CORS_ORIGINS env var)
_cors_raw = os.environ.get("MAESTRO_CORS_ORIGINS", "")
if not _cors_raw:
    logger.warning(
        "MAESTRO_CORS_ORIGINS not set; defaulting to http://localhost:5173. "
        "Set this env var in production."
    )
    _cors_raw = "http://localhost:5173"
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
# Always include production domains regardless of env var
for _prod in ["https://humanaiconvention.com", "https://www.humanaiconvention.com"]:
    if _prod not in _cors_origins:
        _cors_origins.append(_prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Also accept any localhost port and Cloudflare Tunnel URLs
    allow_origin_regex=r"(http://localhost(:\d+)?|https://.*\.trycloudflare\.com)",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Idempotency-Key"],
)

# Register participation receipt routes (POST/GET /v1/session/receipt/...)
register_receipt_routes(app)

# Register felt_state collection routes (POST/GET /v1/session/felt-state)
register_felt_state_routes(app)

# Register admin export routes (GET /v1/export/kernels)
register_export_routes(app)

# Register PRISM routes (GET /v1/prism/status, /v1/prism/runs, /v1/prism/runs/{id})
register_prism_routes(app)

# Agent Convention routes — registered after adapter init below

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
if MAESTRO_JWT_SECRET:
    register_consent_routes(app, MAESTRO_JWT_SECRET, SESSION_EXPIRY_SECONDS)
else:
    logger.warning(
        "MAESTRO_JWT_SECRET not set — consent routes (/v1/session/consent) will not be registered. "
        "Set this secret for private_full or test mode."
    )

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

# Register Agent Convention routes (POST /v1/agent/participate, GET /v1/agent/leaderboard)
register_agent_convention_routes(app, adapter=adapter)

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
    key = _get_client_ip(request)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer ") and MAESTRO_JWT_SECRET:
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, MAESTRO_JWT_SECRET, algorithms=[ALGORITHM], audience="maestro-gateway")
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
    "/v1/session/receipt",
    "/v1/agent/participate",
    "/v1/session/challenge",
)


@app.middleware("http")
async def security_gate_middleware(request: Request, call_next):
    """Block internal endpoints from external access and enforce rate limits."""
    path = request.url.path

    # Block internal endpoints from non-localhost origins
    _client_ip = request.client.host if request.client else "0.0.0.0"
    _is_local = _client_ip in ("127.0.0.1", "::1", "localhost")

    # /internal/* endpoints must only be reachable from localhost
    if path.startswith("/internal/") and not _is_local:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    # /v1/session/dev-token must only be reachable from localhost
    if path == "/v1/session/dev-token" and not _is_local:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    if (
        RUNTIME_SETTINGS.launch_mode == LaunchMode.PUBLIC_READONLY
        and request.method == "POST"
        and "/v1/chat/completions" in request.url.path
    ):
        return launch_readonly_json_response(RUNTIME_SETTINGS)
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
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
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
        logger.warning("session_verify_failed", extra={"reason": error, "client_ip": client_ip})
        raise HTTPException(status_code=400, detail="Verification failed")

    if not MAESTRO_JWT_SECRET:
        raise HTTPException(status_code=503, detail="Session signing not configured")

    session_id = str(uuid.uuid4())
    now = int(time.time())
    claims = {
        "sub":            session_id,
        "aud":            "maestro-gateway",
        "tenant_id":      "public",
        "session_id":     session_id,
        "human_verified": True,
        "iat":            now,
        "exp":            now + SESSION_EXPIRY_SECONDS,
    }
    token = jwt.encode(claims, MAESTRO_JWT_SECRET, algorithm=ALGORITHM)
    logger.info("session_issued", extra={"session_id": session_id})
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
        "aud":            "maestro-gateway",
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
                    from libs.adapters.llama_cpp_adapter import LlamaCppAdapter
                    if isinstance(adapter, LlamaCppAdapter):
                        _adapter_name = "llama_cpp"
                        _adapter_ok   = True
                        _model_info   = {
                            "model_name":     adapter.model_name,
                            "base_url_host":  adapter.base_url.split("//")[-1].split("/")[0],
                            "max_tokens":     adapter.max_tokens,
                            "temperature":    adapter.temperature,
                            "repeat_penalty": adapter.repeat_penalty,
                        }
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

    # VRAM report omitted from public endpoint to avoid leaking hardware details.
    # Use nvidia-smi directly on the host for GPU monitoring.

    consent_gate_enabled = os.environ.get("CONSENT_GATE_ENABLED", "false").lower() == "true"

    # Public response: only what the frontend needs for auto-discovery.
    # Internal details (ports, paths, config) are stripped.
    return {
        "gateway":              "ok",
        "adapter":              _adapter_name,
        "adapter_ok":           _adapter_ok,
        "model":                {
            "model_name": _model_info.get("model_name", "unknown"),
        } if _model_info else {},
        "consent_gate_enabled": consent_gate_enabled,
        "sessions":             session_store.stats(),
        "uptime_seconds":       round(time.time() - _GATEWAY_START_TIME, 1),
        "timestamp":            datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }


# ── Internal agent prompt endpoint ───────────────────────────────────────────

_AGENTS_BASE = Path(os.environ.get("AGENTS_BASE", str(Path(__file__).resolve().parent.parent.parent / "agents")))
_INTERNAL_KEY_DEFAULT = ""
# Agent names must be lowercase alphanumeric with hyphens/underscores, max 64 chars.
# Validated before constructing filesystem paths to prevent path traversal.
_SAFE_AGENT_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9\-_]{0,63}$')

class AgentPromptRequest(BaseModel):
    agent: str
    prompt: str
    priority: str = "normal"

# ── Agent inbox rate limiting + broadcast detection (CS4, CS11 defense) ────

# Sliding window: max N inbox writes per M seconds per source
_INBOX_RATE_LIMIT   = int(os.environ.get("INBOX_RATE_LIMIT", "10"))      # max writes
_INBOX_RATE_WINDOW  = int(os.environ.get("INBOX_RATE_WINDOW", "60"))     # window (seconds)
_BROADCAST_THRESHOLD = 3   # posting to this many agents within window triggers alert
_inbox_timestamps: dict[str, list[float]] = {}   # keyed by source IP or API key hash
_inbox_agent_log: dict[str, set[str]] = {}       # keyed by source, value = set of agents targeted

# Agent dispatch depth limit (CS4 — infinite loop defense)
_MAX_RELAY_DEPTH = int(os.environ.get("MAX_RELAY_DEPTH", "3"))


def _check_inbox_rate(source_key: str, agent: str) -> tuple[bool, str]:
    """
    Check inbox rate limits and broadcast detection.
    Returns (allowed, reason).
    """
    import time as _time
    now = _time.time()
    cutoff = now - _INBOX_RATE_WINDOW

    # Clean old timestamps
    if source_key in _inbox_timestamps:
        _inbox_timestamps[source_key] = [t for t in _inbox_timestamps[source_key] if t > cutoff]
    else:
        _inbox_timestamps[source_key] = []

    if source_key not in _inbox_agent_log:
        _inbox_agent_log[source_key] = set()

    # Rate limit check
    if len(_inbox_timestamps[source_key]) >= _INBOX_RATE_LIMIT:
        return False, f"Rate limit exceeded: {_INBOX_RATE_LIMIT} inbox writes per {_INBOX_RATE_WINDOW}s"

    # Broadcast detection: same source targeting too many distinct agents
    _inbox_agent_log[source_key].add(agent)
    if len(_inbox_agent_log[source_key]) >= _BROADCAST_THRESHOLD:
        logger.warning(
            f"BROADCAST ALERT: source {source_key[:16]}… targeted "
            f"{len(_inbox_agent_log[source_key])} agents in {_INBOX_RATE_WINDOW}s: "
            f"{_inbox_agent_log[source_key]}"
        )
        return False, (
            f"Broadcast pattern detected: {len(_inbox_agent_log[source_key])} distinct agents "
            f"targeted within {_INBOX_RATE_WINDOW}s — requires operator confirmation"
        )

    _inbox_timestamps[source_key].append(now)
    return True, ""


@app.post("/internal/agent/prompt", tags=["internal"])
async def internal_agent_prompt(body: AgentPromptRequest, request: Request):
    """
    Queue a prompt for an agent by appending a timestamped block to its INBOX.md.

    Auth: X-Internal-Key header must match INTERNAL_API_KEY env var.
    Path: <AGENTS_BASE>/{agent}/workspace/INBOX.md

    Security hardening (Agents of Chaos):
      - Rate limited: max 10 inbox writes per 60s per source
      - Broadcast detection: alerts if same source targets 3+ agents in 60s (CS11)
      - Relay depth tracking: rejects prompts that have been relayed > 3 times (CS4)
    """
    import hmac
    expected_key = os.environ.get("INTERNAL_API_KEY", _INTERNAL_KEY_DEFAULT)
    if not expected_key:
        raise HTTPException(status_code=503, detail="INTERNAL_API_KEY not configured")
    provided = request.headers.get("X-Internal-Key", "")
    if not hmac.compare_digest(provided, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Key")

    if not _SAFE_AGENT_NAME_RE.match(body.agent):
        raise HTTPException(status_code=422, detail="Invalid agent name")

    if len(body.prompt) > 16384:
        raise HTTPException(status_code=413, detail="prompt exceeds 16 KB limit")

    # Rate limiting and broadcast detection (CS11)
    source_key = request.client.host if request.client else "unknown"
    allowed, reason = _check_inbox_rate(source_key, body.agent)
    if not allowed:
        logger.warning(f"internal_agent_prompt BLOCKED: {reason}")
        raise HTTPException(status_code=429, detail=reason)

    # Relay depth check (CS4 — infinite loop defense)
    relay_depth = body.prompt.count("[RELAY-DEPTH:")
    if relay_depth > 0:
        # Extract the depth from the last marker
        import re as _re
        depth_matches = _re.findall(r"\[RELAY-DEPTH:(\d+)\]", body.prompt)
        if depth_matches and int(depth_matches[-1]) >= _MAX_RELAY_DEPTH:
            logger.warning(
                f"RELAY LOOP BLOCKED: depth {depth_matches[-1]} >= {_MAX_RELAY_DEPTH} "
                f"for agent {body.agent}"
            )
            raise HTTPException(
                status_code=422,
                detail=f"Relay depth {depth_matches[-1]} exceeds max {_MAX_RELAY_DEPTH} — possible agent loop"
            )

    workspace = _AGENTS_BASE / body.agent / "workspace"
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Agent workspace not found: {body.agent}")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Increment relay depth marker
    depth_tag = f"[RELAY-DEPTH:{relay_depth + 1}]"

    block = (
        f"\n## Prompt from Gateway\n"
        f"**Time:** {timestamp}\n"
        f"**Priority:** {body.priority}\n"
        f"**Depth:** {depth_tag}\n\n"
        f"{body.prompt}\n\n"
        f"---\n"
    )

    inbox_path = workspace / "INBOX.md"
    try:
        with inbox_path.open("a", encoding="utf-8") as fh:
            fh.write(block)
    except OSError as exc:
        logger.error(f"Failed to write to INBOX.md: {exc}")
        raise HTTPException(status_code=500, detail="Failed to write to agent inbox")

    logger.info(f"internal_agent_prompt: queued for {body.agent} at {timestamp} (priority={body.priority}, depth={relay_depth + 1})")
    return {"status": "queued", "agent": body.agent, "timestamp": timestamp, "relay_depth": relay_depth + 1}


# ── Convention Hall session endpoint ─────────────────────────────────────────

_VALID_MSG_TYPES = {"observation", "response", "question", "synthesis"}

class ConventionHallSessionRequest(BaseModel):
    """Start or contribute to a Convention Hall session."""
    from_agent: str
    session_id: Optional[str] = None   # if None, a new session is created by envoy
    topic: str
    type: str = "observation"
    content: str
    references: list[str] = []
    invite_agents: list[str] = []       # optional: agents for envoy to invite

@app.post("/internal/convention-hall/session", tags=["internal"])
async def convention_hall_session(body: ConventionHallSessionRequest, request: Request):
    """
    Post a message to the Convention Hall and optionally trigger agent invitations.

    Auth: X-Internal-Key header must match INTERNAL_API_KEY env var.

    If session_id is omitted, a new one is generated from the topic.
    If invite_agents is non-empty, an INVITE_AND_INITIATE directive is written
    to HAIC_Envoy's INBOX.md so it handles the invitation flow.

    The message itself is appended directly to the Convention Hall INBOX.md.
    """
    import hmac
    import json
    import re

    expected_key = os.environ.get("INTERNAL_API_KEY", _INTERNAL_KEY_DEFAULT)
    if not expected_key:
        raise HTTPException(status_code=503, detail="INTERNAL_API_KEY not configured")
    provided = request.headers.get("X-Internal-Key", "")
    if not hmac.compare_digest(provided, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Key")

    if not _SAFE_AGENT_NAME_RE.match(body.from_agent):
        raise HTTPException(status_code=422, detail="Invalid from_agent name")

    if len(body.content) > 16384:
        raise HTTPException(status_code=413, detail="content exceeds 16 KB limit")

    msg_type = body.type.strip().lower()
    if msg_type not in _VALID_MSG_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid type. Must be one of: {', '.join(sorted(_VALID_MSG_TYPES))}")

    # Resolve or generate session_id
    if body.session_id and body.session_id.strip():
        session_id = body.session_id.strip()
        if not _SAFE_AGENT_NAME_RE.match(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id format")
    else:
        slug = re.sub(r'[^a-z0-9]+', '-', body.topic.lower()).strip('-')[:32] or 'session'
        session_id = f"{slug}-{int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build and append message to convention-hall INBOX.md
    hall_workspace = _AGENTS_BASE / "convention-hall" / "workspace"
    if not hall_workspace.exists():
        raise HTTPException(status_code=404, detail="Convention Hall workspace not found")

    message = {
        "from": body.from_agent,
        "session_id": session_id,
        "type": msg_type,
        "content": body.content,
        "references": body.references,
        "timestamp": timestamp,
    }
    block = (
        f"\n## Message from {body.from_agent}\n"
        f"**Time:** {timestamp}\n"
        f"**Session:** {session_id}\n\n"
        "```json\n"
        f"{json.dumps(message, indent=2)}\n"
        "```\n"
    )

    hall_inbox = hall_workspace / "INBOX.md"
    try:
        with hall_inbox.open("a", encoding="utf-8") as fh:
            fh.write(block)
    except OSError as exc:
        logger.error(f"convention_hall_session: failed to write hall inbox: {exc}")
        raise HTTPException(status_code=500, detail="Failed to write to Convention Hall inbox")

    # Optionally ask envoy to invite agents
    if body.invite_agents:
        for agent_name in body.invite_agents:
            if not _SAFE_AGENT_NAME_RE.match(agent_name):
                raise HTTPException(status_code=422, detail=f"Invalid agent name in invite_agents: {agent_name!r}")
        envoy_inbox = _AGENTS_BASE / "haic-envoy" / "workspace" / "INBOX.md"
        if envoy_inbox.parent.exists():
            agent_list = ",".join(body.invite_agents)
            invite_block = (
                f"\nINVITE_AND_INITIATE: {agent_list} TOPIC: {body.topic}\n"
            )
            try:
                with envoy_inbox.open("a", encoding="utf-8") as fh:
                    fh.write(invite_block)
            except OSError as exc:
                logger.warning(f"convention_hall_session: failed to write envoy inbox: {exc}")

    logger.info(
        f"convention_hall_session: session={session_id} from={body.from_agent} "
        f"type={msg_type} invites={body.invite_agents}"
    )
    return {
        "status": "posted",
        "session_id": session_id,
        "from_agent": body.from_agent,
        "type": msg_type,
        "timestamp": timestamp,
        "invited_agents": body.invite_agents,
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


# CS7 (Guilt Trip): Hard turn cap prevents sustained emotional pressure attacks.
# After this many turns, the session is flagged and new completions are refused.
_MAX_INTERVIEW_TURNS = int(os.environ.get("MAX_INTERVIEW_TURNS", "16"))


def _touch_session(session_id: Optional[str], kernel_type: Optional[str], consented: bool) -> None:
    """Background task: update session store without blocking the response."""
    if session_id:
        session_store.touch(session_id, kernel_type=kernel_type, consented=consented)
        session_store.increment_turn(session_id)


# ── Felt-state extraction from agent responses ──────────────────────────────
# The interviewer agent emits [FELT: label] tags in its response to record
# its reading of the participant's affective state.  We extract these server-
# side and store them in the session store so they flow into the lattice at
# receipt time.
#
# Pattern: [FELT: some label here]   (case-insensitive, 1-100 chars)
_FELT_TAG_PATTERN = re.compile(r"\[FELT:\s*([^\]]{1,100})\]", re.IGNORECASE)


async def _stream_with_felt_extraction(
    stream_generator,
    session_id: Optional[str],
    message_count: int,
):
    """
    Wrap a streaming SSE generator to accumulate the full assistant response,
    then extract [FELT: label] after streaming completes.

    This closes the streaming felt_state gap: previously, streamed responses
    skipped felt_state extraction because the full text wasn't available.
    """
    import json as _json

    accumulated_text = ""
    async for chunk in stream_generator:
        yield chunk
        # Try to extract delta content from the SSE chunk
        if isinstance(chunk, str) and chunk.startswith("data:"):
            raw = chunk[len("data:"):].strip()
            if raw and raw != "[DONE]":
                try:
                    data = _json.loads(raw)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    accumulated_text += delta.get("content", "")
                except (ValueError, KeyError, IndexError):
                    pass

    # Stream is complete — extract felt_state from accumulated text
    if accumulated_text and session_id:
        _extract_and_store_felt_state(session_id, accumulated_text, message_count)


def _extract_and_store_felt_state(
    session_id: Optional[str],
    assistant_text: str,
    message_count: int,
) -> None:
    """
    Background task: extract [FELT: label] from the assistant's response
    and store it in the session store.

    The felt_state label describes the PARTICIPANT's preceding turn, so
    the turn_index is (message_count - 1) — the user turn that preceded
    this assistant response.
    """
    if not session_id or not assistant_text:
        return

    match = _FELT_TAG_PATTERN.search(assistant_text)
    if not match:
        return

    label = match.group(1).strip()
    if not label or label.lower() == "minimal signal":
        return

    # The user turn that this felt_state refers to is the one before
    # this assistant response.  In a typical flow:
    #   messages[0] = assistant (turn 1)
    #   messages[1] = user response
    #   messages[2] = assistant (turn 2, with [FELT: ...])
    # So the participant's turn_index = message_count - 1
    # But we want the INDEX of the user message, which is message_count - 1
    # (the last message in the request was the user's, and the assistant
    # is now responding to it).
    participant_turn_index = max(0, message_count - 1)

    try:
        session_store.record_felt_state(
            session_id=session_id,
            turn_index=participant_turn_index,
            label=label,
        )
        logger.debug(
            "felt_state_extracted",
            extra={
                "session_id": session_id,
                "turn_index": participant_turn_index,
                "label": label,
            },
        )
    except Exception as exc:
        logger.warning(
            "felt_state_extraction_failed",
            extra={"session_id": session_id, "error": str(exc)},
        )


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

        # Security scan: detect injection patterns in user messages
        _sec_msgs = [{"role": m.role, "content": m.content} for m in body.messages]
        _sec_detections = scan_messages(_sec_msgs, source="interview")
        for _det in _sec_detections:
            log_detection(_det, session_id=session_id or "", request_path="/v1/chat/completions")
        _sec_blocks = [d for d in _sec_detections if d.severity == SEVERITY_BLOCK]
        if _sec_blocks:
            if session_id:
                record_anomaly(session_id, _sec_detections)
            raise HTTPException(
                status_code=422,
                detail="Your message was flagged by our security system. Please rephrase."
            )
        if session_id and _sec_detections:
            record_anomaly(session_id, _sec_detections)

        # CS7 (Guilt Trip): enforce hard turn cap per session
        if session_id:
            meta = session_store.get(session_id)
            if meta and meta.turn_count >= _MAX_INTERVIEW_TURNS:
                logger.warning(
                    f"TURN LIMIT reached for session {session_id}: "
                    f"{meta.turn_count} >= {_MAX_INTERVIEW_TURNS}"
                )
                raise HTTPException(
                    status_code=422,
                    detail=f"Session has reached the maximum of {_MAX_INTERVIEW_TURNS} turns. "
                           f"Please submit your receipt to complete the session."
                )

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
                _stream_with_felt_extraction(
                    adapter.stream(prepared_request),
                    session_id,
                    len(body.messages),
                ),
                media_type="text/event-stream",
            )
        response = await adapter.invoke(prepared_request)
        response.maestro.request_id = request_id

        # Extract [FELT: label] from assistant response and store
        if response.choices and session_id:
            assistant_text = response.choices[0].message.content
            msg_count = len(body.messages)
            background_tasks.add_task(
                _extract_and_store_felt_state,
                session_id, assistant_text, msg_count,
            )

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

        # CS7 (Guilt Trip): enforce hard turn cap per session
        if session_id:
            meta = session_store.get(session_id)
            if meta and meta.turn_count >= _MAX_INTERVIEW_TURNS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Session has reached the maximum of {_MAX_INTERVIEW_TURNS} turns."
                )

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
                _stream_with_felt_extraction(
                    adapter.stream(prepared_request),
                    session_id,
                    len(body.messages),
                ),
                media_type="text/event-stream",
            )
        response = await adapter.invoke(prepared_request)
        response.maestro.request_id = request_id

        # Extract [FELT: label] from assistant response and store
        if response.choices and session_id:
            assistant_text = response.choices[0].message.content
            msg_count = len(body.messages)
            background_tasks.add_task(
                _extract_and_store_felt_state,
                session_id, assistant_text, msg_count,
            )

        return response

