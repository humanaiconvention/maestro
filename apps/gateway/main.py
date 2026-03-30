import os
import uuid
import time
import logging
import traceback
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from jose import jwt, JWTError

from libs.schemas.api import ChatCompletionRequest, ChatCompletionResponse
from libs.adapters.mock import MockAdapter
from apps.gateway.auth import verify_auth, ALGORITHM
from apps.gateway.rate_limiter import SlidingWindowRateLimiter
from apps.gateway.prism import register_prism_routes

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Maestro API Gateway")

# Register PRISM routes (GET /v1/prism/status, /v1/prism/runs, /v1/prism/runs/{id})
register_prism_routes(app)

# Rate Limiter Initialization
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))
rate_limiter = SlidingWindowRateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60)

# JWT Secret for middleware-level extraction
MAESTRO_JWT_SECRET = os.environ.get("MAESTRO_JWT_SECRET")

# Health Models
class HealthResponse(BaseModel):
    status: str

class LiveResponse(BaseModel):
    alive: bool

class ReadyResponse(BaseModel):
    ready: bool
    adapter: str
    anthropic: Optional[str] = None

# Read USE_MOCK_ADAPTER=true/false from environment
USE_MOCK_ADAPTER_STR = os.environ.get("USE_MOCK_ADAPTER", "true").lower()
USE_MOCK_ADAPTER = USE_MOCK_ADAPTER_STR == "true"

adapter = None
if USE_MOCK_ADAPTER:
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
    return JSONResponse(
        status_code=422,
        content={"error": {"type": "invalid_request_error", "message": str(exc)}},
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": {"type": "invalid_request_error", "message": str(exc)}},
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
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "POST" and "/v1/chat/completions" in request.url.path:
        # Extract tenant_id from JWT or use IP
        key = request.client.host
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer ") and MAESTRO_JWT_SECRET:
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, MAESTRO_JWT_SECRET, algorithms=[ALGORITHM])
                tenant_id = payload.get("tenant_id")
                if tenant_id:
                    key = f"tenant:{tenant_id}"
            except JWTError:
                pass # Fallback to IP if token is invalid (auth dependency will catch it properly)

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

# Health Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.get("/health/live", response_model=LiveResponse)
def health_live():
    return {"alive": True}

@app.get("/health/ready", response_model=ReadyResponse)
async def health_ready():
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

def record_audit_log(request_id: str, tenant_id: str, action: str, metadata: dict = None):
    meta_str = f" metadata={metadata}" if metadata else ""
    print(f"Audit log: [{action}] req={request_id} tenant={tenant_id}{meta_str}")

@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    auth_context: dict = Depends(verify_auth)
):
    if adapter is None:
        raise HTTPException(status_code=503, detail="Adapter not initialized")

    request_id = request.state.request_id
    tenant_id = auth_context["tenant_id"]

    total_chars = sum(len(msg.content) for msg in body.messages)
    if total_chars > 100000:
        raise HTTPException(status_code=413, detail="Request exceeds maximum size")

    idempotency_key = request.headers.get("X-Idempotency-Key")
    audit_metadata = {"idempotency_key": idempotency_key} if idempotency_key else {}
    background_tasks.add_task(record_audit_log, request_id, str(tenant_id), "chat_completion_requested", audit_metadata)

    prepared_request = adapter.prepare(body)
    if body.stream:
        return StreamingResponse(
            adapter.stream(prepared_request),
            media_type="text/event-stream"
        )
    else:
        response = await adapter.invoke(prepared_request)
        response.maestro.request_id = request_id
        return response

