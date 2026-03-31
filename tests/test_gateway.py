import os
import sys
import pytest
from fastapi.testclient import TestClient
import jwt

# Set environment variables BEFORE importing the app
os.environ["MAESTRO_JWT_SECRET"] = "test-secret-key-for-testing"
os.environ["USE_MOCK_ADAPTER"] = "true"
os.environ["RATE_LIMIT_RPM"] = "20" # High enough for other tests, low enough to hammer
os.environ["MAESTRO_LAUNCH_MODE"] = "private_full"

# TODO: remove after editable install — see pyproject.toml
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.gateway.main import app

client = TestClient(app)

SECRET = "test-secret-key-for-testing"
ALGORITHM = "HS256"

def generate_token(user_id="user_123", tenant_id="tenant_456"):
    payload = {"sub": user_id, "tenant_id": tenant_id}
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_completion_no_auth():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code in (401, 403)

def test_chat_completion_invalid_token():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}]
    }
    headers = {"Authorization": "Bearer invalid-token-here"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "Could not validate credentials (invalid JWT)"

def test_chat_completion_success():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}]
    }
    token = generate_token(tenant_id="tenant_success")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "maestro-default"
    assert "choices" in data
    assert "x-maestro-request-id" in response.headers

def test_chat_completion_empty_messages():
    payload = {
        "model": "maestro-default",
        "messages": []
    }
    token = generate_token(tenant_id="tenant_empty")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 422
    assert "error" in response.json()

def test_chat_completion_invalid_role():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "invalid", "content": "hello"}]
    }
    token = generate_token(tenant_id="tenant_invalid_role")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 422
    assert "error" in response.json()

def test_chat_completion_stream():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }
    token = generate_token(tenant_id="tenant_stream")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text

def test_request_too_large():
    long_content = "a" * 100001
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": long_content}]
    }
    token = generate_token(tenant_id="tenant_large")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 413
    assert "Request exceeds maximum size" in response.json()["error"]["message"]

def test_idempotency_key_captured():
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}]
    }
    token = generate_token(tenant_id="tenant_idempotency")
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Idempotency-Key": "test-key-123"
    }
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200

def test_rate_limiting():
    # Rate limit is set to 20 per minute
    payload = {
        "model": "maestro-default",
        "messages": [{"role": "user", "content": "hello"}]
    }
    token = generate_token(user_id="rate_limit_user", tenant_id="tenant_to_hammer")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Send 20 requests
    for i in range(20):
        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
    
    # Request 21: Rate Limited
    resp21 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp21.status_code == 429
    assert resp21.json()["error"]["type"] == "rate_limited"
    assert "Retry-After" in resp21.headers
