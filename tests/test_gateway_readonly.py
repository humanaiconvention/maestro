import importlib
import os
import sys

from fastapi.testclient import TestClient


def _load_gateway_app():
    os.environ.pop("MAESTRO_JWT_SECRET", None)
    os.environ["MAESTRO_LAUNCH_MODE"] = "public_readonly"
    os.environ["USE_MOCK_ADAPTER"] = "false"

    for name in ["apps.gateway.main"]:
        sys.modules.pop(name, None)

    module = importlib.import_module("apps.gateway.main")
    return module.app


def test_gateway_readonly_mode_disables_chat():
    app = _load_gateway_app()
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/health/live").status_code == 200

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "adapter": "disabled", "anthropic": None}

    response = client.post(
        "/v1/chat/completions",
        json={"model": "maestro-default", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "launch_readonly"
