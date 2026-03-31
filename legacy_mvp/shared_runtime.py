import os
from dataclasses import dataclass
from enum import Enum

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse


READONLY_STATUS_CODE = 503
READONLY_ERROR_CODE = "launch_readonly"
READONLY_MESSAGE = "This public launch is read-only. Auth and write flows are not yet enabled."
DEMO_DISABLED_ERROR_CODE = "demo_identity_disabled"
DEMO_DISABLED_MESSAGE = "Demo identity flows are disabled in this deployment."


class LaunchMode(str, Enum):
    PUBLIC_READONLY = "public_readonly"
    PRIVATE_FULL = "private_full"
    TEST = "test"


@dataclass(frozen=True)
class RuntimeSettings:
    launch_mode: LaunchMode
    testing: bool
    demo_identity_enabled: bool


DEFAULT_PRIVATE_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_runtime_settings() -> RuntimeSettings:
    testing = _env_flag("TESTING", False)
    raw_mode = os.getenv("MAESTRO_LAUNCH_MODE")

    if raw_mode:
        try:
            launch_mode = LaunchMode(raw_mode.strip().lower())
        except ValueError as exc:
            raise RuntimeError(
                "MAESTRO_LAUNCH_MODE must be one of: public_readonly, private_full, test."
            ) from exc
    elif testing:
        launch_mode = LaunchMode.TEST
    else:
        launch_mode = LaunchMode.PUBLIC_READONLY

    return RuntimeSettings(
        launch_mode=launch_mode,
        testing=testing,
        demo_identity_enabled=_env_flag("MAESTRO_ENABLE_DEMO_IDENTITY", False),
    )


def mode_allows_private_operations(settings: RuntimeSettings) -> bool:
    return settings.launch_mode in {LaunchMode.PRIVATE_FULL, LaunchMode.TEST}


def get_cors_origins(settings: RuntimeSettings | None = None) -> list[str]:
    resolved = settings or get_runtime_settings()
    raw = os.getenv("MAESTRO_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    if mode_allows_private_operations(resolved):
        return list(DEFAULT_PRIVATE_CORS_ORIGINS)
    return []


def validate_required_envs(service_name: str, env_names: list[str]) -> None:
    missing = [name for name in env_names if not os.getenv(name)]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise RuntimeError(
            f"{service_name} is missing required environment variables: {formatted}"
        )


def validate_legacy_api_runtime() -> RuntimeSettings:
    settings = get_runtime_settings()
    if settings.testing:
        return settings
    if settings.launch_mode == LaunchMode.PRIVATE_FULL:
        validate_required_envs("Maestro legacy API", ["MAESTRO_API_KEY"])
    return settings


def validate_gateway_runtime() -> RuntimeSettings:
    settings = get_runtime_settings()
    if settings.testing:
        return settings
    if settings.launch_mode == LaunchMode.PRIVATE_FULL:
        validate_required_envs("Maestro gateway", ["MAESTRO_JWT_SECRET"])
    return settings


def validate_identity_runtime() -> RuntimeSettings:
    settings = get_runtime_settings()
    if settings.testing:
        return settings

    required = ["PG_PASS"]
    if settings.launch_mode == LaunchMode.PRIVATE_FULL:
        required.extend(["SESSION_SECRET", "HYDRA_ADMIN_URL", "AUTHENTIK_URL"])

    validate_required_envs("Maestro identity service", required)
    return settings


def launch_readonly_payload(settings: RuntimeSettings | None = None) -> dict:
    resolved = settings or get_runtime_settings()
    return {
        "error": {
            "code": READONLY_ERROR_CODE,
            "message": READONLY_MESSAGE,
            "mode": resolved.launch_mode.value,
        }
    }


def launch_readonly_json_response(settings: RuntimeSettings | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=READONLY_STATUS_CODE,
        content=launch_readonly_payload(settings),
    )


def request_prefers_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return (
        "application/json" in accept
        or "application/json" in content_type
        or request.headers.get("x-requested-with") == "XMLHttpRequest"
    )


def _html_page(title: str, message: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(135deg, #f7f4ea, #e8ece2);
      color: #162018;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    main {{
      max-width: 720px;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(22, 32, 24, 0.14);
      border-radius: 18px;
      padding: 32px;
      box-shadow: 0 24px 60px rgba(22, 32, 24, 0.12);
    }}
    h1 {{
      margin-top: 0;
      font-size: clamp(2rem, 4vw, 2.8rem);
      line-height: 1.05;
    }}
    p {{
      font-size: 1.05rem;
      line-height: 1.6;
      margin-bottom: 0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p>{message}</p>
  </main>
</body>
</html>"""


def launch_readonly_html_response(settings: RuntimeSettings | None = None) -> HTMLResponse:
    return HTMLResponse(
        status_code=READONLY_STATUS_CODE,
        content=_html_page("HumanAI Convention Launch Mode", READONLY_MESSAGE),
    )


def demo_identity_disabled_response(request: Request) -> HTMLResponse | JSONResponse:
    payload = {
        "error": {
            "code": DEMO_DISABLED_ERROR_CODE,
            "message": DEMO_DISABLED_MESSAGE,
            "mode": get_runtime_settings().launch_mode.value,
        }
    }
    if request_prefers_json(request):
        return JSONResponse(status_code=READONLY_STATUS_CODE, content=payload)
    return HTMLResponse(
        status_code=READONLY_STATUS_CODE,
        content=_html_page("Demo Identity Disabled", DEMO_DISABLED_MESSAGE),
    )


def token_has_scope(token_data: dict, required_scope: str) -> bool:
    scope_value = token_data.get("scope", "")
    if isinstance(scope_value, str):
        scopes = set(scope_value.split())
    elif isinstance(scope_value, (list, tuple, set)):
        scopes = {str(value) for value in scope_value}
    else:
        scopes = set()
    return required_scope in scopes
