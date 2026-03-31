import pytest

from legacy_mvp.shared_runtime import (
    LaunchMode,
    get_runtime_settings,
    validate_gateway_runtime,
    validate_identity_runtime,
    validate_legacy_api_runtime,
)


def _clear_runtime_env(monkeypatch):
    for key in [
        "TESTING",
        "MAESTRO_LAUNCH_MODE",
        "MAESTRO_API_KEY",
        "MAESTRO_JWT_SECRET",
        "SESSION_SECRET",
        "HYDRA_ADMIN_URL",
        "AUTHENTIK_URL",
        "PG_PASS",
        "MAESTRO_ENABLE_DEMO_IDENTITY",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_launch_mode_is_public_readonly(monkeypatch):
    _clear_runtime_env(monkeypatch)
    settings = get_runtime_settings()
    assert settings.launch_mode == LaunchMode.PUBLIC_READONLY
    assert settings.testing is False


def test_validate_legacy_runtime_requires_api_key_in_private_mode(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("MAESTRO_LAUNCH_MODE", "private_full")

    with pytest.raises(RuntimeError, match="MAESTRO_API_KEY"):
        validate_legacy_api_runtime()


def test_validate_gateway_runtime_requires_jwt_secret_in_private_mode(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("MAESTRO_LAUNCH_MODE", "private_full")

    with pytest.raises(RuntimeError, match="MAESTRO_JWT_SECRET"):
        validate_gateway_runtime()


def test_validate_identity_runtime_requires_pg_pass_outside_tests(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("MAESTRO_LAUNCH_MODE", "public_readonly")

    with pytest.raises(RuntimeError, match="PG_PASS"):
        validate_identity_runtime()


def test_testing_mode_skips_secret_validation(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("TESTING", "true")

    assert validate_legacy_api_runtime().testing is True
    assert validate_gateway_runtime().testing is True
    assert validate_identity_runtime().testing is True
