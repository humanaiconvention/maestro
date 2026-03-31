"""
conftest.py — Shared pytest configuration for Maestro test suite.

Sets gateway environment variables BEFORE any test module is imported.
pytest loads conftest.py before collecting test files, so these values are
guaranteed to be in os.environ when apps.gateway.main is first imported —
regardless of which test file triggers the import.

If a test file overrides a variable with os.environ[...] = ... AFTER
conftest runs, the module-level constants in main.py (captured at import
time) will reflect the conftest values, NOT the per-file override.  Test
files that need consistent JWT verification should use the shared secret
defined here.

Important: test_gateway.py also sets these values via direct assignment.
The conftest sets them first so the module initializes correctly; the
per-file assignments then update os.environ (affecting dynamic reads like
auth.py's _get_jwt_secret) but do not change the already-captured
module-level constants in main.py.
"""

import os

# ── Core secrets ─────────────────────────────────────────────────────────────
os.environ.setdefault("MAESTRO_JWT_SECRET", "test-secret-key-for-testing")
os.environ.setdefault("CAPTCHA_HMAC_KEY",   "test-captcha-hmac-key")

# ── Adapter + mode ───────────────────────────────────────────────────────────
# Explicitly pin adapter flags so load_dotenv() (called in main.py at import
# time with override=False) does not overwrite them with .env values.
os.environ.setdefault("USE_LLAMACPP_ADAPTER", "false")
os.environ.setdefault("USE_LOCAL_ADAPTER",    "false")
os.environ.setdefault("USE_MOCK_ADAPTER",     "true")
os.environ.setdefault("MAESTRO_LAUNCH_MODE",  "private_full")

# ── Rate limiting ─────────────────────────────────────────────────────────────
os.environ.setdefault("RATE_LIMIT_RPM",     "20")
os.environ.setdefault("SESSION_VERIFY_RPM", "5")   # low ceiling for verify-rate tests

# ── Proxy headers (needed so rate-limit tests can inject distinct IPs) ────────
os.environ.setdefault("TRUST_PROXY_HEADERS", "true")
