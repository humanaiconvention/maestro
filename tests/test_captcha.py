"""
test_captcha.py — Unit tests for the Altcha PoW captcha module.

Tests: challenge generation, correct solution verification, rejection of
wrong solutions, expired challenges, and missing HMAC key.
"""

import hashlib
import os
import time
import pytest

os.environ["CAPTCHA_HMAC_KEY"] = "test-secret-key-for-unit-tests"
os.environ["CAPTCHA_MAX_NUMBER"] = "1000"   # small range for fast tests
os.environ["CAPTCHA_EXPIRY_SECONDS"] = "60"

from apps.gateway.captcha import generate_challenge, verify_solution
from base64 import b64encode
import json


def _solve(challenge_dict: dict) -> str:
    """Brute-force the PoW locally and return base64-encoded solution payload."""
    salt = challenge_dict["salt"]
    target = challenge_dict["challenge"]
    maxn = challenge_dict["maxnumber"]

    for i in range(maxn + 1):
        h = hashlib.sha256((salt + str(i)).encode()).hexdigest()
        if h == target:
            payload = {
                "algorithm": challenge_dict["algorithm"],
                "challenge": challenge_dict["challenge"],
                "number":    i,
                "salt":      salt,
                "signature": challenge_dict["signature"],
            }
            return b64encode(json.dumps(payload).encode()).decode()

    raise AssertionError("Could not solve challenge — test range too small?")


def test_generate_challenge_has_required_fields():
    ch = generate_challenge()
    for field in ("algorithm", "challenge", "salt", "signature", "maxnumber"):
        assert field in ch, f"Missing field: {field}"
    assert ch["algorithm"] == "SHA-256"
    assert "?expire=" in ch["salt"]


def test_correct_solution_verifies():
    ch = generate_challenge()
    payload_b64 = _solve(ch)
    ok, err = verify_solution(payload_b64)
    assert ok, f"Expected OK but got error: {err}"
    assert err is None


def test_wrong_number_rejected():
    ch = generate_challenge()
    payload = {
        "algorithm": ch["algorithm"],
        "challenge": ch["challenge"],
        "number":    0,   # wrong
        "salt":      ch["salt"],
        "signature": ch["signature"],
    }
    b64 = b64encode(json.dumps(payload).encode()).decode()
    ok, err = verify_solution(b64)
    assert not ok
    assert "PoW" in err or "solution" in err.lower()


def test_tampered_signature_rejected():
    ch = generate_challenge()
    payload_b64 = _solve(ch)
    import base64
    raw = json.loads(base64.b64decode(payload_b64 + "=="))
    raw["signature"] = "0" * 64   # tampered
    b64 = b64encode(json.dumps(raw).encode()).decode()
    ok, err = verify_solution(b64)
    assert not ok
    assert "HMAC" in err or "signature" in err.lower()


def test_expired_challenge_rejected():
    ch = generate_challenge()
    # Manually replace expiry with a past timestamp
    old_salt = ch["salt"]
    past = int(time.time()) - 10
    new_salt = old_salt.split("?expire=")[0] + f"?expire={past}"
    # Recompute challenge+signature with new salt to get a technically valid-but-expired payload
    import hashlib, hmac as _hmac
    secret = os.environ["CAPTCHA_HMAC_KEY"]
    for i in range(int(os.environ["CAPTCHA_MAX_NUMBER"]) + 1):
        h = hashlib.sha256((new_salt + str(i)).encode()).hexdigest()
        if h == ch["challenge"]:
            # Found matching number — but salt has changed so we need to recompute
            break
    # Just test with a forged expiry — the expiry check runs before the PoW check
    payload = {
        "algorithm": "SHA-256",
        "challenge": ch["challenge"],
        "number":    0,
        "salt":      new_salt,
        "signature": ch["signature"],
    }
    b64 = b64encode(json.dumps(payload).encode()).decode()
    ok, err = verify_solution(b64)
    assert not ok
    assert "expired" in err.lower()


def test_invalid_base64_rejected():
    ok, err = verify_solution("not-valid-base64!!!")
    assert not ok
    assert "encoding" in err.lower() or "invalid" in err.lower()


def test_missing_hmac_key_raises():
    original = os.environ.pop("CAPTCHA_HMAC_KEY")
    try:
        import importlib
        import apps.gateway.captcha as cap
        importlib.reload(cap)
        with pytest.raises(RuntimeError, match="CAPTCHA_HMAC_KEY"):
            cap.generate_challenge()
    finally:
        os.environ["CAPTCHA_HMAC_KEY"] = original
        import importlib
        import apps.gateway.captcha as cap
        importlib.reload(cap)
