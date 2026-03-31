"""
captcha.py — Altcha-compatible proof-of-work human verification.

Implements the Altcha PoW algorithm in pure Python (stdlib only — hashlib,
hmac, secrets, time). No external dependencies.

Algorithm (compatible with @altcha-org/altcha frontend widget):
  1. Server picks a random integer N in [1, MAX_NUMBER]
  2. challenge = SHA-256(salt + str(N))        as hex
  3. signature = HMAC-SHA-256(challenge, KEY)   as hex
  4. Client iterates i=0..MAX_NUMBER until SHA-256(salt + str(i)) == challenge
  5. Client submits base64(JSON({algorithm, challenge, number=i, salt, signature}))
  6. Server verifies: recompute challenge from i, check HMAC, check expiry

The salt embeds an expiry timestamp as "?expire=<unix_ts>" so the server can
reject replayed or stale solutions without a nonce store.

Environment variables:
  CAPTCHA_HMAC_KEY       Required. Secret key for HMAC signature.
  CAPTCHA_MAX_NUMBER     Max PoW search range. Default 100000 (~0.5s on a browser).
  CAPTCHA_EXPIRY_SECONDS Challenge lifetime in seconds. Default 600 (10 min).
"""

import hashlib
import hmac as _hmac
import json
import os
import secrets
import time
from base64 import b64decode
from typing import Optional

CAPTCHA_HMAC_KEY: str = os.environ.get("CAPTCHA_HMAC_KEY", "")
MAX_NUMBER: int = int(os.environ.get("CAPTCHA_MAX_NUMBER", "100000"))
CHALLENGE_EXPIRY_SECONDS: int = int(os.environ.get("CAPTCHA_EXPIRY_SECONDS", "600"))


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _hmac_hex(data: str, key: str) -> str:
    return _hmac.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()


def _require_key() -> str:
    if not CAPTCHA_HMAC_KEY:
        raise RuntimeError("CAPTCHA_HMAC_KEY environment variable is not set.")
    return CAPTCHA_HMAC_KEY


def generate_challenge() -> dict:
    """
    Generate an Altcha PoW challenge dict to send directly to the client.

    Returns:
        {
            "algorithm": "SHA-256",
            "challenge": "<hex>",
            "salt":      "<hex>?expire=<unix_ts>",
            "signature": "<hex>",
            "maxnumber": <int>,
        }
    """
    key = _require_key()
    salt_raw = secrets.token_hex(12)
    expire_ts = int(time.time()) + CHALLENGE_EXPIRY_SECONDS
    salt = f"{salt_raw}?expire={expire_ts}"

    secret_number = secrets.randbelow(MAX_NUMBER) + 1
    challenge = _sha256_hex(salt + str(secret_number))
    signature = _hmac_hex(challenge, key)

    return {
        "algorithm": "SHA-256",
        "challenge": challenge,
        "salt":      salt,
        "signature": signature,
        "maxnumber": MAX_NUMBER,
    }


def verify_solution(payload_b64: str) -> tuple[bool, Optional[str]]:
    """
    Verify an Altcha solution payload (base64-encoded JSON from the widget).

    Args:
        payload_b64: The value of the altcha hidden form field or event payload.

    Returns:
        (True, None) on success.
        (False, "<reason>") on failure — never raises.
    """
    key = _require_key()

    try:
        raw = b64decode(payload_b64 + "==")        # padding-tolerant
        payload = json.loads(raw)
    except Exception:
        return False, "Invalid payload encoding"

    algorithm = payload.get("algorithm")
    challenge  = payload.get("challenge")
    number     = payload.get("number")
    salt       = payload.get("salt")
    signature  = payload.get("signature")

    if not all([algorithm, challenge, number is not None, salt, signature]):
        return False, "Missing required fields in payload"

    if algorithm != "SHA-256":
        return False, f"Unsupported algorithm: {algorithm!r}"

    # Enforce expiry embedded in salt ("...?expire=<ts>")
    if "?expire=" in salt:
        try:
            expire_ts = int(salt.split("?expire=")[1].split("&")[0])
            if time.time() > expire_ts:
                return False, "Challenge expired"
        except (ValueError, IndexError):
            return False, "Malformed expiry in salt"

    # Recompute challenge from the submitted number
    expected_challenge = _sha256_hex(salt + str(number))
    if not secrets.compare_digest(expected_challenge, challenge):
        return False, "PoW solution does not match challenge"

    # Verify HMAC — proves the challenge was issued by this server
    expected_sig = _hmac_hex(challenge, key)
    if not secrets.compare_digest(expected_sig, signature):
        return False, "HMAC signature invalid"

    return True, None
