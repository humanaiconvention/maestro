"""
test_rate_limiter.py — Unit tests for SlidingWindowRateLimiter.

Tests: under-limit passes, at-limit blocks, window slides (old entries
expire), get_retry_after, and cleanup of stale keys.
"""

import time
import pytest

from apps.gateway.rate_limiter import SlidingWindowRateLimiter


# ──────────────────────────────────────────────────────────────────────────────
# Under limit
# ──────────────────────────────────────────────────────────────────────────────

def test_first_request_allowed():
    rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    assert rl.is_allowed("key1") is True


def test_requests_under_limit_all_allowed():
    rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert rl.is_allowed("key1") is True


# ──────────────────────────────────────────────────────────────────────────────
# At / over limit
# ──────────────────────────────────────────────────────────────────────────────

def test_request_at_limit_blocked():
    rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        rl.is_allowed("k")
    assert rl.is_allowed("k") is False


def test_multiple_requests_over_limit_stay_blocked():
    rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    rl.is_allowed("k")
    rl.is_allowed("k")
    assert rl.is_allowed("k") is False
    assert rl.is_allowed("k") is False


def test_different_keys_are_independent():
    rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    # Exhaust key-A
    rl.is_allowed("key-A")
    rl.is_allowed("key-A")
    assert rl.is_allowed("key-A") is False
    # key-B is unaffected
    assert rl.is_allowed("key-B") is True


# ──────────────────────────────────────────────────────────────────────────────
# Sliding window — old entries expire
# ──────────────────────────────────────────────────────────────────────────────

def test_old_entries_expire_and_allow_new_requests():
    """
    Inject stale timestamps directly so the test doesn't sleep.

    Pre-populate the key with timestamps 65s ago (outside the 60s window).
    is_allowed filters them out → treated as 0 in-window requests → allowed.
    """
    rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    past = time.time() - 65
    rl.requests["k"] = [past, past, past]

    assert rl.is_allowed("k") is True  # stale entries pruned; slot free


def test_partially_expired_entries():
    """2 of 3 entries are stale; only 1 remains in-window → still allowed."""
    rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    past = time.time() - 65
    recent = time.time() - 5
    rl.requests["k"] = [past, past, recent]

    assert rl.is_allowed("k") is True  # 1 live + this one = 2 < 3


# ──────────────────────────────────────────────────────────────────────────────
# get_retry_after
# ──────────────────────────────────────────────────────────────────────────────

def test_retry_after_unknown_key_returns_zero():
    rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    assert rl.get_retry_after("nonexistent") == 0


def test_retry_after_returns_positive_when_blocked():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    rl.is_allowed("k")   # consumes the single slot
    rl.is_allowed("k")   # blocked
    wait = rl.get_retry_after("k")
    assert wait >= 1


def test_retry_after_is_at_most_window_seconds():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    rl.is_allowed("k")
    wait = rl.get_retry_after("k")
    assert wait <= 60


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────

def test_cleanup_removes_stale_keys():
    """
    Cleanup fires when len(requests) > 10000.  Stale keys (newest timestamp
    older than 2×window) are removed.

    Strategy: seed 10001 stale-* keys plus one "active" key with a recent
    timestamp. Call is_allowed("active") — it finds the existing key, goes
    through the full code path (not the early-return for new keys), and
    after the request count check triggers _cleanup.
    """
    rl = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    stale_ts = time.time() - 200  # older than 2×window (120s)

    for i in range(10001):
        rl.requests[f"stale-{i}"] = [stale_ts]

    # "active" is already in the dict — is_allowed takes the filter path
    # (not the early-return for brand-new keys) and sees len(requests) > 10000.
    rl.requests["active"] = [time.time()]
    rl.is_allowed("active")

    remaining_stale = [k for k in rl.requests if k.startswith("stale-")]
    assert len(remaining_stale) == 0


def test_cleanup_preserves_active_keys():
    rl = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    stale_ts = time.time() - 200

    for i in range(10000):
        rl.requests[f"stale-{i}"] = [stale_ts]

    rl.requests["active"] = [time.time()]
    rl.is_allowed("active")

    assert "active" in rl.requests
