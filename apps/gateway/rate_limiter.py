import os
import time
import logging
import threading
from typing import Dict, List

logger = logging.getLogger(__name__)

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            if key not in self.requests:
                self.requests[key] = [now]
                return True

            # Filter timestamps within the sliding window
            timestamps = [t for t in self.requests[key] if now - t < self.window_seconds]

            allowed = False
            if len(timestamps) < self.max_requests:
                timestamps.append(now)
                self.requests[key] = timestamps
                allowed = True
            else:
                self.requests[key] = timestamps
                allowed = False
            
            # Periodic cleanup: at 5k keys prune all entries older than one
            # full window (eager), not just keys idle for 2x window (lazy).
            if len(self.requests) > 5000:
                self._cleanup_eager(now)
                
            return allowed

    def _cleanup(self, now: float):
        """Removes keys whose newest timestamp is older than 2x window_seconds."""
        keys_to_remove = []
        cleanup_threshold = 2 * self.window_seconds
        for key, timestamps in self.requests.items():
            if not timestamps or (now - timestamps[-1]) > cleanup_threshold:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.requests[key]

    def _cleanup_eager(self, now: float):
        """Eager cleanup triggered at 5k keys.

        Prunes the timestamp list for every key down to only entries within
        the current window, then removes keys with no remaining timestamps.
        This bounds memory more tightly than _cleanup (which only removes keys
        idle for 2x the window).
        """
        keys_to_remove = []
        for key, timestamps in self.requests.items():
            active = [t for t in timestamps if now - t < self.window_seconds]
            if not active:
                keys_to_remove.append(key)
            else:
                self.requests[key] = active
        for key in keys_to_remove:
            del self.requests[key]

    def get_retry_after(self, key: str) -> int:
        now = time.time()
        with self.lock:
            if key not in self.requests or not self.requests[key]:
                return 0
            # Smallest wait is the oldest timestamp + window - now
            oldest = self.requests[key][0]
            wait = int(oldest + self.window_seconds - now)
            return max(1, wait)


class RedisRateLimiter:
    """Sliding window rate limiter backed by Redis (ZADD + ZREMRANGEBYSCORE).

    Falls back to an in-memory SlidingWindowRateLimiter if Redis is unavailable.
    """

    def _connect(self):
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            logger.info(f"RedisRateLimiter connected to {redis_url}")
        except Exception as e:
            logger.warning(
                f"RedisRateLimiter: could not connect to Redis ({e}). "
                "Falling back to in-memory rate limiter."
            )
            self._use_fallback = True

    # Atomic sliding-window check-and-increment via Lua.
    # KEYS[1] = redis key, ARGV[1] = window_start, ARGV[2] = now,
    # ARGV[3] = max_requests, ARGV[4] = ttl_seconds
    # Returns 1 if allowed, 0 if denied.
    _LUA_SCRIPT = """
local key         = KEYS[1]
local window_start = tonumber(ARGV[1])
local now          = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local ttl          = tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
local count = redis.call('ZCARD', key)
if count < max_requests then
    redis.call('ZADD', key, now, tostring(now))
    redis.call('EXPIRE', key, ttl)
    return 1
end
return 0
"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._redis = None
        self._fallback: SlidingWindowRateLimiter = SlidingWindowRateLimiter(
            max_requests=max_requests, window_seconds=window_seconds
        )
        self._use_fallback = False
        self._script = None
        self._connect()

    def is_allowed(self, key: str) -> bool:
        if self._use_fallback:
            return self._fallback.is_allowed(key)
        try:
            now = time.time()
            window_start = now - self.window_seconds
            redis_key = f"rl:{key}"
            if self._script is None:
                self._script = self._redis.register_script(self._LUA_SCRIPT)
            result = self._script(
                keys=[redis_key],
                args=[window_start, now, self.max_requests, self.window_seconds * 2],
            )
            return bool(result)
        except Exception as e:
            logger.warning(
                f"RedisRateLimiter: Redis error ({e}). Falling back to in-memory."
            )
            self._use_fallback = True
            return self._fallback.is_allowed(key)

    def get_retry_after(self, key: str) -> int:
        if self._use_fallback:
            return self._fallback.get_retry_after(key)
        try:
            redis_key = f"rl:{key}"
            # Oldest score still in the window
            oldest = self._redis.zrange(redis_key, 0, 0, withscores=True)
            if not oldest:
                return 1
            oldest_ts = oldest[0][1]
            wait = int(oldest_ts + self.window_seconds - time.time())
            return max(1, wait)
        except Exception as e:
            logger.warning(f"RedisRateLimiter: Redis error in get_retry_after ({e}).")
            return self._fallback.get_retry_after(key)
