import time
import threading
from typing import Dict, List

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
            
            # Periodic cleanup if tracking too many keys
            if len(self.requests) > 10000:
                self._cleanup(now)
                
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

    def get_retry_after(self, key: str) -> int:
        now = time.time()
        with self.lock:
            if key not in self.requests or not self.requests[key]:
                return 0
            # Smallest wait is the oldest timestamp + window - now
            oldest = self.requests[key][0]
            wait = int(oldest + self.window_seconds - now)
            return max(1, wait)
