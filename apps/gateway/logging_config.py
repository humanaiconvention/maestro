"""
logging_config.py — Structured JSON logging for Maestro gateway.

Emits log records as JSON lines (one JSON object per log record).

Fields included in every record:
  ts        ISO-8601 UTC timestamp
  level     DEBUG / INFO / WARNING / ERROR / CRITICAL
  logger    Logger name (e.g. "apps.gateway.main")
  msg       Human-readable message

Extra keyword arguments passed via ``extra={...}`` are merged into the
top-level JSON object, e.g.:
    logger.info("consent_recorded", extra={"session_id": sid, "kernel_type": kt})

When ``LOG_FORMAT=json`` is set (or ``use_json=True`` is passed), all
output goes to stdout as line-delimited JSON suitable for log aggregators
(Loki, Datadog, CloudWatch, etc.).

Without ``LOG_FORMAT=json`` the formatter falls back to a human-readable
``asctime level logger: message`` line — convenient for local dev.

Usage (call once at gateway startup, before the first log line):
    from apps.gateway.logging_config import configure_logging
    configure_logging()
"""

import datetime
import json
import logging
import sys
from typing import Optional

# Fields that live on every LogRecord but are not useful in JSON output.
_STDLIB_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "taskName",
    "message",
})


class _JsonFormatter(logging.Formatter):
    """Format each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "ts":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Merge caller-supplied extra fields (skip stdlib internals)
        for key, value in record.__dict__.items():
            if key in _STDLIB_FIELDS or key.startswith("_"):
                continue
            try:
                json.dumps(value)   # skip non-serialisable values silently
                obj[key] = value
            except (TypeError, ValueError):
                obj[key] = str(value)

        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(obj, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    use_json: Optional[bool] = None,
) -> None:
    """
    Configure the root logger.

    Args:
        level:    Log level string.  Defaults to the ``LOG_LEVEL`` env var,
                  or ``"INFO"`` if that is not set.
        use_json: Emit JSON lines when True.  Auto-detects when None:
                  JSON if ``LOG_FORMAT=json`` is set, plain text otherwise.

    Call once at application startup (main.py lifespan or module level).
    Subsequent calls replace the handler list, so it is safe to call again
    if settings change (e.g. during testing).
    """
    import os

    if use_json is None:
        use_json = os.environ.get("LOG_FORMAT", "").lower() == "json"

    effective_level = os.environ.get("LOG_LEVEL", level).upper()

    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, effective_level, logging.INFO))
    root.handlers = [handler]

    # Silence high-volume third-party loggers that add noise without value.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
