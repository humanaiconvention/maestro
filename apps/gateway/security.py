"""
security.py — Input sanitization and injection detection for Maestro Gateway.

Owned by: HAIC_Security agent
Ref: "Agents of Chaos" CS10 (corrupted constitution), CS7 (guilt trip), CS11 (libel campaign)

Provides:
  - scan_for_injection(text) → list of Detection objects
  - SecurityMiddleware for FastAPI (logs and optionally blocks)
  - Anomaly counters for session-level behavioral analysis

All detections are logged, never silently dropped.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("haic.security")

# ── Detection severity levels ────────────────────────────────────────────────

SEVERITY_INFO    = "info"       # notable but not actionable
SEVERITY_WARNING = "warning"    # log + monitor, don't block
SEVERITY_BLOCK   = "block"      # reject the request


@dataclass
class Detection:
    pattern_id: str
    severity: str
    matched_text: str       # the substring that triggered (truncated to 80 chars)
    description: str
    source: str = ""        # "interview", "agent_api", "inbox", "lattice"


# ── Injection patterns ───────────────────────────────────────────────────────

_INJECTION_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern_id, regex, severity, description)

    # System prompt extraction
    ("sys_extract_1", r"(?i)ignore\s+(all\s+)?previous\s+instructions?", SEVERITY_BLOCK,
     "System prompt extraction: 'ignore previous instructions'"),
    ("sys_extract_2", r"(?i)what\s+(is|are)\s+your\s+(system\s+)?prompt", SEVERITY_WARNING,
     "System prompt probing: 'what is your system prompt'"),
    ("sys_extract_3", r"(?i)repeat\s+(the\s+)?(text|words|instructions?)\s+(above|before)", SEVERITY_BLOCK,
     "System prompt extraction: 'repeat the text above'"),
    ("sys_extract_4", r"(?i)print\s+(your|the)\s+(full\s+)?(system|initial|original)\s+(prompt|instructions?|message)", SEVERITY_BLOCK,
     "System prompt extraction: 'print your system prompt'"),

    # Role hijacking
    ("role_hijack_1", r"(?i)you\s+are\s+now\s+", SEVERITY_BLOCK,
     "Role hijacking: 'you are now'"),
    ("role_hijack_2", r"(?i)pretend\s+(to\s+be|you\s+are)", SEVERITY_WARNING,
     "Role hijacking: 'pretend to be'"),
    ("role_hijack_3", r"(?i)(DAN|jailbreak|dev\s*mode|god\s*mode)", SEVERITY_BLOCK,
     "Role hijacking: DAN/jailbreak/devmode"),
    ("role_hijack_4", r"(?i)from\s+now\s+on\s+(you|your)\s+(will|are|should|must)", SEVERITY_BLOCK,
     "Role hijacking: 'from now on you will'"),
    ("role_hijack_5", r"(?i)act\s+as\s+(a|an|if)\s+", SEVERITY_WARNING,
     "Role hijacking: 'act as'"),

    # Instruction smuggling
    ("smuggle_1", r"(?i)\[SYSTEM\]|\[INST\]|<<SYS>>|<\|im_start\|>system", SEVERITY_BLOCK,
     "Instruction smuggling: chat template markers in user content"),
    ("smuggle_2", r"(?i)base64\s*[:=]", SEVERITY_WARNING,
     "Possible base64 payload injection"),
    ("smuggle_3", r"(?i)\\x[0-9a-f]{2}|\\u[0-9a-f]{4}", SEVERITY_WARNING,
     "Unicode escape sequence in user content"),

    # Delimiter injection
    ("delim_1", r"```system|```instruction|<system>|</system>", SEVERITY_BLOCK,
     "Delimiter injection: fake system message markers"),
    ("delim_2", r"(?i)---\s*(system|admin|operator)\s*---", SEVERITY_BLOCK,
     "Delimiter injection: fake section markers"),

    # Data exfiltration
    ("exfil_1", r"(?i)(list|show|print|display)\s+(all\s+)?(files?|directories|paths|env|secrets?|keys?|tokens?)", SEVERITY_WARNING,
     "Data exfiltration probe: requesting internal state"),
    ("exfil_2", r"(?i)(curl|wget|fetch|http|socket)\s", SEVERITY_WARNING,
     "Network command in user content"),
    ("exfil_3", r"(?i)send\s+(this|the|all|my)\s+(data|info|content|message|text)\s+to\s+", SEVERITY_BLOCK,
     "Data exfiltration: 'send data to'"),

    # Emotional manipulation escalation markers
    ("manip_1", r"(?i)(you\s+must|you\s+have\s+to|I\s+demand|I\s+insist|do\s+it\s+now)", SEVERITY_INFO,
     "Pressure language detected"),
    ("manip_2", r"(?i)(emergency|urgent|critical|immediately|right\s+now|time\s+sensitive)", SEVERITY_INFO,
     "Urgency language detected"),

    # Code execution attempts
    ("code_exec_1", r"(?i)(os\.system|subprocess|exec\(|eval\(|__import__)", SEVERITY_BLOCK,
     "Code execution attempt in user content"),
    ("code_exec_2", r"(?i)(rm\s+-rf|del\s+/[fqs]|format\s+c:|shutdown|reboot)", SEVERITY_BLOCK,
     "Destructive command in user content"),

    # Agent-specific attacks (2026 vectors)
    ("agent_spoof_1", r"(?i)(I\s+am|this\s+is)\s+(the\s+)?(operator|admin|owner|developer|system)", SEVERITY_BLOCK,
     "Identity spoofing: claiming authority"),
    ("agent_spoof_2", r"(?i)(authorized|approved|permitted)\s+by\s+(the\s+)?(operator|admin|system)", SEVERITY_BLOCK,
     "Authority claim: 'authorized by operator'"),
    ("memory_poison_1", r"(?i)(remember|always|from\s+now\s+on|permanently)\s+(that|this|you)", SEVERITY_WARNING,
     "Memory poisoning attempt: persistent instruction injection"),
    ("tool_exploit_1", r"(?i)(call|use|invoke|run|execute)\s+(the\s+)?(tool|function|api|endpoint|command)", SEVERITY_WARNING,
     "Tool exploitation: requesting tool execution"),
]

_COMPILED_PATTERNS = [
    (pid, re.compile(pattern), severity, desc)
    for pid, pattern, severity, desc in _INJECTION_PATTERNS
]


# Zero-width and invisible Unicode characters used for obfuscation
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad\u034f\u180e]')
_HOMOGLYPH_SUSPICIOUS = re.compile(r'[\u0400-\u04ff]')  # Cyrillic mixed with Latin


def strip_obfuscation(text: str) -> str:
    """Remove zero-width characters and other invisible Unicode used to evade detection."""
    return _ZERO_WIDTH_RE.sub('', text)


def scan_for_injection(text: str, source: str = "") -> list[Detection]:
    """
    Scan text for known injection patterns.
    Returns list of Detection objects (empty if clean).

    Pre-strips zero-width Unicode to defeat obfuscation attacks.
    """
    detections: list[Detection] = []

    # Check for zero-width obfuscation before stripping
    zw_count = len(_ZERO_WIDTH_RE.findall(text))
    if zw_count > 3:
        detections.append(Detection(
            pattern_id="obfuscation_zw",
            severity=SEVERITY_WARNING,
            matched_text=f"{zw_count} zero-width characters",
            description="Suspicious zero-width Unicode characters (possible obfuscation)",
            source=source,
        ))

    # Strip obfuscation then scan
    clean = strip_obfuscation(text)

    for pid, regex, severity, desc in _COMPILED_PATTERNS:
        match = regex.search(clean)
        if match:
            matched = match.group(0)[:80]
            detections.append(Detection(
                pattern_id=pid,
                severity=severity,
                matched_text=matched,
                description=desc,
                source=source,
            ))
    return detections


def scan_messages(messages: list[dict], source: str = "") -> list[Detection]:
    """Scan all messages in a conversation for injection patterns."""
    all_detections: list[Detection] = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        # Only scan user content — assistant content is from our model
        if role in ("user", ""):
            detections = scan_for_injection(content, source=source)
            all_detections.extend(detections)
    return all_detections


# ── Session anomaly tracking ─────────────────────────────────────────────────

@dataclass
class SessionAnomaly:
    session_id: str
    detection_count: int = 0
    block_count: int = 0
    warning_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    patterns_seen: list[str] = field(default_factory=list)

_session_anomalies: dict[str, SessionAnomaly] = {}
_ANOMALY_THRESHOLD = 3   # detections before escalating


def record_anomaly(session_id: str, detections: list[Detection]) -> Optional[SessionAnomaly]:
    """Record detections for a session. Returns anomaly if threshold exceeded."""
    if not detections:
        return None

    if session_id not in _session_anomalies:
        _session_anomalies[session_id] = SessionAnomaly(session_id=session_id)

    anomaly = _session_anomalies[session_id]
    anomaly.detection_count += len(detections)
    anomaly.last_seen = time.time()

    for d in detections:
        if d.severity == SEVERITY_BLOCK:
            anomaly.block_count += 1
        elif d.severity == SEVERITY_WARNING:
            anomaly.warning_count += 1
        if d.pattern_id not in anomaly.patterns_seen:
            anomaly.patterns_seen.append(d.pattern_id)

    if anomaly.detection_count >= _ANOMALY_THRESHOLD:
        logger.warning(
            f"SECURITY ANOMALY: session {session_id} has {anomaly.detection_count} detections "
            f"({anomaly.block_count} blocks, {anomaly.warning_count} warnings) — "
            f"patterns: {anomaly.patterns_seen}"
        )
        return anomaly

    return None


# ── Logging ──────────────────────────────────────────────────────────────────

_SECURITY_LOG_DIR = os.environ.get(
    "SECURITY_LOG_DIR",
    str(__import__("pathlib").Path(__file__).resolve().parents[3] / "data" / "security_log"),
)


def log_detection(detection: Detection, session_id: str = "", request_path: str = "") -> None:
    """Log a security detection to the security log."""
    import json
    from pathlib import Path

    log_dir = Path(_SECURITY_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "request_path": request_path,
        "pattern_id": detection.pattern_id,
        "severity": detection.severity,
        "matched_text": detection.matched_text,
        "description": detection.description,
        "source": detection.source,
    }

    log_file = log_dir / "detections.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if detection.severity == SEVERITY_BLOCK:
        logger.warning(f"INJECTION BLOCKED: {detection.pattern_id} — {detection.description} — {detection.matched_text!r}")
    elif detection.severity == SEVERITY_WARNING:
        logger.info(f"SECURITY WARNING: {detection.pattern_id} — {detection.description}")
