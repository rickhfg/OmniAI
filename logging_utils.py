# logging_utils.py
import json
from copy import deepcopy
from typing import Dict, Any, Optional
from flask import current_app, has_app_context

_global_stats_tracker = None
_SENSITIVE_LOG_KEYS = frozenset({
    "content",
    "data",
    "input",
    "input_text",
    "last_input",
    "output",
    "output_text",
    "prompt",
    "response",
    "reasoning",
    "reasoning_content",
    "text",
})


def _debug_logging_enabled(debug_status: bool) -> bool:
    """Return whether debug messages may reach logs or dashboard stats."""
    # Callers pass an explicit True for opt-in diagnostics (for example the
    # payload-debug environment flag), even when Flask's interactive DEBUG
    # mode remains disabled.
    return bool(debug_status)


def _redact_log_value(value: Any, key: Optional[str] = None) -> Any:
    """Copy structured log data without retaining prompt/response text."""
    normalized_key = str(key).lower() if key is not None else ""
    if normalized_key in _SENSITIVE_LOG_KEYS:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return f"[redacted {normalized_key}; {len(value)} chars]"
        return f"[redacted {normalized_key}]"

    if isinstance(value, dict):
        return {item_key: _redact_log_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_log_value(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return f"[redacted bytes; {len(value)} bytes]"
    return value


def _format_log_object(obj: Any) -> str:
    """Format an optional debug object after privacy-safe redaction."""
    if isinstance(obj, dict):
        obj = _prepare_payload_for_logging(obj)
    else:
        obj = _redact_log_value(obj)
    return repr(obj)

# ─── Internal Logging Infrastructure ─────────────────────────────────────────
class _LogWrapper:
    def __init__(self, logger_instance, debug_status: bool, prefix: str, sample_n: Optional[int] = None):
        self.logger = logger_instance
        self.debug_status = debug_status
        self.prefix = prefix
        # Sampling: log every Nth debug call if set
        self.sample_n = sample_n or 1
        self._debug_counter = 0

    def debug(self, msg, obj=None):
        if not _debug_logging_enabled(self.debug_status):
            return

        s = f"[{self.prefix}] {msg}"
        if obj is not None:
            s += " -> " + _format_log_object(obj)

        try:
            if self.sample_n:
                eff_sample = int(self.sample_n)
            elif has_app_context():
                eff_sample = int(current_app.config.get('STREAM_DEBUG_SAMPLE_N', 1))
            else:
                eff_sample = 1
        except Exception:
            eff_sample = 1
        eff_sample = max(1, eff_sample)

        self._debug_counter += 1
        if self._debug_counter % eff_sample != 0:
            return

        try:
            tracker = _global_stats_tracker
            if tracker:
                tracker.add_log(s)
            elif has_app_context() and current_app and hasattr(current_app, 'stats_tracker'):
                current_app.stats_tracker.add_log(s)
        except Exception:
            pass

        if self.logger:
            self.logger.debug(s)

    def warning(self, msg, exc_info=False):
        s = f"[{self.prefix}] {msg}"
        try:
            tracker = _global_stats_tracker
            if tracker:
                tracker.add_log(f"[WARNING] {s}")
            elif has_app_context() and current_app and hasattr(current_app, 'stats_tracker'):
                current_app.stats_tracker.add_log(f"[WARNING] {s}")
        except Exception:
            pass
        if self.logger:
            self.logger.warning(s, exc_info=exc_info)

    def error(self, msg, exc_info=False):
        s = f"[{self.prefix}] {msg}"
        try:
            tracker = _global_stats_tracker
            if tracker:
                tracker.add_log(f"[ERROR] {s}")
            elif has_app_context() and current_app and hasattr(current_app, 'stats_tracker'):
                current_app.stats_tracker.add_log(f"[ERROR] {s}")
        except Exception:
            pass
        if self.logger:
            self.logger.error(s, exc_info=exc_info)

def _truncate_base64_data(data: str, max_len: int = 64, ellipsis: str = "[...]") -> str:
    """Truncates a base64 string for logging purposes."""
    if len(data) > max_len:
        half_len = (max_len - len(ellipsis)) // 2
        return data[:half_len] + ellipsis + data[-half_len:]
    return data

def _truncate_text_content(text: str, max_words: int = 3, ellipsis: str = "[...]") -> str:
    """Truncates a text string by word count for logging purposes."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + ellipsis
    return text

def _prepare_payload_for_logging(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively truncates base64 data and text content in a payload for logging."""
    try:
        loggable_payload = json.loads(json.dumps(payload)) # Deep copy to avoid modifying original
    except (TypeError, ValueError):
        loggable_payload = deepcopy(payload)

    def _process_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                if "inline_data" in v and isinstance(v.get("inline_data"), dict) and "data" in v["inline_data"]:
                    v["inline_data"]["data"] = _truncate_base64_data(v["inline_data"]["data"])
                else:
                    _process_dict(v)
            elif isinstance(v, list):
                if k == "messages":
                    for message in v:
                        if isinstance(message, dict) and "content" in message:
                            content = message["content"]
                            if isinstance(content, str):
                                message["content"] = _truncate_text_content(content)
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                                        part["text"] = _truncate_text_content(part["text"])
                else:
                    for item in v:
                        if isinstance(item, dict): _process_dict(item)

    _process_dict(loggable_payload)
    return _redact_log_value(loggable_payload)

def flask_log_debug(msg, obj=None, logger_instance=None, debug_flag=None):
    # Prioritize explicitly passed logger/debug_flag
    effective_logger = logger_instance
    effective_debug_status = debug_flag

    # Fallback to current_app context if not explicitly provided
    if effective_logger is None or effective_debug_status is None:
        try:
            if has_app_context():
                if effective_logger is None: effective_logger = current_app.logger
                if effective_debug_status is None: effective_debug_status = current_app.debug
        except RuntimeError: # No active Flask app context
            pass

    effective_debug_status = bool(effective_debug_status)
    if not _debug_logging_enabled(effective_debug_status):
        return

    s = f"[DEBUG] {msg}"
    if obj is not None:
        s += " -> " + _format_log_object(obj)

    # Push to StatsTracker only when debug logging is explicitly enabled.
    # This keeps normal dashboard stats aggregate-only and avoids retaining
    # payload/debug messages in the recent-log ring by default.
    try:
        tracker = _global_stats_tracker
        if tracker:
            tracker.add_log(s)
        elif has_app_context() and hasattr(current_app, 'stats_tracker'):
            current_app.stats_tracker.add_log(s)
    except (ImportError, RuntimeError, AttributeError):
        pass

    if effective_logger:
        effective_logger.debug(s)

_debug = flask_log_debug
