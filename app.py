"""OmniAI's small, authenticated chat-completions proxy."""

import fnmatch
import hmac
import os
import threading
import time
import traceback
import uuid
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from flask import (
    Blueprint,
    Flask,
    Response,
    current_app,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)
from requests.adapters import HTTPAdapter

import logging_utils
from converters import (
    convert_anthropic_response_to_openai,
    messages_to_anthropic,
    normalize_openai_response,
)
from logging_utils import _debug
from models import models
from response_utils import _error_response, _json_error_response
from streaming import (
    _pseudo_stream_non_stream_response,
    anthropic_stream_converter,
    dynamic_smooth_stream_generator,
)
from text_processing import (
    _process_non_stream_text_content,
    _strip_dynamic_system_prompt_content,
    _strip_formatting_marker,
)


SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "openrouter"})


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _payload_debug_enabled() -> bool:
    """Payload diagnostics are opt-in and are never enabled by Flask debug."""
    return _env_flag("OMNI_DEBUG_PAYLOADS", False)


def _diagnostic_logging_enabled() -> bool:
    return _env_flag("OMNI_DEBUG_LOGS", False) or _payload_debug_enabled()


class RateLimiter:
    def __init__(self, min_interval_seconds: float = 2.0):
        self.min_interval = float(min_interval_seconds)
        self.last_request_times: Dict[str, float] = {}
        self.lock = threading.Lock()

    def update_config(self, min_interval_seconds: float):
        with self.lock:
            self.min_interval = max(0.0, float(min_interval_seconds))

    def is_allowed(self, model_id: Optional[str] = None) -> Tuple[bool, float]:
        key = model_id or "global"
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_times.get(key, 0.0)
            if elapsed >= self.min_interval:
                self.last_request_times[key] = now
                return True, 0.0
            return False, self.min_interval - elapsed


rate_limiter = RateLimiter()


class StatsTracker:
    """Store operational counters without retaining prompts or responses."""

    def __init__(self, logging_enabled: Optional[bool] = None):
        self.lock = threading.RLock()
        self.start_time = time.time()
        self.daily_requests = 0
        self.daily_tokens = 0
        self.last_model = None
        self.model_distribution: Dict[str, int] = {}
        self.model_tokens: Dict[str, int] = {}
        self.recent_logs = []
        self.hourly_requests = [0] * 24
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history = []
        self._stats_sequence = 0
        self.logging_enabled = (
            _diagnostic_logging_enabled()
            if logging_enabled is None
            else bool(logging_enabled)
        )

    def _next_sequence(self) -> int:
        self._stats_sequence += 1
        return self._stats_sequence

    @staticmethod
    def _public_item(item: Dict[str, Any], include_sequence: bool = False):
        result = {key: value for key, value in item.items() if key != "_stats_seq"}
        if include_sequence:
            result["seq"] = item["_stats_seq"]
        return result

    def start_request(
        self,
        model_id: str,
        input_tokens: int = 0,
        **_ignored: Any,
    ) -> str:
        with self.lock:
            req_id = uuid.uuid4().hex[:8]
            input_tokens = max(0, int(input_tokens or 0))
            self.daily_requests += 1
            self.daily_tokens += input_tokens
            self.last_model = model_id
            self.model_distribution[model_id] = self.model_distribution.get(model_id, 0) + 1
            self.model_tokens[model_id] = self.model_tokens.get(model_id, 0) + input_tokens
            self.hourly_requests[time.localtime().tm_hour] += 1
            self.active_requests[req_id] = {
                "start_time": time.time(),
                "model": model_id,
                "input_tokens": input_tokens,
                "ttft": None,
            }
            return req_id

    def set_ttft(self, req_id: str):
        with self.lock:
            request_info = self.active_requests.get(req_id)
            if request_info is not None and request_info["ttft"] is None:
                request_info["ttft"] = time.time() - request_info["start_time"]

    def finish_request(
        self,
        req_id: str,
        output_tokens: int = 0,
        **_ignored: Any,
    ):
        with self.lock:
            request_info = self.active_requests.pop(req_id, None)
            if request_info is None:
                return
            output_tokens = max(0, int(output_tokens or 0))
            self.daily_tokens += output_tokens
            model_id = request_info["model"]
            self.model_tokens[model_id] = self.model_tokens.get(model_id, 0) + output_tokens
            item = {
                "id": req_id,
                "timestamp": time.strftime("%H:%M:%S"),
                "model": model_id,
                "input_tokens": request_info["input_tokens"],
                "output_tokens": output_tokens,
                "ttft": round(request_info["ttft"], 3)
                if request_info["ttft"] is not None
                else None,
                "duration": round(time.time() - request_info["start_time"], 3),
            }
            item["_stats_seq"] = self._next_sequence()
            self.request_history.insert(0, item)
            del self.request_history[50:]

    def add_tokens(self, count: int):
        with self.lock:
            self.daily_tokens += max(0, int(count or 0))

    def add_log(self, msg: str):
        if not self.logging_enabled:
            return
        with self.lock:
            self.recent_logs.append({
                "time": time.strftime("%H:%M:%S"),
                "msg": str(msg)[:500],
                "_stats_seq": self._next_sequence(),
            })
            del self.recent_logs[:-50]

    def get_stats(self, since: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            history = [self._public_item(item) for item in self.request_history]
            logs = [self._public_item(item) for item in reversed(self.recent_logs)]
            result: Dict[str, Any] = {
                "uptime": time.time() - self.start_time,
                "daily_requests": self.daily_requests,
                "daily_tokens": self.daily_tokens,
                "last_model": self.last_model,
                "model_distribution": dict(self.model_distribution),
                "model_tokens": dict(self.model_tokens),
                "recent_logs": logs,
                "request_history": history,
                "hourly_stats": {
                    "labels": [f"{hour}:00" for hour in range(24)],
                    "values": list(self.hourly_requests),
                },
            }
            if since is None:
                return result

            current_sequence = self._stats_sequence
            all_items = self.request_history + self.recent_logs
            oldest = min(
                (item["_stats_seq"] for item in all_items),
                default=current_sequence + 1,
            )
            full_snapshot = since == 0 or (all_items and since < oldest - 1)
            if full_snapshot:
                history = [
                    self._public_item(item, include_sequence=True)
                    for item in self.request_history
                ]
                logs = [
                    self._public_item(item, include_sequence=True)
                    for item in reversed(self.recent_logs)
                ]
            else:
                history = [
                    self._public_item(item, include_sequence=True)
                    for item in self.request_history
                    if item["_stats_seq"] > since
                ]
                logs = [
                    self._public_item(item, include_sequence=True)
                    for item in reversed(self.recent_logs)
                    if item["_stats_seq"] > since
                ]
            result.update({
                "recent_logs": logs,
                "request_history": history,
                "stats_seq": current_sequence,
                "full_snapshot": full_snapshot,
            })
            return result


stats_tracker = StatsTracker()
logging_utils._global_stats_tracker = stats_tracker


app = Flask(__name__)
app.stats_tracker = stats_tracker
app.config.from_mapping(
    PROXY_AUTH_KEY=os.getenv("PROXY_AUTH_KEY"),
    RATE_LIMITING_ENABLED=False,
    RATE_LIMIT_INTERVAL_SECONDS=2.0,
    HTTP_CONNECT_TIMEOUT=float(os.getenv("HTTP_CONNECT_TIMEOUT", "10")),
    HTTP_READ_TIMEOUT=float(os.getenv("HTTP_READ_TIMEOUT", "180")),
    PAYLOAD_DEBUG_LOGGING=False,
    STREAM_CHUNK_DEBUG_LOGGING=False,
    SMOOTH_MAX_CHUNK_SIZE=3,
    SMOOTH_QUEUE_TIMEOUT=0.05,
    SMOOTH_FIRST_CHUNK_IMMEDIATE=True,
    STREAM_DEBUG_SAMPLE_N=1,
    OPENAI_INCLUDE_USAGE=False,
    STRIP_THOUGHT_BLOCKS=False,
    STRIP_SYSTEM_PROMPT_CHARACTERS_ENABLED=False,
    SYSTEM_PROMPT_CHARACTERS_TO_STRIP=[],
    DYNAMIC_SYSTEM_PROMPT_STRIP_ENABLED=False,
    DYNAMIC_SYSTEM_PROMPT_PATTERNS_TO_STRIP=[],
)


def _diag(message: str):
    if _diagnostic_logging_enabled():
        _debug(
            message,
            logger_instance=current_app.logger,
            debug_flag=True,
        )


_MODEL_CONFIG_CACHE_LOCK = threading.Lock()
_MODEL_CONFIG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _model_config_cache() -> Dict[str, Dict[str, Any]]:
    global _MODEL_CONFIG_CACHE
    if _MODEL_CONFIG_CACHE is None:
        with _MODEL_CONFIG_CACHE_LOCK:
            if _MODEL_CONFIG_CACHE is None:
                cache: Dict[str, Dict[str, Any]] = {}
                configured_sizes = app.config.get(
                    "MODEL_SPECIFIC_SMOOTH_MAX_CHUNK_SIZES", {}
                )
                for model_id, raw_definition in models.items():
                    provider = raw_definition.get("provider")
                    if provider not in SUPPORTED_PROVIDERS:
                        continue
                    definition = dict(raw_definition)
                    options = definition.pop("provider_options", None)
                    if isinstance(options, dict):
                        definition.update(options)
                    for chunk_size, entries in configured_sizes.items():
                        for entry in entries or []:
                            if entry.get("provider") not in {"all", provider}:
                                continue
                            pattern = entry.get("model_id", "all")
                            if pattern == "all" or fnmatch.fnmatch(model_id, pattern):
                                definition["SMOOTH_MAX_CHUNK_SIZE"] = chunk_size
                                break
                    cache[model_id] = definition
                _MODEL_CONFIG_CACHE = cache
    return _MODEL_CONFIG_CACHE


def _get_model_config(model_id: str):
    definition = _model_config_cache().get(model_id)
    if definition is None:
        return None, _error_response(
            f"Unknown model: {model_id}",
            404,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    provider = definition.get("provider")
    if provider not in PROVIDERS:
        return None, _error_response(
            f"No handler for provider '{provider}'.",
            501,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    if not definition.get("api_key"):
        return None, _error_response(
            f"API key configuration is missing for model {model_id}.",
            503,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    return definition, None


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        configured_key = current_app.config.get("PROXY_AUTH_KEY") or os.getenv(
            "PROXY_AUTH_KEY"
        )
        if not configured_key:
            return _json_error_response(
                "Proxy authentication is not configured.",
                503,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        authorization = request.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token
            or not hmac.compare_digest(str(token), str(configured_key))
        ):
            return _json_error_response(
                "Unauthorized: invalid or missing Bearer token.",
                401,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        return fn(*args, **kwargs)

    return wrapper


def rate_limit_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_app.config.get("RATE_LIMITING_ENABLED", False):
            return fn(*args, **kwargs)
        model_id = None
        body = request.get_json(silent=True) if request.is_json else None
        if isinstance(body, dict):
            model_id = body.get("model")
        rate_limiter.update_config(
            current_app.config.get("RATE_LIMIT_INTERVAL_SECONDS", 2.0)
        )
        allowed, wait_time = rate_limiter.is_allowed(model_id)
        if not allowed:
            _diag("Rate limit rejected a request.")
            response = _json_error_response(
                "Rate limit exceeded.",
                429,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
            response.headers["Retry-After"] = str(max(1, int(wait_time) + 1))
            return response
        return fn(*args, **kwargs)

    return wrapper


def handle_json(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.method != "POST":
            return fn(None, *args, **kwargs)
        if not request.is_json:
            return _json_error_response(
                "Invalid Content-Type, expected application/json.",
                415,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        try:
            data = request.get_json(force=True)
        except Exception:
            return _json_error_response(
                "Bad JSON request body.",
                400,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        if not isinstance(data, dict):
            return _json_error_response(
                "JSON request body must be an object.",
                400,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        if _payload_debug_enabled():
            _diag(f"Received JSON object with {len(data)} top-level fields.")
        return fn(data, *args, **kwargs)

    return wrapper


def _standardize_messages(
    messages_in: Iterable[Dict[str, Any]],
    provider_name: Optional[str] = None,
):
    del provider_name
    messages_out = []
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "developer": "system",
        "tool": "tool",
    }
    for message in messages_in or []:
        if not isinstance(message, dict):
            continue
        role = role_map.get(str(message.get("role", "user")).lower())
        if role is None:
            continue
        content = message.get("content")
        if content is None:
            parts = [{"type": "text", "text": ""}]
        elif isinstance(content, str):
            parts = [{"type": "text", "text": _strip_formatting_marker(content)}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and "text" in part:
                    parts.append({
                        "type": "text",
                        "text": _strip_formatting_marker(str(part["text"])),
                    })
                elif (
                    part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                    and part["image_url"].get("url")
                ):
                    image = {"url": str(part["image_url"]["url"])}
                    if "detail" in part["image_url"]:
                        image["detail"] = part["image_url"]["detail"]
                    parts.append({"type": "image_url", "image_url": image})
        else:
            continue
        if not parts:
            continue

        if role == "system" and current_app.config.get(
            "STRIP_SYSTEM_PROMPT_CHARACTERS_ENABLED", False
        ):
            chars = current_app.config.get("SYSTEM_PROMPT_CHARACTERS_TO_STRIP", [])
            for part in parts:
                if part.get("type") != "text":
                    continue
                text = _strip_dynamic_system_prompt_content(
                    part.get("text", ""),
                    current_app.config,
                    current_app.logger,
                    _diagnostic_logging_enabled(),
                )
                for char in chars:
                    text = text.replace(char, "")
                part["text"] = text
        messages_out.append({"role": role, "content": parts})
    return messages_out or [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _prepare_common_headers(api_key: Optional[str] = None, extra_headers=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


_SHARED_HTTP_ADAPTER = HTTPAdapter(
    pool_connections=64,
    pool_maxsize=64,
    max_retries=0,
    pool_block=False,
)


def _http_post(
    endpoint: str,
    *,
    json_payload: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool = False,
    timeout=None,
    proxies=None,
):
    session = requests.Session()
    session.mount("http://", _SHARED_HTTP_ADAPTER)
    session.mount("https://", _SHARED_HTTP_ADAPTER)
    kwargs = {"json": json_payload, "headers": headers, "stream": stream}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if proxies is not None:
        kwargs["proxies"] = proxies
    response = session.post(endpoint, **kwargs)
    session.cookies.clear()
    return response


def _validated_float(data: Dict[str, Any], key: str) -> Optional[float]:
    if key not in data:
        return None
    try:
        return float(data[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{key}' value.") from exc


def _build_openai_compatible(data: Dict[str, Any], md: Dict[str, Any], mid: str, provider: str):
    endpoint = md.get("endpoint") or md.get("chat_endpoint")
    if not endpoint:
        raise ValueError(f"Missing endpoint for {provider} model {mid}.")
    if not md.get("api_key"):
        raise ValueError(f"API key for {provider} model '{mid}' is missing.")

    payload = dict(data)
    payload.pop("api_key", None)
    payload.pop("proxy_url", None)
    payload["model"] = md.get("original_model_name", mid)
    payload["messages"] = _standardize_messages(data.get("messages", []), provider)
    payload["stream"] = bool(data.get("stream", False))

    if "temperature" in data:
        payload["temperature"] = _validated_float(data, "temperature")
    if "top_p" in data:
        payload["top_p"] = _validated_float(data, "top_p")

    if provider == "openai":
        if mid.startswith("gpt-5.1"):
            payload.pop("max_tokens", None)
            payload.pop("temperature", None)
        if mid == "gpt-5-high":
            payload["reasoning_effort"] = "high"
        elif md.get("supports_reasoning_effort"):
            effort = data.get("reasoning_effort")
            payload["reasoning_effort"] = effort if effort in {
                "minimal", "low", "medium", "high", "xhigh"
            } else "medium"
        else:
            payload.pop("reasoning_effort", None)

    if provider == "openrouter" and "kimi-k3" in mid.lower():
        payload["provider"] = {"only": ["modal/mxfp4"], "allow_fallbacks": False}
    return payload, endpoint, _prepare_common_headers(md.get("api_key"))


def build_openai(data: Dict[str, Any], md: Dict[str, Any], mid: str):
    return _build_openai_compatible(data, md, mid, "openai")


def build_openrouter(data: Dict[str, Any], md: Dict[str, Any], mid: str):
    return _build_openai_compatible(data, md, mid, "openrouter")


def build_anthropic(data: Dict[str, Any], md: Dict[str, Any], mid: str):
    endpoint = md.get("endpoint") or md.get("chat_endpoint")
    if not endpoint:
        raise ValueError(f"Missing endpoint for Anthropic model {mid}.")
    api_key = md.get("api_key")
    if not api_key:
        raise ValueError(f"API key for Anthropic model '{mid}' is missing.")

    standardized = _standardize_messages(data.get("messages", []), "anthropic")
    system_parts = [message for message in standardized if message["role"] == "system"]
    regular_messages = [message for message in standardized if message["role"] != "system"]
    system_text = "\n".join(
        _content_to_text(message.get("content")) for message in system_parts
    ).strip()
    payload: Dict[str, Any] = {
        "model": md.get("original_model_name", mid),
        "messages": messages_to_anthropic(regular_messages),
        "stream": bool(data.get("stream", False)),
        "max_tokens": 4096,
    }
    if system_text:
        payload["system"] = system_text
    if "max_tokens" in data:
        try:
            payload["max_tokens"] = int(data["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid 'max_tokens' value.") from exc
        if payload["max_tokens"] <= 0:
            raise ValueError("'max_tokens' must be positive.")

    original_model = str(payload["model"]).lower()
    supports_effort = bool(md.get("supports_reasoning_effort")) or any(
        marker in original_model for marker in ("4-5", "4-6", "4-7", "4-8", "fable")
    )
    thinking = data.get("thinking")
    if thinking is not None:
        if isinstance(thinking, dict) and supports_effort:
            effort = data.get("reasoning_effort") or thinking.get("effort") or "high"
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {
                "effort": effort if effort in {"low", "medium", "high"} else "high"
            }
        elif isinstance(thinking, dict):
            payload["thinking"] = dict(thinking)
            if payload["thinking"].get("type") == "enabled":
                budget = int(payload["thinking"].get("budget_tokens", 16000))
                payload["thinking"]["budget_tokens"] = budget
                payload["max_tokens"] = max(payload["max_tokens"], budget + 4000)
        else:
            payload["thinking"] = thinking
    elif data.get("thinking_enabled") or "thinking_budget" in data or "reasoning_effort" in data:
        if supports_effort:
            effort = data.get("reasoning_effort", "high")
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {
                "effort": effort if effort in {"low", "medium", "high"} else "high"
            }
        elif md.get("supports_thinking"):
            budget = int(data.get("thinking_budget") or 16000)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(payload["max_tokens"], budget + 4000)

    # Anthropic's compatibility payload deliberately does not forward temperature.
    top_p = _validated_float(data, "top_p")
    if top_p is not None:
        payload["top_p"] = top_p
    if "stop" in data:
        payload["stop_sequences"] = data["stop"] if isinstance(data["stop"], list) else [data["stop"]]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    return payload, endpoint, headers


PROVIDERS: Dict[str, Any] = {}


def load_providers():
    """Register only the three supported request builders."""
    PROVIDERS.clear()
    PROVIDERS.update({
        "openai": build_openai,
        "anthropic": build_anthropic,
        "openrouter": build_openrouter,
    })
    return PROVIDERS


def _provider_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return "Upstream provider request failed."
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail")
            if detail:
                return str(detail)[:500]
        if data.get("message"):
            return str(data["message"])[:500]
    return "Upstream provider request failed."


def _make_request(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool = False,
    timeout=None,
    provider_name: Optional[str] = None,
    proxies=None,
):
    del provider_name
    attempts = int(current_app.config.get("HTTP_MAX_RETRIES", 2)) + 1
    if timeout is None:
        timeout = (
            current_app.config.get("HTTP_CONNECT_TIMEOUT", 10.0),
            current_app.config.get("HTTP_READ_TIMEOUT", 180.0),
        )
    for attempt in range(attempts):
        response = None
        try:
            response = _http_post(
                endpoint,
                json_payload=payload,
                headers=headers,
                stream=stream,
                timeout=timeout,
                proxies=proxies,
            )
            if response.ok:
                return response, None
            status = response.status_code
            detail = _provider_error_detail(response)
            transient = status == 429 or status >= 500
            response.close()
            if transient and attempt + 1 < attempts:
                time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                continue
            public_status = status if 400 <= status < 500 else 502
            return None, _json_error_response(
                "Provider request failed.",
                public_status,
                detail,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        except requests.exceptions.Timeout as exc:
            if response is not None:
                response.close()
            if attempt + 1 < attempts:
                continue
            return None, _json_error_response(
                "Provider request timed out.",
                504,
                str(exc),
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        except requests.exceptions.RequestException as exc:
            if response is not None:
                response.close()
            if attempt + 1 < attempts:
                continue
            return None, _json_error_response(
                "Unable to contact provider.",
                502,
                str(exc),
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
        except Exception:
            if response is not None:
                response.close()
            return None, _json_error_response(
                "Internal provider request error.",
                500,
                logger_instance=current_app.logger,
                debug_flag=_diagnostic_logging_enabled(),
            )
    return None, _json_error_response(
        "Provider request failed.",
        502,
        logger_instance=current_app.logger,
        debug_flag=_diagnostic_logging_enabled(),
    )


def _estimate_input_tokens(messages: Iterable[Dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages or []:
        if isinstance(message, dict):
            total_chars += len(_content_to_text(message.get("content")))
    return total_chars // 4


def _finish_request(req_id: Optional[str], output_tokens: int = 0):
    if req_id:
        stats_tracker.finish_request(req_id, output_tokens=output_tokens)


def _stream_error_response(error_response, model_id: str):
    try:
        payload = error_response.get_json() or {}
        message = payload.get("error", {}).get("message", "Provider request failed.")
    except Exception:
        message = "Provider request failed."
    return Response(
        _pseudo_stream_non_stream_response(
            {
                "choices": [{
                    "message": {"content": f"[Provider error] {message}"},
                    "finish_reason": "error",
                }]
            },
            model_id,
            current_app.config,
            current_app.logger,
            _diagnostic_logging_enabled(),
        ),
        mimetype="text/event-stream",
    )


def _process_normalized_response(
    response_data: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    choices = response_data.get("choices")
    if not isinstance(choices, list):
        return response_data
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _process_non_stream_text_content(
                content,
                current_app.config,
                current_app.logger,
                _diagnostic_logging_enabled(),
                model_id,
            )
    return response_data


def _handle_chat_completions(data: Dict[str, Any]):
    if not data:
        return _json_error_response(
            "JSON body required.",
            400,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    model_id = data.get("model")
    if not isinstance(model_id, str) or not model_id:
        return _json_error_response(
            "Missing 'model' field.",
            400,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )

    model_definition, error_data = _get_model_config(model_id)
    if error_data:
        return jsonify(error_data), error_data["error"]["code"]
    provider_name = model_definition["provider"]
    req_id = stats_tracker.start_request(
        model_id,
        input_tokens=_estimate_input_tokens(data.get("messages", [])),
    )
    user_stream = bool(data.get("stream", False))
    try:
        builder = PROVIDERS[provider_name]
        payload, endpoint, headers = builder(data, model_definition, model_id)
        proxies = None
        proxy_url = model_definition.get("proxy_url")
        if proxy_url:
            proxies = {"https": proxy_url}
        response, error_response = _make_request(
            endpoint,
            payload,
            headers,
            stream=user_stream,
            provider_name=provider_name,
            proxies=proxies,
        )
        if error_response is not None:
            _finish_request(req_id)
            return _stream_error_response(error_response, model_id) if user_stream else error_response

        if user_stream:
            def stream_body():
                try:
                    if provider_name == "anthropic":
                        generator = anthropic_stream_converter(
                            response.iter_lines(decode_unicode=False),
                            model_id,
                            current_app.config,
                            current_app.logger,
                            _diagnostic_logging_enabled(),
                            req_id=req_id,
                            stats_tracker=stats_tracker,
                        )
                    else:
                        generator = dynamic_smooth_stream_generator(
                            response.iter_lines(decode_unicode=False),
                            model_id,
                            provider_name,
                            current_app.config,
                            current_app.logger,
                            _diagnostic_logging_enabled(),
                            reader_specific_config_param={"is_cumulative_delta": False},
                            model_config=model_definition,
                            req_id=req_id,
                            stats_tracker=stats_tracker,
                        )
                    yield from generator
                finally:
                    response.close()

            return Response(
                stream_with_context(stream_body()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        try:
            response_data = response.json()
        finally:
            response.close()
        if provider_name == "anthropic":
            normalized = convert_anthropic_response_to_openai(response_data, model_id)
        else:
            normalized = normalize_openai_response(response_data, model_id)
        normalized = _process_normalized_response(normalized, model_id)
        content = ""
        try:
            content = normalized["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            pass
        output_tokens = normalized.get("usage", {}).get("completion_tokens")
        if output_tokens is None:
            output_tokens = len(content) // 4 if isinstance(content, str) else 0
        _finish_request(req_id, output_tokens)
        return jsonify(normalized)
    except ValueError as exc:
        _finish_request(req_id)
        return _json_error_response(
            f"Bad request: {exc}",
            400,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    except requests.exceptions.JSONDecodeError:
        _finish_request(req_id)
        return _json_error_response(
            "Provider returned invalid JSON.",
            502,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    except Exception:
        _finish_request(req_id)
        if _diagnostic_logging_enabled():
            _debug(
                "Chat completion processing failed.",
                traceback.format_exc(),
                logger_instance=current_app.logger,
                debug_flag=True,
            )
        return _json_error_response(
            "Failed to process provider response.",
            500,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )


PROVIDERS = load_providers()
proxy_bp = Blueprint("proxy", __name__, url_prefix="/v1")


@proxy_bp.route("/chat/completions", methods=["POST"])
@auth_required
@rate_limit_required
@handle_json
def route_chat_completions(data):
    return _handle_chat_completions(data)


@proxy_bp.route("/models", methods=["GET"])
@auth_required
def route_list_models():
    model_list = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": definition.get("provider"),
        }
        for model_id, definition in _model_config_cache().items()
    ]
    return jsonify({"object": "list", "data": model_list})


app.register_blueprint(proxy_bp)


@app.route("/dashboard")
@app.route("/dashboard/")
def route_dashboard():
    return send_from_directory("dashboard", "index.html")


@app.route("/dashboard/<path:filename>")
def route_dashboard_static(filename):
    return send_from_directory("dashboard", filename)


@app.route("/stats")
@auth_required
def route_stats():
    since_value = request.args.get("since")
    if since_value is None:
        return jsonify(stats_tracker.get_stats())
    try:
        since = int(since_value)
        if since < 0:
            raise ValueError
    except (TypeError, ValueError):
        return _json_error_response(
            "'since' must be a non-negative integer.",
            400,
            logger_instance=current_app.logger,
            debug_flag=_diagnostic_logging_enabled(),
        )
    return jsonify(stats_tracker.get_stats(since=since))


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "message": "OmniAI chat proxy is running.",
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    app.run(port=8000, debug=False, use_reloader=False)
