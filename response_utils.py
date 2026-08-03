"""Small Flask response helpers shared by the proxy routes."""

import json
import time
import uuid
from typing import Any, Dict

from flask import Response, jsonify

from logging_utils import _LogWrapper, flask_log_debug


def _error_response(
    msg,
    code=400,
    details=None,
    logger_instance=None,
    debug_flag=None,
) -> Dict[str, Any]:
    error = {"message": str(msg), "type": "proxy_error", "code": code}
    if details:
        error["details"] = str(details)
    flask_log_debug(
        f"Error response {code}: {str(msg)[:160]}",
        logger_instance=logger_instance,
        debug_flag=debug_flag,
    )
    return {"error": error}


def _json_error_response(
    msg,
    code=400,
    details=None,
    logger_instance=None,
    debug_flag=None,
):
    response = jsonify(_error_response(msg, code, details, logger_instance, debug_flag))
    response.status_code = code
    return response


def aggregate_stream_response(
    flask_response: Response,
    logger,
    debug_status: bool,
) -> Dict[str, Any]:
    """Aggregate a normalized OpenAI SSE response for compatibility callers."""
    log = _LogWrapper(logger, debug_status, "STREAM_AGGREGATOR")
    if not hasattr(flask_response, "iter_encoded"):
        return _error_response(
            "Invalid stream response for aggregation.",
            500,
            logger_instance=logger,
            debug_flag=debug_status,
        )

    content_parts = []
    reasoning_parts = []
    final_reason = "stop"
    model_id = "unknown_model"
    base_id = "unknown_id"
    try:
        for line_bytes in flask_response.iter_encoded():
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                log.warning("Ignoring malformed stream event during aggregation.")
                continue
            if not isinstance(chunk, dict):
                continue
            model_id = chunk.get("model", model_id)
            chunk_id = chunk.get("id")
            if base_id == "unknown_id" and isinstance(chunk_id, str):
                base_id = chunk_id
            choices = chunk.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(str(reasoning))
            if choice.get("finish_reason"):
                final_reason = choice["finish_reason"]

        content = "".join(reasoning_parts)
        if content:
            content = f"<think>{content}</think>\n\n"
        content += "".join(content_parts)
        return {
            "id": f"chatcmpl-agg-{model_id.replace('/', '-')}-{base_id}-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": final_reason,
            }],
        }
    except Exception:
        log.error("Stream aggregation failed.", exc_info=debug_status)
        return _error_response(
            "Error aggregating stream.",
            500,
            logger_instance=logger,
            debug_flag=debug_status,
        )
