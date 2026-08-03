"""Payload and response adapters for the supported chat protocols."""

from copy import deepcopy
import time
import uuid
from typing import Any, Dict, Iterable, List


def _text_from_content(content: Any) -> str:
    """Return the text portions of OpenAI-style message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def messages_to_anthropic(
    messages: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert standardized chat messages to the Anthropic content shape."""
    converted: List[Dict[str, Any]] = []
    for message in messages or []:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        raw_content = message.get("content", "")
        parts = raw_content if isinstance(raw_content, list) else [
            {"type": "text", "text": str(raw_content)}
        ]
        anthropic_parts: List[Dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                anthropic_parts.append({
                    "type": "text",
                    "text": str(part.get("text", "")),
                })
                continue
            if part_type != "image_url":
                continue

            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = str(image_url.get("url", ""))
            if url.startswith("data:") and ";base64," in url:
                header, encoded = url.split(";base64,", 1)
                media_type = header[5:].strip() or "application/octet-stream"
                anthropic_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded,
                    },
                })
            elif url:
                anthropic_parts.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })

        if anthropic_parts:
            converted.append({"role": role, "content": anthropic_parts})

    return converted or [{
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }]


def _with_reasoning(message: Dict[str, Any], content: Any) -> Any:
    """Put a provider reasoning field in the normalized visible content."""
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if not reasoning:
        return content
    reasoning_text = str(reasoning).strip()
    if not reasoning_text:
        return content
    if not (
        reasoning_text.startswith("<think>")
        and reasoning_text.endswith("</think>")
    ):
        reasoning_text = f"<think>{reasoning_text}</think>"
    visible_content = "" if content is None else str(content)
    return f"{reasoning_text}\n\n{visible_content}"


def normalize_openai_response(
    response_data: Dict[str, Any],
    requested_model: str,
) -> Dict[str, Any]:
    """Normalize an OpenAI-compatible response without exposing side fields."""
    normalized = deepcopy(response_data)
    normalized.setdefault("object", "chat.completion")
    normalized.setdefault("created", int(time.time()))
    normalized.setdefault("model", requested_model)

    choices = normalized.get("choices")
    if not isinstance(choices, list):
        return normalized
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if "content" in message and isinstance(message.get("content"), str):
            message["content"] = _with_reasoning(message, message.get("content"))
        elif message.get("reasoning") or message.get("reasoning_content"):
            message["content"] = _with_reasoning(message, "")
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
    return normalized


def convert_anthropic_response_to_openai(
    response_data: Dict[str, Any],
    requested_model: str,
) -> Dict[str, Any]:
    """Convert an Anthropic message response to Chat Completions JSON."""
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    blocks = response_data.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                value = block.get("thinking") or block.get("text")
                if value:
                    thinking_parts.append(str(value))
            elif block_type == "text" and block.get("text") is not None:
                text_parts.append(str(block.get("text")))

    content = "".join(text_parts)
    if thinking_parts:
        content = f"<think>{''.join(thinking_parts)}</think>\n\n{content}"

    stop_reason = response_data.get("stop_reason")
    finish_reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason, "stop")

    result: Dict[str, Any] = {
        "id": f"chatcmpl-anthropic-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response_data.get("model") or requested_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": response_data.get("role", "assistant"),
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
    }

    usage = response_data.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        result["usage"] = {
            "prompt_tokens": int(input_tokens or 0),
            "completion_tokens": int(output_tokens or 0),
            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
        }
    return result
