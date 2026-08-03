"""Concurrent readers for OpenAI-compatible and Anthropic SSE streams."""

import json
import threading
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Union

from requests.exceptions import ChunkedEncodingError

from logging_utils import _LogWrapper


def _finalize_reader_thread_processing(
    output_queue: Queue,
    reader_done_event: threading.Event,
    log_wrapper: _LogWrapper,
    source_iterator_for_closing: Optional[Iterator[Any]] = None,
    default_finish_reason_on_abrupt_end: str = "length",
):
    queued_items = [item for item in list(output_queue.queue) if isinstance(item, tuple)]
    already_terminated = any(item[0] in {"FINISH", "ERROR"} for item in queued_items)
    if not reader_done_event.is_set() and not already_terminated:
        output_queue.put(("FINISH", default_finish_reason_on_abrupt_end))

    reader_done_event.set()
    output_queue.put(None)
    log_wrapper.debug("Reader thread finalized.")

    close = getattr(source_iterator_for_closing, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            log_wrapper.warning("Unable to close the upstream stream iterator.")


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("text") is not None
        )
    return "" if value is None else str(value)


def unified_sse_event_reader_to_queue(
    sse_stream_iterator: Iterator[Union[bytes, str]],
    output_queue: Queue,
    reader_done_event: threading.Event,
    logger,
    debug_status: bool,
    log_prefix: str,
    reader_config: Dict[str, Any],
):
    """Read Chat Completions SSE and emit normalized queue events."""
    config = dict(reader_config or {})
    is_cumulative_delta = bool(config.get("is_cumulative_delta", False))
    log = _LogWrapper(
        logger,
        debug_status,
        log_prefix,
        sample_n=config.get("sample_n", 1),
    )
    event_parts: List[str] = []
    previous_content = ""
    in_thinking = False

    def close_thinking():
        nonlocal in_thinking
        if in_thinking:
            output_queue.put(("TEXT", "</think>"))
            in_thinking = False

    def process_event(data_text: str) -> bool:
        nonlocal previous_content, in_thinking
        if data_text == "[DONE]":
            close_thinking()
            output_queue.put(("FINISH", "stop"))
            return True

        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            log.warning("Ignoring malformed SSE event.")
            return False
        if not isinstance(event, dict):
            return False

        if event.get("error") is not None:
            error = event.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("detail") or "Upstream error"
            else:
                message = str(error)
            output_queue.put(("ERROR", str(message)[:500]))
            return True

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}

        role = delta.get("role")
        if role:
            output_queue.put(("ROLE", str(role)))

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        content = delta.get("content")
        if content is None:
            content = delta.get("text")

        if reasoning is not None:
            if not in_thinking:
                output_queue.put(("TEXT", "<think>"))
                in_thinking = True
            output_queue.put(("TEXT", _text_value(reasoning)))

        if content is not None:
            content_text = _text_value(content)
            if is_cumulative_delta:
                if content_text.startswith(previous_content):
                    content_text = content_text[len(previous_content):]
                previous_content = _text_value(content)
            if content_text:
                close_thinking()
                output_queue.put(("TEXT", content_text))

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            close_thinking()
            if finish_reason == "error":
                output_queue.put(("ERROR", "Upstream signaled an error finish reason."))
            else:
                output_queue.put(("FINISH", str(finish_reason)))
            return True
        return False

    try:
        for raw_line in sse_stream_iterator:
            if reader_done_event.is_set():
                break
            if isinstance(raw_line, (bytes, bytearray)):
                line = bytes(raw_line).decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()

            if not line:
                if event_parts:
                    if process_event("\n".join(event_parts)):
                        break
                    event_parts.clear()
                continue
            if line.startswith("data:"):
                event_parts.append(line[5:].lstrip())
            elif line.startswith(":"):
                output_queue.put(("KEEP_ALIVE", line[1:].strip()))
            else:
                # Some compatible gateways return a JSON error without SSE framing.
                if process_event(line):
                    break

        if event_parts and not reader_done_event.is_set():
            process_event("\n".join(event_parts))
    except ChunkedEncodingError:
        close_thinking()
        output_queue.put(("ERROR", "The upstream stream ended unexpectedly."))
    except Exception as exc:
        close_thinking()
        output_queue.put(("ERROR", str(exc)[:500]))
        log.error("Generic SSE reader failed.", exc_info=debug_status)
    finally:
        close_thinking()
        _finalize_reader_thread_processing(
            output_queue,
            reader_done_event,
            log,
            sse_stream_iterator,
            "length",
        )


def _anthropic_event_to_queue(
    event_type: str,
    event_data: Dict[str, Any],
    output_queue: Queue,
    state: Dict[str, bool],
):
    if event_type == "message_start":
        role = event_data.get("message", {}).get("role", "assistant")
        output_queue.put(("ROLE", role))
    elif event_type == "content_block_start":
        block_type = event_data.get("content_block", {}).get("type")
        if block_type == "thinking" and not state["thinking"]:
            state["thinking"] = True
            output_queue.put(("TEXT", "<think>"))
    elif event_type == "content_block_delta":
        delta = event_data.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta" and delta.get("text"):
            output_queue.put(("TEXT", str(delta["text"])))
        elif delta_type == "thinking_delta" and delta.get("thinking"):
            output_queue.put(("TEXT", str(delta["thinking"])))
    elif event_type == "content_block_stop" and state["thinking"]:
        state["thinking"] = False
        output_queue.put(("TEXT", "</think>"))
    elif event_type == "message_delta":
        stop_reason = (event_data.get("delta") or {}).get("stop_reason")
        if stop_reason:
            reason = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }.get(stop_reason, "stop")
            output_queue.put(("FINISH", reason))
    elif event_type == "message_stop":
        output_queue.put(("FINISH", "stop"))
    elif event_type == "error":
        error = event_data.get("error") or {}
        output_queue.put(("ERROR", str(error.get("message", "Upstream error"))[:500]))


def _anthropic_stream_reader(
    anthropic_resp_iterator: Iterator[bytes],
    output_queue: Queue,
    reader_done_event: threading.Event,
    logger,
    debug_status: bool,
    log_prefix: str,
    reader_specific_config: Dict[str, Any],
):
    config = dict(reader_specific_config or {})
    log = _LogWrapper(
        logger,
        debug_status,
        log_prefix,
        sample_n=config.get("sample_n", 1),
    )
    current_event: Optional[str] = None
    current_data: List[str] = []
    state = {"thinking": False}

    def process_current_event():
        nonlocal current_event, current_data
        if not current_data:
            current_event = None
            return
        data_text = "".join(current_data)
        event_name, current_event, current_data = current_event, None, []
        try:
            parsed = json.loads(data_text)
        except json.JSONDecodeError:
            log.warning("Ignoring malformed Anthropic SSE event.")
            return
        if isinstance(parsed, dict):
            event_type = parsed.get("type") or event_name or ""
            _anthropic_event_to_queue(event_type, parsed, output_queue, state)

    try:
        for raw_line in anthropic_resp_iterator:
            if reader_done_event.is_set():
                break
            if isinstance(raw_line, (bytes, bytearray)):
                line = bytes(raw_line).decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()

            if not line:
                process_current_event()
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data.append(line[5:].strip())
            elif line.startswith(":"):
                output_queue.put(("KEEP_ALIVE", line[1:].strip()))

        process_current_event()
    except Exception as exc:
        output_queue.put(("ERROR", str(exc)[:500]))
        log.error("Anthropic SSE reader failed.", exc_info=debug_status)
    finally:
        if state["thinking"]:
            output_queue.put(("TEXT", "</think>"))
            state["thinking"] = False
        _finalize_reader_thread_processing(
            output_queue,
            reader_done_event,
            log,
            anthropic_resp_iterator,
            "length",
        )
