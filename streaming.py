"""Protocol-neutral stream smoothing and response chunk generation."""

import os
import time
import uuid
import threading
from collections import deque
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Union

from logging_utils import _LogWrapper
from stream_readers import (
    _anthropic_stream_reader,
    unified_sse_event_reader_to_queue,
)
from text_processing import (
    ContextAwareStreamingProcessor,
    StreamingTextProcessor,
    _has_contextual_replacement_rules,
)


class SseChunker:
    def __init__(self, model_id: str, provider_name_for_id: str):
        self.model_id = model_id
        self.timestamp = int(time.time())
        provider = provider_name_for_id.lower()
        safe_model = model_id.replace("/", "-").replace(":", "-")
        self.message_id_base = f"{provider}-{safe_model}-{self.timestamp}"
        self.response_id = f"chatcmpl-{self.message_id_base}-{uuid.uuid4().hex[:12]}"
        self._sse_choice_prefix = (
            'data: {"id": '
            + _json(self.response_id)
            + ', "object": "chat.completion.chunk", "created": '
            + str(self.timestamp)
            + ', "model": '
            + _json(self.model_id)
            + ', "choices": [{"delta": '
        )
        self._unfinished_choice_suffix = ', "index": 0, "finish_reason": null}]}\n\n'

    def _streaming_chunk(self, delta_payload: Dict[str, Any]) -> str:
        return (
            self._sse_choice_prefix
            + _json(delta_payload)
            + self._unfinished_choice_suffix
        )

    def initial_role_chunk(self, role: str = "assistant") -> str:
        return self._streaming_chunk({"role": role})

    def content_chunk(self, content: str) -> str:
        return self._streaming_chunk({"content": content})

    def final_chunk(self, finish_reason: str) -> str:
        return (
            self._sse_choice_prefix
            + '{}'
            + ', "index": 0, "finish_reason": '
            + _json(finish_reason)
            + '}]}\n\n'
        )

    @staticmethod
    def done_marker() -> str:
        return "data: [DONE]\n\n"


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


class _TextFragmentBuffer:
    """Mutable text buffer that consumes prefixes without copying the tail."""

    def __init__(self):
        self._fragments = deque()
        self._head_offset = 0
        self._length = 0

    def __iadd__(self, text: str):
        if text:
            if not isinstance(text, str):
                raise TypeError("stream fragments must be strings")
            self._fragments.append(text)
            self._length += len(text)
        return self

    def __bool__(self):
        return self._length > 0

    def __len__(self):
        return self._length

    def __str__(self):
        if not self._fragments:
            return ""
        first = self._fragments[0]
        if self._head_offset:
            fragments = iter(self._fragments)
            next(fragments)
            return first[self._head_offset:] + "".join(fragments)
        return "".join(self._fragments)

    def take_prefix(self, count: int) -> str:
        if count <= 0 or count > self._length:
            raise ValueError("prefix length must be within the current buffer")
        parts = []
        remaining = count
        while remaining:
            fragment = self._fragments[0]
            available = len(fragment) - self._head_offset
            take = min(remaining, available)
            start = self._head_offset
            parts.append(fragment[start:start + take])
            self._head_offset += take
            self._length -= take
            remaining -= take
            if self._head_offset == len(fragment):
                self._fragments.popleft()
                self._head_offset = 0
        return "".join(parts)

    def clear(self):
        self._fragments.clear()
        self._head_offset = 0
        self._length = 0


def _stream_debug_enabled() -> bool:
    value = os.getenv("OMNI_DEBUG_STREAM_CHUNKS")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _unified_stream_generator_logic(
    item_queue: Queue,
    reader_done_event: threading.Event,
    sse_chunker: SseChunker,
    text_processor: Union[StreamingTextProcessor, ContextAwareStreamingProcessor],
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    log_prefix_str: str,
    model_config: Optional[Dict[str, Any]] = None,
    req_id: Optional[str] = None,
    stats_tracker: Optional[Any] = None,
) -> Iterator[str]:
    """Turn reader queue events into a normalized OpenAI SSE stream."""
    log = _LogWrapper(logger, debug_status, log_prefix_str)
    chunk_debug_enabled = _stream_debug_enabled()
    output_buffer = _TextFragmentBuffer()
    role_yielded = False
    final_reason_from_provider: Optional[str] = None
    error_from_provider: Optional[str] = None
    total_yielded_len = 0
    in_thinking_block = False
    queue_timeout = float(app_config.get("SMOOTH_QUEUE_TIMEOUT", 0.01))
    configured_size = (
        model_config.get("SMOOTH_MAX_CHUNK_SIZE")
        if model_config
        else None
    )
    max_chunk_size = configured_size
    if max_chunk_size is None:
        max_chunk_size = app_config.get("SMOOTH_MAX_CHUNK_SIZE", 3)
    try:
        max_chunk_size = max(0, int(max_chunk_size))
    except (TypeError, ValueError):
        max_chunk_size = 3
    first_chunk_immediate = bool(app_config.get("SMOOTH_FIRST_CHUNK_IMMEDIATE", False))
    min_delay = 0.005
    model_id = sse_chunker.model_id
    strip_config = app_config.get("STRIP_THOUGHT_BLOCKS", False)
    strip_thought = (
        strip_config is True
        or isinstance(strip_config, list) and model_id in strip_config
    )

    def check_ttft():
        if req_id and total_yielded_len == 0 and stats_tracker is not None:
            try:
                stats_tracker.set_ttft(req_id)
            except Exception:
                pass

    def yield_text(text: str) -> str:
        nonlocal total_yielded_len
        check_ttft()
        total_yielded_len += len(text)
        return sse_chunker.content_chunk(text)

    def drain_buffer(force: bool = False) -> Iterator[str]:
        if not output_buffer:
            return
        if force or max_chunk_size == 0:
            yield yield_text(str(output_buffer))
            output_buffer.clear()
            return
        while len(output_buffer) >= max_chunk_size:
            chunk = output_buffer.take_prefix(max_chunk_size)
            if chunk_debug_enabled:
                log.debug("Yielding a smoothed content fragment.")
            yield yield_text(chunk)
            if total_yielded_len > len(chunk) or not first_chunk_immediate:
                time.sleep(min_delay)

    def process_text(value: Any) -> Iterator[str]:
        nonlocal in_thinking_block, output_buffer
        for fragment in text_processor.process_chunk(str(value)):
            if not fragment:
                continue
            is_start = fragment in {"<think>", "<thought>"}
            is_end = fragment in {"</think>", "</thought>"}
            if strip_thought:
                if is_start:
                    in_thinking_block = True
                    continue
                if is_end:
                    in_thinking_block = False
                    continue
                if in_thinking_block:
                    continue
            if is_start or is_end:
                yield from drain_buffer(force=True)
                yield yield_text(fragment)
                continue
            output_buffer += fragment
            yield from drain_buffer()

    try:
        yield sse_chunker.initial_role_chunk()
        role_yielded = True
        while True:
            try:
                item = item_queue.get(timeout=queue_timeout)
            except Empty:
                if max_chunk_size == 0:
                    yield from drain_buffer(force=True)
                if reader_done_event.is_set() and item_queue.empty():
                    break
                continue

            if item is None:
                break
            item_type, payload = item
            if chunk_debug_enabled:
                log.debug("Dequeued a stream event.")

            if item_type == "TEXT":
                yield from process_text(payload)
            elif item_type == "ROLE":
                yield from drain_buffer(force=True)
                yield sse_chunker.initial_role_chunk(role=str(payload))
                role_yielded = True
            elif item_type == "FINISH":
                final_reason_from_provider = str(payload)[:64]
            elif item_type == "KEEP_ALIVE":
                yield f": {str(payload).replace(chr(10), ' ')}\n\n"
            elif item_type == "ERROR":
                error_from_provider = str(payload)[:500]
            if final_reason_from_provider or error_from_provider:
                break

        yield from drain_buffer(force=True)
        finalize = getattr(text_processor, "finalize", None)
        if callable(finalize):
            for fragment in finalize():
                yield from process_text(fragment)
        yield from drain_buffer(force=True)

        if total_yielded_len == 0 and not error_from_provider:
            yield yield_text(
                "\n[OmniAI: The provider connection closed without content.]"
            )
        final_reason = final_reason_from_provider or "stop"
        if error_from_provider:
            yield yield_text(
                f"\n\n[Provider error]: {' '.join(error_from_provider.split())}"
            )
            final_reason = "error"
        yield sse_chunker.final_chunk(final_reason)
        yield sse_chunker.done_marker()
    except GeneratorExit:
        reader_done_event.set()
        raise
    except Exception:
        reader_done_event.set()
        log.error("Stream generation failed.", exc_info=debug_status)
        try:
            if not role_yielded:
                yield sse_chunker.initial_role_chunk()
            yield yield_text("\n[OmniAI: Stream processing failed.]" )
            yield sse_chunker.final_chunk("error")
            yield sse_chunker.done_marker()
        except Exception:
            log.error("Unable to emit stream failure response.", exc_info=debug_status)
    finally:
        try:
            if stats_tracker is not None and req_id:
                stats_tracker.finish_request(
                    req_id,
                    output_tokens=total_yielded_len // 4,
                )
        except Exception:
            log.error("Unable to finalize request statistics.")


def _create_text_processor(
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    log_prefix: str,
) -> Union[StreamingTextProcessor, ContextAwareStreamingProcessor]:
    if _has_contextual_replacement_rules(app_config):
        return ContextAwareStreamingProcessor(
            app_config,
            logger,
            debug_status,
            f"{log_prefix}_Proc",
        )
    return StreamingTextProcessor(app_config, logger, debug_status, f"{log_prefix}_Proc")


def _create_generic_stream_generator(
    source_iterator: Iterator[Any],
    model_id: str,
    provider_name_for_id: str,
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    reader_thread_target: Callable,
    reader_specific_config: Optional[Dict[str, Any]] = None,
    is_smoothing_dynamic_sse: bool = False,
    model_config: Optional[Dict[str, Any]] = None,
    req_id: Optional[str] = None,
    stats_tracker: Optional[Any] = None,
) -> Iterator[str]:
    base_prefix = f"{provider_name_for_id.upper()}_STREAM ({model_id})"
    reader_config = dict(reader_specific_config or {})
    reader_config.setdefault(
        "sample_n",
        int(app_config.get("STREAM_DEBUG_SAMPLE_N", 1) or 1),
    )
    queue: Queue = Queue()
    done = threading.Event()
    provider_id = (
        f"smooth-{provider_name_for_id}"
        if is_smoothing_dynamic_sse
        else provider_name_for_id
    )
    chunker = SseChunker(model_id, provider_id)
    processor = _create_text_processor(app_config, logger, debug_status, base_prefix)
    thread = threading.Thread(
        target=reader_thread_target,
        args=(
            source_iterator,
            queue,
            done,
            logger,
            debug_status,
            f"{base_prefix}_Reader",
            reader_config,
        ),
        daemon=True,
    )
    thread.start()
    return _unified_stream_generator_logic(
        queue,
        done,
        chunker,
        processor,
        app_config,
        logger,
        debug_status,
        f"{base_prefix}_Generator",
        model_config=model_config,
        req_id=req_id,
        stats_tracker=stats_tracker,
    )


def dynamic_smooth_stream_generator(
    sse_stream_iterator: Iterator[Union[bytes, str]],
    model_id: str,
    provider_name: str,
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    reader_specific_config_param: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    req_id: Optional[str] = None,
    stats_tracker: Optional[Any] = None,
) -> Iterator[str]:
    reader_config = dict(reader_specific_config_param or {})
    reader_config.update({"provider_name": provider_name, "model_id": model_id})
    return _create_generic_stream_generator(
        sse_stream_iterator,
        model_id,
        provider_name,
        app_config,
        logger,
        debug_status,
        unified_sse_event_reader_to_queue,
        reader_specific_config=reader_config,
        model_config=model_config,
        req_id=req_id,
        stats_tracker=stats_tracker,
    )


def anthropic_stream_converter(
    anthropic_resp_iterator: Iterator[bytes],
    model_id: str,
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    req_id: Optional[str] = None,
    stats_tracker: Optional[Any] = None,
) -> Iterator[str]:
    return _create_generic_stream_generator(
        anthropic_resp_iterator,
        model_id,
        "anthropic",
        app_config,
        logger,
        debug_status,
        _anthropic_stream_reader,
        req_id=req_id,
        stats_tracker=stats_tracker,
    )


def _pseudo_stream_non_stream_response(
    non_stream_json_response: Dict[str, Any],
    model_id: str,
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    tokens_per_second: Optional[float] = None,
) -> Iterator[str]:
    """Expose a completed chat response through the streaming envelope."""
    chunker = SseChunker(model_id, "pseudo")
    processor = _create_text_processor(app_config, logger, debug_status, "PseudoStream")
    try:
        yield chunker.initial_role_chunk()
        choices = non_stream_json_response.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            content = f"<think>{str(reasoning).strip()}</think>\n\n{content}"
        processed = "".join(processor.process_chunk(str(content)))
        delay = 0.01
        if tokens_per_second and tokens_per_second > 0:
            delay = 3 / (float(tokens_per_second) * 4)
        for start in range(0, len(processed), 3):
            if start:
                time.sleep(max(0.001, delay))
            yield chunker.content_chunk(processed[start:start + 3])
        finalize = getattr(processor, "finalize", None)
        if callable(finalize):
            for fragment in finalize():
                if fragment:
                    yield chunker.content_chunk(fragment)
        yield chunker.final_chunk(choice.get("finish_reason", "stop"))
        yield chunker.done_marker()
    except Exception:
        log = _LogWrapper(logger, debug_status, "PseudoStream")
        log.error("Pseudo-stream generation failed.", exc_info=debug_status)
        yield chunker.final_chunk("error")
        yield chunker.done_marker()


def _create_generic_stream_response(
    source_iterator: Iterator[Any],
    model_id: str,
    provider_name_for_id: str,
    app_config: Dict[str, Any],
    logger,
    debug_status: bool,
    reader_thread_target: Callable,
    reader_specific_config: Optional[Dict[str, Any]] = None,
    is_smoothing_dynamic_sse: bool = False,
    model_config: Optional[Dict[str, Any]] = None,
    req_id: Optional[str] = None,
    stats_tracker: Optional[Any] = None,
) -> Iterator[str]:
    return _create_generic_stream_generator(
        source_iterator,
        model_id,
        provider_name_for_id,
        app_config,
        logger,
        debug_status,
        reader_thread_target,
        reader_specific_config=reader_specific_config,
        is_smoothing_dynamic_sse=is_smoothing_dynamic_sse,
        model_config=model_config,
        req_id=req_id,
        stats_tracker=stats_tracker,
    )
