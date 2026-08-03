# run.py
"""Launch the public OmniAI proxy with conservative, local-only defaults."""

import argparse
import logging
import os
import sys
from typing import Optional


PUBLIC_PROVIDERS = frozenset({"openai", "anthropic", "openrouter"})
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default
    return max(value, minimum) if minimum is not None else value


def _env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default
    return max(value, minimum) if minimum is not None else value


def _env_port(name: str, default: int = 8000) -> int:
    value = _env_int(name, default)
    return value if 1 <= value <= 65535 else default


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def _require_proxy_auth_key() -> str:
    """Return the configured proxy key or fail before importing the app."""
    key = os.getenv("PROXY_AUTH_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PROXY_AUTH_KEY is required. Set a non-empty Bearer token before starting OmniAI."
        )
    return key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OmniAI public proxy.")
    parser.add_argument(
        "--host",
        default=os.getenv("OMNIAI_HOST", "127.0.0.1").strip() or "127.0.0.1",
        help="Bind address (default: OMNIAI_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=_env_port("OMNIAI_PORT"),
        help="Bind port (default: OMNIAI_PORT or 8000).",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--debug", dest="debug", action="store_true", help="Enable Flask debug logging.")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false", help="Keep debug logging disabled.")
    parser.set_defaults(debug=None)
    parser.add_argument(
        "--include-usage",
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        help="Include usage in streaming responses (default: OPENAI_INCLUDE_USAGE or false).",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        proxy_auth_key = _require_proxy_auth_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Import only after validating the required secret. This keeps startup
    # fail-closed even if another entry point has different import defaults.
    from app import app

    debug = _env_flag("OMNIAI_DEBUG", False) if args.debug is None else args.debug
    include_usage = (
        args.include_usage
        if args.include_usage is not None
        else _env_flag("OPENAI_INCLUDE_USAGE", False)
    )

    # Text processing is provider-neutral. Provider/model-specific cases do
    # not belong in the public launch configuration.
    strip_characters_enabled = True
    characters_to_strip = []
    replace_characters_enabled = True
    replacements = {
        " / ": {"replacement": "/", "context": "outside_code"},
        "        ": {"replacement": "", "context": "outside_code"},
        "     ": {"replacement": "", "context": "outside_code"},
        ".  ": {"replacement": ". ", "context": "outside_code"},
        "    ": {"replacement": "", "context": "outside_code"},
        " — ": {"replacement": " - ", "context": "outside_code"},
        "**": {"replacement": "*", "context": "outside_code"},
        "###### ": {"replacement": "→ ", "context": "start"},
        "##### ": {"replacement": "→ ", "context": "start"},
        "#### ": {"replacement": "→ ", "context": "start"},
        "### ": {"replacement": "→ ", "context": "start"},
        "## ": {"replacement": "→ ", "context": "start"},
        "# ": {"replacement": "→ ", "context": "start"},
        "- ": {"replacement": "→ ", "context": "start"},
        "，": {"replacement": ", ", "context": "outside_code"},
        "。": {"replacement": ".", "context": "outside_code"},
        "“": '"',
        "”": '"',
        "…": "...",
    }

    strip_system_prompt_characters_enabled = True
    system_prompt_characters_to_strip = [
        "Formatting re-enabled\n",
        " Additional info for this conversation: \n\n",
    ]
    dynamic_system_prompt_strip_enabled = True
    dynamic_system_prompt_patterns_to_strip = [
        r"^Current model: .*\n",
        r"^Current date: .*\n",
    ]

    # Reasoning is handled by the public provider adapters. Do not embed
    # private model IDs in this configuration.
    strip_reasoning_tokens = False
    strip_thought_blocks = False

    # Safe logging defaults: normal operation does not retain payloads or
    # raw stream chunks in the file logger or dashboard stats.
    detailed_openai_stream_logging = _env_flag("OMNI_DETAILED_STREAM_LOGGING", False)
    openai_stream_logging_level = _env_int("OMNI_STREAM_LOGGING_LEVEL", 0, minimum=0)
    stream_debug_sample_n = _env_int("STREAM_DEBUG_SAMPLE_N", 1, minimum=1)
    payload_debug_logging = _env_flag("OMNI_DEBUG_PAYLOADS", False)
    stream_chunk_debug_logging = _env_flag("OMNI_DEBUG_STREAM_CHUNKS", False)

    allow_include_thoughts = _env_flag("ALLOW_INCLUDE_THOUGHTS", False)
    rate_limiting_enabled = True
    rate_limit_interval_seconds = 0.5

    http_connect_timeout = _env_float("HTTP_CONNECT_TIMEOUT", 10.0, minimum=0.0)
    http_read_timeout = _env_float("HTTP_READ_TIMEOUT", 600.0, minimum=0.0)

    # Keep smoothing generic. Public provider/model-specific overrides are
    # intentionally absent so unneeded provider cases cannot leak back in.
    smooth_max_chunk_size = 3
    model_specific_smooth_max_chunk_sizes = {}

    app.config.update(
        DEBUG=debug,
        PROXY_AUTH_KEY=proxy_auth_key,
        SERVER_HOST=args.host,
        SERVER_PORT=args.port,
        SUPPORTED_PROVIDERS=tuple(sorted(PUBLIC_PROVIDERS)),

        STRIP_CHARACTERS_ENABLED=strip_characters_enabled,
        CHARACTERS_TO_STRIP=characters_to_strip,
        REPLACE_CHARACTERS_ENABLED=replace_characters_enabled,
        REPLACEMENTS=replacements,
        STRIP_SYSTEM_PROMPT_CHARACTERS_ENABLED=strip_system_prompt_characters_enabled,
        SYSTEM_PROMPT_CHARACTERS_TO_STRIP=system_prompt_characters_to_strip,
        DYNAMIC_SYSTEM_PROMPT_STRIP_ENABLED=dynamic_system_prompt_strip_enabled,
        STRIP_SYSTEM_PROMPT_PATTERNS_TO_STRIP=dynamic_system_prompt_patterns_to_strip,
        STRIP_REASONING_TOKENS=strip_reasoning_tokens,
        STRIP_THOUGHT_BLOCKS=strip_thought_blocks,

        DETAILED_OPENAI_STREAM_LOGGING=detailed_openai_stream_logging,
        OPENAI_STREAM_LOGGING_LEVEL=openai_stream_logging_level,
        STREAM_DEBUG_SAMPLE_N=stream_debug_sample_n,
        PAYLOAD_DEBUG_LOGGING=payload_debug_logging,
        STREAM_CHUNK_DEBUG_LOGGING=stream_chunk_debug_logging,
        OPENAI_INCLUDE_USAGE=include_usage,

        ALLOW_INCLUDE_THOUGHTS=allow_include_thoughts,
        RATE_LIMITING_ENABLED=rate_limiting_enabled,
        RATE_LIMIT_INTERVAL_SECONDS=rate_limit_interval_seconds,
        HTTP_CONNECT_TIMEOUT=http_connect_timeout,
        HTTP_READ_TIMEOUT=http_read_timeout,
        SMOOTH_MAX_CHUNK_SIZE=smooth_max_chunk_size,
        MODEL_SPECIFIC_SMOOTH_MAX_CHUNK_SIZES=model_specific_smooth_max_chunk_sizes,
        DASHBOARD_PRIVACY_SAFE=True,
    )

    if debug:
        app.logger.setLevel(logging.DEBUG)
        print("OmniAI debug logging enabled.", file=sys.stderr)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=debug,
            use_reloader=False,
            use_debugger=False,
            use_evalex=False,
        )
    except KeyboardInterrupt:
        print("\n[run.py] Server shut down by user.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
