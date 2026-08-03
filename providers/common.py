# providers/common.py
import re

_OT_RE = re.compile(r'\s*/ot\s*')
_NON_STANDARD_PARAMS = (
    "enable_thinking", "thinking_budget", "thinkingConfig", "include_thoughts",
    "thinking_enabled", "thinking", "reasoning_effort",
)
_THINKING_NON_STANDARD_PARAMS = (
    "enable_thinking", "thinking_budget", "thinkingConfig", "include_thoughts",
    "thinking_enabled",
)

def _prepare_common_headers(api_key=None, extra_headers=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers: headers.update(extra_headers)
    return headers

def _resolve_endpoint(md):
    """Return endpoint for a provider, preferring 'endpoint' then 'chat_endpoint'."""
    return md.get("endpoint") or md.get("chat_endpoint")

def _alias_model(payload, md, mid):
    """Set payload["model"] to the alias expected by downstream API."""
    payload["model"] = md.get("original_model_name", mid)

def _flatten_messages_to_string(messages):
    """Convert OpenAI-style messages (possibly with parts) to role+string content messages."""
    final_messages = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        final_messages.append({"role": role, "content": str(content)})
    return final_messages or [{"role": "user", "content": "Hello"}]

def make_openai_compat_builder(
    provider_name,
    _standardize_messages_func, # Pass the function as an argument
    message_provider_name=None,
    string_content=False,
    force_stream=None,
    remove_params=None,
    extra_headers=None,
    postprocess=None,
):
    """
    Factory for simple OpenAI-compatible providers.
    - provider_name: used only for logging/debug.
    - _standardize_messages_func: The function to use for standardizing messages.
    - message_provider_name: which provider name to pass to _standardize_messages; defaults to provider_name.
    - string_content: if True, flattens message content into strings.
    - force_stream: if not None, sets payload["stream"] to this boolean.
    - remove_params: list of parameter keys to remove from the payload.
    - extra_headers: additional headers to add.
    """
    def _builder(data, md, mid):
        payload = data.copy()
        _alias_model(payload, md, mid)
        endpoint = _resolve_endpoint(md)
        if not endpoint:
            raise ValueError(f"Missing endpoint for {provider_name} model {mid}")
        headers = _prepare_common_headers(md.get("api_key"), extra_headers)

        # Messages
        raw_messages = data.get("messages", [])
        if string_content:
            payload["messages"] = _flatten_messages_to_string(raw_messages)
        else:
            mp = message_provider_name or provider_name
            payload["messages"] = _standardize_messages_func(raw_messages, provider_name=mp)

        # Stream behavior
        if force_stream is not None:
            payload["stream"] = bool(force_stream)

        # Remove non-standard parameters that cause issues with OpenAI-compatible providers.
        # Keep the thinking controls only for models that explicitly support them.
        non_standard_params = _NON_STANDARD_PARAMS
        if md.get("supports_thinking") or md.get("supports_reasoning_effort"):
            non_standard_params = _THINKING_NON_STANDARD_PARAMS

        for k in non_standard_params:
            payload.pop(k, None)

        # Remove params if requested
        if remove_params:
            for k in remove_params:
                payload.pop(k, None)

        # Optional postprocess hook for provider-specific tweaks
        if callable(postprocess):
            payload = postprocess(payload, data, md, mid)

        return payload, endpoint, headers
    return _builder
