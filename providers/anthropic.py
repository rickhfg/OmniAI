# providers/anthropic.py
def _to_anthropic_content(standardized_messages):
    anthropic_messages = []
    for msg in standardized_messages:
        role = msg.get("role")
        content_parts = msg.get("content", [])
        if role not in ["user", "assistant"]:
            continue
        anthropic_content = []
        for part in content_parts:
            part_type = part.get("type")
            if part_type == "text":
                anthropic_content.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url = part.get("image_url", {}).get("url", "")
                if image_url.startswith("data:"):
                    try:
                        header, base64_data = image_url.split(";base64,", 1)
                        mime_type = header.replace("data:", "").strip()
                        anthropic_content.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime_type, "data": base64_data}
                        })
                    except ValueError:
                        pass
                else:
                    anthropic_content.append({
                        "type": "image",
                        "source": {"type": "url", "url": image_url}
                    })
        if anthropic_content:
            anthropic_messages.append({"role": role, "content": anthropic_content})
    return anthropic_messages or [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

def make_anthropic_compat_builder(provider_name, _standardize_messages_func, _debug_func, version_header="2023-06-01", default_max_tokens=4096, extra_headers=None):
    def _builder(data, md, mid):
        _debug_func(f"Building request for {provider_name} (anthropic-compatible, model: {mid})")
        api_key = md.get("api_key")
        if not api_key:
            raise ValueError(f"API key for {provider_name} model '{mid}' is missing.")
        endpoint = md.get("endpoint") or md.get("chat_endpoint")
        if not endpoint:
            raise ValueError(f"Endpoint for {provider_name} model '{mid}' is missing.")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": version_header,
            "Accept": "application/json, text/event-stream",
        }
        if extra_headers:
            headers.update(extra_headers)

        payload = {}
        payload["model"] = md.get("original_model_name", mid)
        payload["stream"] = data.get("stream", False)

        messages = data.get("messages", [])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        if system_msgs:
            system_content = "".join([str(m.get("content", "")) for m in system_msgs])
            payload["system"] = system_content

        standardized_messages = _standardize_messages_func(other_msgs, provider_name="anthropic")
        payload["messages"] = _to_anthropic_content(standardized_messages)

        if "max_tokens" in data:
            try:
                payload["max_tokens"] = int(data["max_tokens"])
            except (ValueError, TypeError):
                raise ValueError("Invalid 'max_tokens' value.")
        else:
            payload["max_tokens"] = default_max_tokens

        # Handle thinking parameters (legacy vs adaptive)
        original_model = str(md.get("original_model_name", mid)).lower()
        supports_reasoning_effort = md.get("supports_reasoning_effort") or any(
            v in original_model for v in ["4-5", "4-6", "4-7", "4-8", "4-9", "4.5", "4.6", "4.7", "4.8", "4.9", "fable", "mythos"]
        )

        if "thinking" in data:
            thinking = data["thinking"]
            if isinstance(thinking, dict):
                if supports_reasoning_effort:
                    payload["thinking"] = {"type": "adaptive"}
                    effort = data.get("reasoning_effort") or thinking.get("effort") or "high"
                    if effort not in ("low", "medium", "high"):
                        effort = "high"
                    payload["output_config"] = {"effort": effort}
                else:
                    payload["thinking"] = thinking.copy()
                    if payload["thinking"].get("type") == "enabled":
                        if "budget_tokens" not in payload["thinking"]:
                            payload["thinking"]["budget_tokens"] = 16000
                        budget = payload["thinking"]["budget_tokens"]
                        if payload["max_tokens"] <= budget:
                            payload["max_tokens"] = budget + 4000
            else:
                payload["thinking"] = thinking
        elif "reasoning_effort" in data or "thinking_budget" in data or data.get("thinking_enabled") is True:
            if supports_reasoning_effort:
                payload["thinking"] = {"type": "adaptive"}
                effort = data.get("reasoning_effort") or "high"
                if effort not in ("low", "medium", "high"):
                    effort = "high"
                payload["output_config"] = {"effort": effort}
            elif md.get("supports_thinking") or any(v in original_model for v in ["claude-3-7", "claude-4"]):
                budget = int(data.get("thinking_budget") or 16000)
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget
                }
                if payload["max_tokens"] <= budget:
                    payload["max_tokens"] = budget + 4000

        if "top_p" in data:
            is_opus_4_8 = "4-8" in original_model or "opus-4-8" in original_model
            if not is_opus_4_8:
                payload["top_p"] = float(data["top_p"])
        if "stop" in data:
            payload["stop_sequences"] = data["stop"] if isinstance(data["stop"], list) else [data["stop"]]

        # Anthropic's Messages API must never receive OpenAI's temperature
        # parameter.  Keep this guard at the final payload boundary as well as
        # omitting it during construction above.
        payload.pop("temperature", None)

        return payload, endpoint, headers
    return _builder

def get_builder(dependencies):
    return make_anthropic_compat_builder(
        "anthropic",
        dependencies["_standardize_messages"],
        dependencies["_debug"],
        version_header="2023-06-01",
        default_max_tokens=4096
    )
