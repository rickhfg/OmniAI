import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- Model Definition ---

@dataclass
class Model:
    """A registered model and the information needed to call its provider."""

    name: str
    provider: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    chat_endpoint: Optional[str] = None
    original_model_name: Optional[str] = None
    provider_options: Optional[Dict[str, Any]] = None
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the model to the dictionary shape consumed by ``app.py``."""
        data = {
            "provider": self.provider,
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "chat_endpoint": self.chat_endpoint,
            "original_model_name": self.original_model_name or self.name,
            "provider_options": self.provider_options,
            **self.flags,
        }
        return {key: value for key, value in data.items() if value is not None}


# --- Model Registration ---

models: Dict[str, Dict[str, Any]] = {}


def add_model(model: Model):
    """Add a model to the central registry."""
    models[model.name] = model.to_dict()


# --- Public credentials ---

def _load_key(env_var_name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(env_var_name, default)


OPENAI_API_KEY = _load_key("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _load_key("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = _load_key("OPENROUTER_API_KEY")


ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


# --- Anthropic ---

# Current Claude 5 IDs are dateless, canonical API IDs.  The effort and
# sampling flags mirror Anthropic's current Messages API behavior: Claude 5
# models use adaptive thinking/effort and should not receive ordinary sampling
# controls through this compatibility layer.
_CLAUDE_5_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_CLAUDE_5_FLAGS = {
    "supports_reasoning_effort": True,
    "supports_vision": True,
    "reasoning_effort_values": _CLAUDE_5_REASONING_EFFORTS,
    "default_reasoning_effort": "high",
    "supports_thinking_disable": True,
    "omit_sampling_parameters": True,
}
_CLAUDE_FABLE_5_FLAGS = {
    **_CLAUDE_5_FLAGS,
    # Fable 5 has always-on thinking and cannot disable it.
    "supports_thinking_disable": False,
}

_ANTHROPIC_MODELS = {
    "claude-opus-5": dict(_CLAUDE_5_FLAGS),
    "claude-sonnet-5": dict(_CLAUDE_5_FLAGS),
    "claude-fable-5": dict(_CLAUDE_FABLE_5_FLAGS),
    # Anthropic documents this pre-4.6 convenience alias for the dated
    # claude-haiku-4-5-20251001 snapshot.
    "claude-haiku-4-5": {"supports_vision": True},
    # Older IDs remain registered for compatibility with existing clients.
    "claude-3-5-sonnet-20241022": {"supports_vision": True},
    "claude-3-7-sonnet-20250219": {"supports_thinking": True, "supports_vision": True},
    "claude-sonnet-4-20250514": {"supports_thinking": True, "supports_vision": True},
    "claude-opus-4-20250514": {"supports_thinking": True, "supports_vision": True},
    "claude-sonnet-4-5-20250929": {"supports_reasoning_effort": True, "supports_vision": True},
    "claude-opus-4-5-20251101": {"supports_reasoning_effort": True, "supports_vision": True},
}

for model_name, flags in _ANTHROPIC_MODELS.items():
    add_model(Model(
        name=model_name,
        provider="anthropic",
        api_key=ANTHROPIC_API_KEY,
        endpoint=ENDPOINTS["anthropic"],
        flags=flags,
    ))


# --- OpenAI ---

_OPENAI_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
_OPENAI_GPT_56_EFFORTS = (*_OPENAI_REASONING_EFFORTS, "max")

_OPENAI_MODELS = {
    # Current frontier models documented for v1/chat/completions.  Pro
    # variants are intentionally absent because their current guidance is
    # Responses-oriented and they are not a safe fit for this proxy route.
    "gpt-5.6-sol": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_GPT_56_EFFORTS,
        "default_reasoning_effort": "medium",
    },
    "gpt-5.6-terra": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_GPT_56_EFFORTS,
        "default_reasoning_effort": "medium",
    },
    "gpt-5.6-luna": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_GPT_56_EFFORTS,
        "default_reasoning_effort": "medium",
    },
    "gpt-5.5": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
    },
    "gpt-5.4": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
    },
    "gpt-5.4-mini": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
    },
    "gpt-5.4-nano": {
        "supports_reasoning_effort": True,
        "supports_vision": True,
        "reasoning_effort_values": _OPENAI_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
    },
    # Older IDs remain registered for compatibility with existing clients.
    "gpt-4.1": {"supports_vision": True},
    "gpt-5": {"supports_reasoning_effort": True, "supports_vision": True},
    "gpt-5.1": {"supports_reasoning_effort": True, "supports_vision": True},
    "gpt-5.2": {"supports_reasoning_effort": True, "supports_vision": True},
    "gpt-5-mini": {"supports_reasoning_effort": True, "supports_vision": True},
    "gpt-5-nano": {"supports_vision": True},
    "gpt-5-chat-latest": {"supports_reasoning_effort": False, "supports_vision": True},
    "gpt-5.1-chat-latest": {"supports_reasoning_effort": False, "supports_vision": True},
    "gpt-5.2-chat-latest": {"supports_reasoning_effort": False, "supports_vision": True},
    "o3": {"supports_reasoning_effort": True, "supports_vision": True},
    "o4-mini": {"supports_reasoning_effort": True, "supports_vision": True},
}

for model_name, flags in _OPENAI_MODELS.items():
    add_model(Model(
        name=model_name,
        provider="openai",
        api_key=OPENAI_API_KEY,
        endpoint=ENDPOINTS["openai"],
        flags=flags,
    ))

# Compatibility with app.py's existing reasoning shortcut.  The upstream
# model name remains the real OpenAI ID; this is not a second credential alias.
add_model(Model(
    name="gpt-5-high",
    provider="openai",
    api_key=OPENAI_API_KEY,
    endpoint=ENDPOINTS["openai"],
    original_model_name="gpt-5",
    flags=_OPENAI_MODELS["gpt-5"],
))


# --- OpenRouter ---

def _configured_model_ids() -> List[str]:
    """Read exact OpenRouter upstream IDs from the public configuration.

    ``OPENROUTER_MODELS`` is a comma- or newline-separated list.  A JSON list
    is also accepted for process managers that prefer structured environment
    values.  ``OPENROUTER_MODEL_IDS`` remains a spelling-compatible fallback.
    Every registered name is the upstream ID itself; no aliases are generated.
    """
    raw = os.getenv("OPENROUTER_MODELS")
    if raw is None:
        raw = os.getenv("OPENROUTER_MODEL_IDS")
    if raw is None:
        raw_values = ["moonshotai/kimi-k3"]
    else:
        stripped = raw.strip()
        if not stripped:
            return []
        raw_values = None
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_values = parsed
        if raw_values is None:
            raw_values = stripped.replace(",", "\n").splitlines()

    model_ids = []
    seen = set()
    for value in raw_values:
        if not isinstance(value, str):
            continue
        model_id = value.strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
    return model_ids


_OPENROUTER_CAPABILITIES = {
    # These exact names and capabilities are present in OpenRouter's current
    # models API.  Arbitrary env-configured IDs remain intentionally opaque.
    "moonshotai/kimi-k3": {"supports_reasoning_effort": True, "supports_vision": True},
    "openai/gpt-5.6-sol": {"supports_reasoning_effort": True, "supports_vision": True},
    "openai/gpt-5.6-terra": {"supports_reasoning_effort": True, "supports_vision": True},
    "openai/gpt-5.6-luna": {"supports_reasoning_effort": True, "supports_vision": True},
    "openai/gpt-5.5": {"supports_reasoning_effort": True, "supports_vision": True},
    "openai/gpt-5.4": {"supports_reasoning_effort": True, "supports_vision": True},
    "anthropic/claude-opus-5": {"supports_reasoning_effort": True, "supports_vision": True},
}


def _openrouter_flags(model_id: str) -> Dict[str, Any]:
    return dict(_OPENROUTER_CAPABILITIES.get(model_id.lower(), {}))


for model_id in _configured_model_ids():
    add_model(Model(
        name=model_id,
        provider="openrouter",
        api_key=OPENROUTER_API_KEY,
        endpoint=ENDPOINTS["openrouter"],
        original_model_name=model_id,
        flags=_openrouter_flags(model_id),
    ))
