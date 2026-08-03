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

_ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022": {"supports_vision": True},
    "claude-3-7-sonnet-20250219": {"supports_thinking": True, "supports_vision": True},
    "claude-sonnet-4-20250514": {"supports_thinking": True, "supports_vision": True},
    "claude-opus-4-20250514": {"supports_thinking": True, "supports_vision": True},
    "claude-sonnet-4-5-20250929": {"supports_reasoning_effort": True, "supports_vision": True},
    "claude-opus-4-5-20251101": {"supports_reasoning_effort": True, "supports_vision": True},
    "claude-fable-5": {"supports_reasoning_effort": True, "supports_vision": True},
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

_OPENAI_MODELS = {
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


def _openrouter_flags(model_id: str) -> Dict[str, Any]:
    # Preserve the known capability and routing behavior for the default
    # upstream while leaving arbitrary configured IDs untouched.
    if model_id.lower() == "moonshotai/kimi-k3":
        return {"supports_reasoning_effort": True, "supports_vision": True}
    return {}


for model_id in _configured_model_ids():
    add_model(Model(
        name=model_id,
        provider="openrouter",
        api_key=OPENROUTER_API_KEY,
        endpoint=ENDPOINTS["openrouter"],
        original_model_name=model_id,
        flags=_openrouter_flags(model_id),
    ))
