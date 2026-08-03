# providers/openrouter.py
from .common import make_openai_compat_builder


def _postprocess(payload, data, md, mid):
    upstream_model = str(md.get("original_model_name") or mid)
    configured_provider = md.get("openrouter_provider")
    if configured_provider is None:
        provider_options = md.get("provider_options")
        if isinstance(provider_options, dict):
            configured_provider = provider_options.get("openrouter_provider")

    # An optional per-model provider policy can be supplied by the registry.
    # Keep the existing Kimi policy for the default public model.
    if configured_provider:
        payload["provider"] = configured_provider
    elif upstream_model.lower() == "moonshotai/kimi-k3":
        payload["provider"] = {
            "only": ["modal/mxfp4"],
            "allow_fallbacks": False,
        }

    return payload


def get_builder(dependencies):
    return make_openai_compat_builder(
        provider_name="openrouter",
        _standardize_messages_func=dependencies["_standardize_messages"],
        message_provider_name="openrouter",
        string_content=False,
        postprocess=_postprocess,
    )
