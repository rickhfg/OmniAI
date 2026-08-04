"""DeepSeek's official OpenAI-compatible Chat Completions adapter."""


def get_builder(openai_compatible_builder):
    """Bind the shared compatibility builder to the public DeepSeek provider."""

    def build(data, model_definition, model_id):
        return openai_compatible_builder(
            data, model_definition, model_id, "deepseek"
        )

    return build
