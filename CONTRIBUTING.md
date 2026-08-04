# Contributing to OmniAI

Contributions should preserve the public contract: OpenAI, Anthropic, OpenRouter, DeepSeek, and Gemini are the supported providers in this edition. Please keep provider credentials, private endpoints, and real user prompts out of commits and tests.

## Development setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` only as a local template. Tests should run without provider credentials and must not make live upstream requests.

## Before opening a change

Run the same command used by CI:

```powershell
python -m pip check
python -m unittest discover -s tests -p 'test_*.py' -v
```

For request or streaming changes, add a focused regression test covering both streamed and non-streamed behavior where applicable. Preserve byte-level SSE framing, error status codes, provider isolation, and the absence of secrets in logs.

## Provider changes

Use the existing provider boundary:

1. Register an explicit public model alias and its capability flags.
2. Implement or update the provider builder under `providers/`.
3. Normalize authentication and message shape without leaking credentials into responses or diagnostics.
4. Test malformed upstream responses, timeouts, rate limits, and split stream chunks.
5. Update the support matrix, `.env.example`, and security notes only when the provider is intentionally part of the public release.

An adapter being present in the source tree does not make it supported. Do not add undocumented credentials or private upstream services to public examples.

## Documentation and security

Document behavior users can observe, including model aliases, unsupported parameters, streaming differences, and operational limits. Never commit secrets or unredacted provider responses.

Contributions are accepted under the repository's [MIT License](LICENSE).
