# OmniAI

OmniAI is a local Flask proxy that presents OpenAI-compatible Chat Completions and Responses interfaces while adapting requests and streaming responses to configured upstream providers.

This repository is the public edition. Its supported provider contract and source tree are intentionally limited to OpenAI, Anthropic, OpenRouter, DeepSeek, and Gemini.

## What it provides

- `POST /v1/chat/completions` with OpenAI-style JSON and SSE responses.
- `POST /v1/responses` with native OpenAI Responses JSON and semantic SSE for supported upstreams.
- `GET /v1/models` for the model aliases registered by the checkout.
- Provider-specific message and stream normalization for the supported matrix below.
- A small command-line client in `omniaicli.py`.
- Local-only request statistics and a dashboard for development diagnostics.

The proxy authenticates callers with its own Bearer token. Provider API keys stay on the proxy and are never supplied by clients.

## Supported provider matrix

| Provider | Environment variable | Upstream route | Registry examples | Notes |
| --- | --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1/chat/completions`, `https://api.openai.com/v1/responses` | `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4` | Chat Completions is adapted through the existing builder. Responses requests and semantic SSE are passed through natively. |
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1/messages` | `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5` | OpenAI-style chat messages are converted to the Anthropic Messages shape. Anthropic does not document a native `/v1/responses` endpoint, so OmniAI does not emulate one. |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1/chat/completions` | `moonshotai/kimi-k3` by default; `openai/gpt-5.5` and `anthropic/claude-opus-5` are current examples | `OPENROUTER_MODELS` registers exact organization-prefixed IDs. The `kimi-k3` adapter forces the upstream `modal/mxfp4` provider with fallbacks disabled. |
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/chat/completions`, `https://api.deepseek.com/responses` | `deepseek-v4-pro`, `deepseek-v4-flash` | Both V4 models support Chat Completions. Native Responses currently supports only `deepseek-v4-flash`; the proxy preserves its semantic SSE and rejects unsupported stateful fields. |
| Gemini | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview` | Uses Google's official OpenAI compatibility layer. Current 3.6/3.5 models omit deprecated sampling parameters; Gemini 3 reasoning cannot be disabled. |

OpenAI, Anthropic, DeepSeek, and Gemini IDs are defined in `models.py`; OpenRouter IDs are loaded from `OPENROUTER_MODELS` (or its compatibility spelling `OPENROUTER_MODEL_IDS`) and default to the currently listed `moonshotai/kimi-k3`. Use `/v1/models` against the running checkout rather than assuming that every upstream model name is available. For OpenRouter, include the provider namespace, for example `openai/gpt-5.6-sol` rather than bare `gpt-5.6-sol`.

Gemini entries are selected from Google's official [model catalog](https://ai.google.dev/gemini-api/docs/models), [deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations), and [OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai). Responses behavior follows the official [OpenAI streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses), DeepSeek [Responses guide](https://api-docs.deepseek.com/guides/responses_api/) and [API reference](https://api-docs.deepseek.com/api/create-response/). Anthropic support follows its documented [API surface](https://platform.claude.com/docs/en/api/overview) and [OpenAI SDK compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).

## Requirements

- Python 3.12 or 3.13 (the CI matrix is the release-tested baseline).
- A credential for at least one supported provider.
- Network access from the host running the proxy to the selected upstream API.

## Install and run locally

Create an isolated environment and install the pinned direct dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The server does not load `.env` automatically. For a local run, either export the variables in the current shell:

```powershell
$env:PROXY_AUTH_KEY = 'replace-with-a-long-random-value'
$env:OPENAI_API_KEY = 'your-openai-key'
python run.py
```

or use the `python-dotenv` launcher so the copied file is loaded:

```powershell
python -m dotenv run -- python run.py
```

The checked-in `run.py` is a local Flask launcher with debug disabled by default. It requires a non-empty `PROXY_AUTH_KEY` before importing the app, binds to `127.0.0.1:8000` by default, and accepts `OMNIAI_HOST`, `OMNIAI_PORT`, `OMNIAI_DEBUG`, `--host`, `--port`, `--debug`, and `--no-debug`. It is still a development server, so use a WSGI process behind a TLS-terminating reverse proxy for deployment, for example from the repository root:

```powershell
waitress-serve --listen=127.0.0.1:8000 app:app
```

See [SECURITY.md](SECURITY.md) before binding the service beyond localhost.

## API surface

All `/v1` routes below require:

```text
Authorization: Bearer <PROXY_AUTH_KEY>
Content-Type: application/json
```

| Method and path | Auth | Public-edition use |
| --- | --- | --- |
| `GET /` | No | Local liveness check. |
| `GET /v1/models` | Yes | List registered model aliases from the five public providers. |
| `POST /v1/chat/completions` | Yes | Supported request path for OpenAI, Anthropic, OpenRouter, DeepSeek, and Gemini aliases. Set `stream` to `true` for SSE. |
| `POST /v1/responses` | Yes | Native Responses passthrough for OpenAI aliases and `deepseek-v4-flash`. OpenAI stateful continuation fields are preserved; DeepSeek remains stateless. |
| `GET /dashboard` | No | Static local dashboard. It asks for the proxy key in memory before loading statistics. |
| `GET /stats` | Yes | Privacy-safe aggregate development diagnostics. |

### PowerShell request example

This payload intentionally uses the common subset of fields. Provider-specific reasoning and vision features remain model-dependent.

```powershell
$body = @{
  model = 'gpt-5.6-sol'
  messages = @(
    @{ role = 'user'; content = 'Reply with one short sentence.' }
  )
  stream = $false
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/v1/chat/completions' `
  -Method Post `
  -Headers @{ Authorization = "Bearer $env:PROXY_AUTH_KEY" } `
  -ContentType 'application/json' `
  -Body $body
```

For a raw streaming response, use `curl.exe` so PowerShell does not alias `curl` to `Invoke-WebRequest`:

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:PROXY_AUTH_KEY" `
  -H "Content-Type: application/json" `
  -d '{"model":"claude-opus-5","messages":[{"role":"user","content":"Say hello."}],"stream":true}'
```

### Responses API example

The Responses route keeps the upstream schema intact. In particular, its semantic SSE events are not converted into Chat Completions chunks.

```powershell
$body = @{
  model = 'deepseek-v4-flash'
  input = 'Reply with one short sentence.'
  reasoning = @{ effort = 'high' }
  stream = $false
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/v1/responses' `
  -Method Post `
  -Headers @{ Authorization = "Bearer $env:PROXY_AUTH_KEY" } `
  -ContentType 'application/json' `
  -Body $body
```

OpenAI requests may use provider-supported fields such as `previous_response_id` and `store`. DeepSeek's Responses implementation is stateless; OmniAI rejects `previous_response_id`, `conversation`, `store: true`, and `background: true` rather than silently ignoring them.

### CLI example

```powershell
python omniaicli.py status
python omniaicli.py models
python omniaicli.py prompt gpt-5.6-sol "Summarize the benefits of a local proxy." --no-stream
# After adding openai/gpt-5.5 to OPENROUTER_MODELS:
python omniaicli.py prompt openai/gpt-5.5 "Summarize the benefits of a local proxy." --no-stream
```

The CLI reads `OMNIAI_PROXY_URL` and `PROXY_AUTH_KEY` with `python-dotenv`. It talks to the proxy, not directly to a provider.

## Configuration reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `PROXY_AUTH_KEY` | Yes | Static inbound Bearer token for `/v1` routes. Replace the example value. |
| `OPENAI_API_KEY` | If using OpenAI | Server-side OpenAI credential. |
| `ANTHROPIC_API_KEY` | If using Anthropic | Server-side Anthropic credential. |
| `OPENROUTER_API_KEY` | If using OpenRouter | Server-side OpenRouter credential. |
| `DEEPSEEK_API_KEY` | If using DeepSeek | Server-side DeepSeek credential. |
| `GEMINI_API_KEY` | If using Gemini | Server-side Google Gemini API credential. |
| `OPENROUTER_MODELS` | No | Comma-, newline-, or JSON-list-separated exact OpenRouter IDs, such as `openai/gpt-5.5`; defaults to `moonshotai/kimi-k3`. |
| `OMNIAI_PROXY_URL` | CLI only | Base URL used by `omniaicli.py`; defaults to `http://localhost:8000`. |
| `OMNIAI_HOST` / `OMNIAI_PORT` | No | Bind address and port for `run.py`; defaults to `127.0.0.1` and `8000`. |
| `OMNIAI_DEBUG` | No | Enables Flask debug logging when true; keep false outside local diagnosis. |
| `OMNI_DEBUG_PAYLOADS` | No | Opt-in payload diagnostics; defaults to `0`. Enabling it may expose sensitive request content. |
| `OMNI_DEBUG_STREAM_CHUNKS` | No | Opt-in raw stream diagnostics; defaults to `0`. |
| `OMNI_DETAILED_STREAM_LOGGING` / `OMNI_STREAM_LOGGING_LEVEL` | No | Controls optional detailed stream logging. |
| `STREAM_DEBUG_SAMPLE_N` | No | Sample interval for stream logs; larger values are quieter. |
| `OPENAI_INCLUDE_USAGE` | No | Requests usage metadata in supported OpenAI-style streams when available. |
| `ALLOW_INCLUDE_THOUGHTS` | No | Enables provider paths that support exposing thinking details; keep disabled unless needed. |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | No | Upstream timeout values in seconds. |

Do not put provider keys in client requests. The client only needs `PROXY_AUTH_KEY`; the proxy selects the upstream credential from its model registry.

## Architecture and extension path

The request flow is deliberately small:

1. Flask authenticates the caller and validates the JSON body.
2. `models.py` resolves a public model alias to a provider, endpoint, and capability flags.
3. Chat Completions requests pass through a provider builder that converts messages and provider-specific parameters.
4. Chat responses are normalized to OpenAI-style JSON or SSE; native Responses payloads and semantic SSE are forwarded without Chat Completions conversion.

To propose another public provider, follow the same boundary:

1. Add a documented model alias and credential name.
2. Add a provider builder exposing the existing dependency contract.
3. Add focused unit tests for payload conversion, errors, and stream boundaries.
4. Document authentication, data handling, limitations, and the supported model set.
5. Update the public matrix only after review; an adapter present in the source is not automatically a supported provider.

## Limitations

- The model registry is static Python configuration, not a live catalog. Provider model availability, quotas, and feature support can change independently of this repository.
- The compatibility surface is intentionally narrower than the full OpenAI API. This public README guarantees Chat Completions, model listing, and native Responses only for OpenAI aliases and `deepseek-v4-flash`.
- DeepSeek Responses is stateless and does not support image/file inputs. Anthropic, OpenRouter, and Gemini are not exposed through `/v1/responses` unless their upstream services document a compatible native route in the future.
- Streaming normalization is best effort. Upstream differences in usage, finish reasons, tool calls, images, and reasoning fields can remain observable.
- Rate limiting, statistics, and recent logs are in-memory. They reset on restart and are not a shared quota or durable observability system across worker processes.
- `/dashboard` is a public static page, while `/stats` requires the proxy Bearer token. Keep both local or protect them before remote deployment.
- The runner defaults to debug off and conservative diagnostics, but `--debug` or `OMNIAI_DEBUG=1` enables Flask debug logging. The Flask development server is still not a production WSGI process.
- Requests and provider responses may contain sensitive prompts or generated content. Do not enable payload logging in a shared environment.
- Provider terms, retention, pricing, regional availability, and safety behavior remain the responsibility of the account holder and the upstream provider.

## Tests and CI

The test suite uses the Python standard library's `unittest` runner:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

GitHub Actions runs the same suite on the supported Python versions, installs `requirements.txt`, and runs `pip check`. Tests use placeholders and do not make provider API calls.

## License

OmniAI is available under the [MIT License](LICENSE).
