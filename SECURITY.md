# Security policy

OmniAI is designed to run as a local proxy. It accepts one static Bearer token for its `/v1` API and then uses server-side provider credentials for upstream calls. Treat the process as a credential-bearing internal service, not as an anonymous public gateway.

## Deployment checklist

- Generate a long, random `PROXY_AUTH_KEY`; the server refuses to start without one.
- Keep `.env` out of version control and out of support bundles. The committed `.env.example` contains placeholders only.
- Store provider keys in the process environment or a secret manager. Do not send them in client payloads or paste them into issues, logs, screenshots, or dashboard exports.
- Bind the application to localhost or a private interface. If remote access is required, put it behind an authenticated, TLS-terminating reverse proxy and restrict source networks.
- Protect `/dashboard` and `/stats` at the reverse proxy. The dashboard is static, while `/stats` requires the proxy Bearer token and returns privacy-safe aggregate metadata.
- Keep `OMNIAI_DEBUG=0` and do not pass `--debug` outside local diagnosis. The checked-in `run.py` defaults debug off, but it is still Flask's development server and should not be internet-facing.
- Keep `OMNI_DEBUG_PAYLOADS=0` and `OMNI_DEBUG_STREAM_CHUNKS=0` in shared or production environments. Prompts, images, provider errors, and generated text can be sensitive.
- Set upstream timeouts and monitor rate limits. The in-memory limiter is not a substitute for an external quota or abuse-control layer.
- Rotate the proxy token and provider keys after suspected exposure. Revoke the upstream key first, then replace the local environment value and restart the process.

## Data handling

Requests are forwarded to the selected upstream provider. OmniAI does not provide durable storage, tenant isolation, encryption at rest, or a retention policy. The in-memory statistics tracker and diagnostic logs can contain model IDs, timing data, and text-derived metadata; assume that debug output is sensitive.

Use only the supported public providers documented in [README.md](README.md). Provider privacy, retention, regional processing, pricing, and acceptable-use terms still apply to every request.
