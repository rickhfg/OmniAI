import json
import unittest
from copy import deepcopy
from unittest.mock import call, patch

from flask import Response

import app as app_module
import omniaicli as cli_module
from models import _openrouter_flags, models as registered_models
from response_utils import aggregate_stream_response
from streaming import (
    SseChunker,
    _TextFragmentBuffer,
    _pseudo_stream_non_stream_response,
    anthropic_stream_converter,
    dynamic_smooth_stream_generator,
)
from text_processing import _process_non_stream_text_content


PUBLIC_PROVIDERS = {"openai", "anthropic", "openrouter", "deepseek", "gemini"}
TEST_PROXY_KEY = "unit-test-proxy-key"


class CurrentModelRegistryTests(unittest.TestCase):
    def test_current_openai_and_anthropic_chat_models_have_verified_capabilities(self):
        for model_id in (
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
        ):
            definition = registered_models[model_id]
            self.assertEqual("openai", definition["provider"])
            self.assertTrue(definition["supports_vision"])
            self.assertTrue(definition["supports_reasoning_effort"])
            self.assertIn("reasoning_effort_values", definition)

        opus = registered_models["claude-opus-5"]
        self.assertEqual("anthropic", opus["provider"])
        self.assertTrue(opus["supports_vision"])
        self.assertTrue(opus["supports_reasoning_effort"])
        self.assertEqual("high", opus["default_reasoning_effort"])

    def test_current_deepseek_and_google_documented_gemini_models_are_registered(self):
        for model_id in ("deepseek-v4-pro", "deepseek-v4-flash"):
            definition = registered_models[model_id]
            self.assertEqual("deepseek", definition["provider"])
            self.assertTrue(definition["supports_reasoning_effort"])
            self.assertTrue(definition["supports_thinking_disable"])
            self.assertTrue(definition["string_message_content"])

        for model_id in (
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
        ):
            definition = registered_models[model_id]
            self.assertEqual("gemini", definition["provider"])
            self.assertTrue(definition["supports_vision"])
            self.assertTrue(definition["supports_reasoning_effort"])

    def test_verified_openrouter_examples_have_vision_and_reasoning_flags(self):
        for model_id in (
            "moonshotai/kimi-k3",
            "openai/gpt-5.6-sol",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.6-luna",
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "anthropic/claude-opus-5",
        ):
            flags = _openrouter_flags(model_id)
            self.assertTrue(flags["supports_vision"])
            self.assertTrue(flags["supports_reasoning_effort"])

    def test_responses_only_pro_variants_are_not_registered(self):
        self.assertNotIn("gpt-5.5-pro", registered_models)
        self.assertNotIn("gpt-5.4-pro", registered_models)


class PublicCliPayloadTests(unittest.TestCase):
    def test_deepseek_v4_cli_payload_preserves_sampling_when_thinking_is_disabled(self):
        payload = cli_module.construct_payload(
            "deepseek-v4-flash",
            [{"role": "user", "content": "hello"}],
            False,
            "off",
        )
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        self.assertEqual(1, payload["temperature"])
        self.assertEqual(1, payload["top_p"])

    def test_latest_gemini_cli_payload_omits_deprecated_sampling_parameters(self):
        payload = cli_module.construct_payload(
            "gemini-3.6-flash",
            [{"role": "user", "content": "hello"}],
            True,
            "low",
        )
        self.assertEqual("low", payload["reasoning_effort"])
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("top_k", payload)


class _IdentityProcessor:
    def process_chunk(self, text):
        return [text]

    def finalize(self):
        return []


class _HttpResponse:
    """Small requests.Response substitute used by all offline route tests."""

    def __init__(self, payload=None, *, status_code=200, lines=()):
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload
        self._lines = list(lines)
        self.content = json.dumps(payload or {}).encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def iter_content(self, chunk_size=4096):
        del chunk_size
        yield self.content

    def close(self):
        self.closed = True


def _sse_payloads(raw):
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    elif not isinstance(raw, str):
        raw = "".join(raw)
    payloads = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payloads.append(json.loads(line[6:]))
    return payloads


def _sse_text(payloads):
    return "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )


class TextFragmentBufferTests(unittest.TestCase):
    def test_prefixes_cross_fragment_boundaries_without_losing_text(self):
        buffer = _TextFragmentBuffer()
        buffer += "ab"
        buffer += "cdef"
        buffer += "g"

        self.assertEqual(7, len(buffer))
        self.assertEqual("abc", buffer.take_prefix(3))
        self.assertEqual("defg", str(buffer))
        self.assertEqual("de", buffer.take_prefix(2))
        self.assertEqual("fg", str(buffer))

        buffer.clear()
        self.assertFalse(buffer)
        self.assertEqual("", str(buffer))

    def test_invalid_prefix_lengths_are_rejected(self):
        buffer = _TextFragmentBuffer()
        buffer += "abc"

        with self.assertRaises(ValueError):
            buffer.take_prefix(0)
        with self.assertRaises(ValueError):
            buffer.take_prefix(4)


class StreamingTextInvariantTests(unittest.TestCase):
    def test_non_stream_processor_strips_thought_blocks_without_touching_plain_text(self):
        config = {
            "STRIP_THOUGHT_BLOCKS": True,
            "REPLACE_CHARACTERS_ENABLED": False,
            "STRIP_CHARACTERS_ENABLED": False,
        }
        self.assertEqual(
            "answer",
            _process_non_stream_text_content(
                "<think>private reasoning</think>answer",
                config,
            ),
        )
        config["STRIP_THOUGHT_BLOCKS"] = False
        self.assertEqual(
            "<think>private reasoning</think>answer",
            _process_non_stream_text_content(
                "<think>private reasoning</think>answer",
                config,
            ),
        )

    def test_sse_chunker_emits_valid_openai_envelopes_and_done_marker(self):
        chunker = SseChunker("demo/model", "openrouter")

        role = json.loads(chunker.initial_role_chunk()[6:])
        content = json.loads(chunker.content_chunk("olá")[6:])
        final = json.loads(chunker.final_chunk("stop")[6:])

        self.assertEqual(chunker.response_id, role["id"])
        self.assertEqual("demo/model", content["model"])
        self.assertEqual("olá", content["choices"][0]["delta"]["content"])
        self.assertEqual("stop", final["choices"][0]["finish_reason"])
        self.assertEqual("data: [DONE]\n\n", chunker.done_marker())

    def test_pseudo_stream_preserves_text_chunks_and_finish_protocol(self):
        response = {
            "choices": [
                {"message": {"content": "abcdefg"}, "finish_reason": "stop"}
            ]
        }

        with (
            patch("streaming._create_text_processor", return_value=_IdentityProcessor()),
            patch("streaming.time.sleep") as sleep_mock,
        ):
            chunks = list(
                _pseudo_stream_non_stream_response(
                    response,
                    "test-model",
                    {},
                    logger=None,
                    debug_status=False,
                )
            )

        payloads = _sse_payloads(chunks)
        content = [
            choice["delta"]["content"]
            for payload in payloads
            for choice in payload.get("choices", [])
            if "content" in choice.get("delta", {})
        ]

        self.assertEqual(["abc", "def", "g"], content)
        self.assertEqual([call(0.01), call(0.01)], sleep_mock.call_args_list)
        self.assertEqual("stop", payloads[-1]["choices"][0]["finish_reason"])
        self.assertEqual("data: [DONE]\n\n", chunks[-1])

    def test_generic_stream_preserves_content_and_reports_upstream_finish(self):
        lines = [
            b'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            b"\n",
            b'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n',
            b"\n",
            b'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        config = {
            "SMOOTH_QUEUE_TIMEOUT": 0.01,
            "SMOOTH_MAX_CHUNK_SIZE": 3,
            "SMOOTH_FIRST_CHUNK_IMMEDIATE": True,
            "STREAM_CHUNK_DEBUG_LOGGING": False,
            "REPLACE_CHARACTERS_ENABLED": False,
            "STRIP_CHARACTERS_ENABLED": False,
        }

        chunks = list(
            dynamic_smooth_stream_generator(
                iter(lines),
                "or-unit",
                "openrouter",
                config,
                logger=None,
                debug_status=False,
            )
        )
        payloads = _sse_payloads(chunks)

        self.assertEqual("hello", _sse_text(payloads))
        self.assertEqual("stop", payloads[-1]["choices"][0]["finish_reason"])
        self.assertEqual("data: [DONE]\n\n", chunks[-1])


class AggregationTests(unittest.TestCase):
    def test_content_and_reasoning_are_aggregated_in_order(self):
        events = [
            'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"reasoning":"r1"},"finish_reason":null}]}\n\n',
            'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"reasoning_content":"r2","content":"ab"},"finish_reason":null}]}\n\n',
            'data: {"id":"stream-1","model":"or-unit","choices":[{"delta":{"content":"c"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        result = aggregate_stream_response(
            Response(events), logger=None, debug_status=False
        )

        self.assertEqual("or-unit", result["model"])
        self.assertEqual("stop", result["choices"][0]["finish_reason"])
        self.assertEqual(
            "<think>r1r2</think>\n\nabc",
            result["choices"][0]["message"]["content"],
        )


class PublicPayloadBuilderTests(unittest.TestCase):
    def setUp(self):
        self._saved_config = {
            "PROXY_AUTH_KEY": app_module.app.config.get("PROXY_AUTH_KEY"),
            "OPENAI_INCLUDE_USAGE": app_module.app.config.get("OPENAI_INCLUDE_USAGE"),
            "PAYLOAD_DEBUG_LOGGING": app_module.app.config.get("PAYLOAD_DEBUG_LOGGING"),
        }
        app_module.app.config.update(
            PROXY_AUTH_KEY=TEST_PROXY_KEY,
            OPENAI_INCLUDE_USAGE=False,
            PAYLOAD_DEBUG_LOGGING=False,
        )

    def tearDown(self):
        app_module.app.config.update(self._saved_config)

    @staticmethod
    def _request(model_id, *, stream=False):
        return {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 32,
            "temperature": 0.25,
            "top_p": 0.9,
            "stream": stream,
        }

    @staticmethod
    def _models():
        return {
            "gpt-unit": {
                "provider": "openai",
                "api_key": "unit-openai-key",
                "endpoint": "http://127.0.0.1/openai",
                "original_model_name": "gpt-upstream",
            },
            "claude-unit": {
                "provider": "anthropic",
                "api_key": "unit-anthropic-key",
                "endpoint": "http://127.0.0.1/anthropic",
                "original_model_name": "claude-upstream",
            },
            "or-unit": {
                "provider": "openrouter",
                "api_key": "unit-openrouter-key",
                "endpoint": "http://127.0.0.1/openrouter",
                "original_model_name": "provider/model",
            },
            "deepseek-unit": {
                "provider": "deepseek",
                "api_key": "unit-deepseek-key",
                "endpoint": "http://127.0.0.1/deepseek",
                "original_model_name": "deepseek-v4-pro",
                "supports_reasoning_effort": True,
                "reasoning_effort_values": ("low", "medium", "high", "xhigh", "max"),
                "supports_thinking_disable": True,
                "omit_sampling_parameters": True,
                "string_message_content": True,
            },
            "gemini-unit": {
                "provider": "gemini",
                "api_key": "unit-gemini-key",
                "endpoint": "http://127.0.0.1/gemini",
                "original_model_name": "gemini-3.6-flash",
                "supports_reasoning_effort": True,
                "reasoning_effort_values": ("minimal", "low", "medium", "high"),
                "omit_sampling_parameters": True,
            },
        }

    def _invoke_builder(self, provider_name, model_id, request, model):
        captured = {}
        fake_response = _HttpResponse({"choices": []})

        def fake_post(endpoint, *, json_payload, headers, stream=False, **kwargs):
            captured.update(
                endpoint=endpoint,
                payload=json_payload,
                headers=headers,
                stream=stream,
                extra=kwargs,
            )
            return fake_response

        builder = app_module.PROVIDERS[provider_name]
        with app_module.app.app_context(), patch.object(
            app_module, "_http_post", side_effect=fake_post
        ):
            result = builder(deepcopy(request), deepcopy(model), model_id)
            if isinstance(result, tuple):
                payload, endpoint, headers, *_ = result
                return payload, endpoint, headers

            self.assertIsInstance(result, Response)
            result.get_data()
            return captured["payload"], captured["endpoint"], captured["headers"]

    def test_openai_payload_builder_sets_alias_auth_and_standard_messages(self):
        request = self._request("gpt-unit")
        payload, endpoint, headers = self._invoke_builder(
            "openai", "gpt-unit", request, self._models()["gpt-unit"]
        )

        self.assertEqual("gpt-upstream", payload["model"])
        self.assertEqual("http://127.0.0.1/openai", endpoint)
        self.assertEqual("Bearer unit-openai-key", headers["Authorization"])
        self.assertEqual(0.25, payload["temperature"])
        self.assertEqual("hello", payload["messages"][-1]["content"][0]["text"])
        self.assertEqual("gpt-unit", request["model"])

    def test_anthropic_payload_builder_omits_temperature_and_uses_anthropic_auth(self):
        request = self._request("claude-unit", stream=True)
        payload, endpoint, headers = self._invoke_builder(
            "anthropic", "claude-unit", request, self._models()["claude-unit"]
        )

        self.assertEqual("claude-upstream", payload["model"])
        self.assertEqual(True, payload["stream"])
        self.assertEqual(32, payload["max_tokens"])
        self.assertNotIn("temperature", payload)
        self.assertEqual("http://127.0.0.1/anthropic", endpoint)
        self.assertEqual("unit-anthropic-key", headers["x-api-key"])
        self.assertEqual("2023-06-01", headers["anthropic-version"])
        self.assertNotIn("Authorization", headers)
        self.assertEqual("Be concise.", payload["system"])

    def test_current_anthropic_payload_uses_adaptive_effort_without_sampling_controls(self):
        current_model = self._models()["claude-unit"] | {
            "original_model_name": "claude-opus-5",
            "supports_reasoning_effort": True,
            "supports_thinking_disable": True,
            "reasoning_effort_values": ("low", "medium", "high", "xhigh", "max"),
            "default_reasoning_effort": "high",
            "omit_sampling_parameters": True,
        }

        default_payload, _, _ = self._invoke_builder(
            "anthropic", "claude-opus-5", self._request("claude-opus-5"), current_model
        )
        self.assertNotIn("temperature", default_payload)
        self.assertNotIn("top_p", default_payload)
        self.assertNotIn("thinking", default_payload)

        off_request = self._request("claude-opus-5")
        off_request["reasoning_effort"] = "off"
        off_payload, _, _ = self._invoke_builder(
            "anthropic", "claude-opus-5", off_request, current_model
        )
        self.assertEqual({"type": "disabled"}, off_payload["thinking"])
        self.assertNotIn("output_config", off_payload)
        self.assertNotIn("temperature", off_payload)
        self.assertNotIn("top_p", off_payload)

    def test_current_openai_payload_uses_registered_reasoning_effort_values(self):
        current_model = self._models()["gpt-unit"] | {
            "original_model_name": "gpt-5.4",
            "supports_reasoning_effort": True,
            "reasoning_effort_values": ("none", "low", "medium", "high", "xhigh"),
            "default_reasoning_effort": "none",
        }
        request = self._request("gpt-5.4")
        request["reasoning_effort"] = "unsupported"

        payload, _, _ = self._invoke_builder(
            "openai", "gpt-5.4", request, current_model
        )
        self.assertEqual("none", payload["reasoning_effort"])

    def test_openrouter_payload_builder_sets_alias_auth_and_standard_messages(self):
        request = self._request("or-unit")
        payload, endpoint, headers = self._invoke_builder(
            "openrouter", "or-unit", request, self._models()["or-unit"]
        )

        self.assertEqual("provider/model", payload["model"])
        self.assertEqual("http://127.0.0.1/openrouter", endpoint)
        self.assertEqual("Bearer unit-openrouter-key", headers["Authorization"])
        self.assertEqual(0.25, payload["temperature"])
        self.assertEqual("hello", payload["messages"][-1]["content"][0]["text"])

    def test_deepseek_v4_payload_uses_text_messages_and_thinking_controls(self):
        request = self._request("deepseek-unit")
        request["reasoning_effort"] = "max"
        payload, endpoint, headers = self._invoke_builder(
            "deepseek", "deepseek-unit", request, self._models()["deepseek-unit"]
        )

        self.assertEqual("deepseek-v4-pro", payload["model"])
        self.assertEqual("http://127.0.0.1/deepseek", endpoint)
        self.assertEqual("Bearer unit-deepseek-key", headers["Authorization"])
        self.assertEqual("max", payload["reasoning_effort"])
        self.assertEqual("hello", payload["messages"][-1]["content"])
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

        off_request = self._request("deepseek-unit")
        off_request["reasoning_effort"] = "off"
        off_payload, _, _ = self._invoke_builder(
            "deepseek", "deepseek-unit", off_request, self._models()["deepseek-unit"]
        )
        self.assertEqual({"type": "disabled"}, off_payload["thinking"])
        self.assertNotIn("reasoning_effort", off_payload)
        self.assertEqual(0.25, off_payload["temperature"])
        self.assertEqual(0.9, off_payload["top_p"])

        invalid_request = self._request("deepseek-unit")
        invalid_request["reasoning_effort"] = []
        with self.assertRaisesRegex(ValueError, "Invalid 'reasoning_effort'"):
            self._invoke_builder(
                "deepseek", "deepseek-unit", invalid_request, self._models()["deepseek-unit"]
            )

    def test_gemini_payload_uses_google_openai_compat_contract(self):
        request = self._request("gemini-unit")
        request["reasoning_effort"] = "medium"
        request["top_k"] = 40
        payload, endpoint, headers = self._invoke_builder(
            "gemini", "gemini-unit", request, self._models()["gemini-unit"]
        )

        self.assertEqual("gemini-3.6-flash", payload["model"])
        self.assertEqual("http://127.0.0.1/gemini", endpoint)
        self.assertEqual("Bearer unit-gemini-key", headers["Authorization"])
        self.assertEqual("medium", payload["reasoning_effort"])
        self.assertEqual("hello", payload["messages"][-1]["content"][0]["text"])
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("top_k", payload)

        invalid_request = self._request("gemini-unit")
        invalid_request["reasoning_effort"] = "none"
        with self.assertRaisesRegex(ValueError, "cannot be disabled"):
            self._invoke_builder(
                "gemini", "gemini-unit", invalid_request, self._models()["gemini-unit"]
            )

        invalid_request["reasoning_effort"] = []
        with self.assertRaisesRegex(ValueError, "Invalid 'reasoning_effort'"):
            self._invoke_builder(
                "gemini", "gemini-unit", invalid_request, self._models()["gemini-unit"]
            )

    def test_provider_registry_contains_only_public_builders(self):
        self.assertEqual(PUBLIC_PROVIDERS, set(app_module.PROVIDERS))


class PublicApiRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self._saved_config = {
            "PROXY_AUTH_KEY": app_module.app.config.get("PROXY_AUTH_KEY"),
            "RATE_LIMITING_ENABLED": app_module.app.config.get("RATE_LIMITING_ENABLED"),
            "PAYLOAD_DEBUG_LOGGING": app_module.app.config.get("PAYLOAD_DEBUG_LOGGING"),
            "STREAM_CHUNK_DEBUG_LOGGING": app_module.app.config.get("STREAM_CHUNK_DEBUG_LOGGING"),
        }
        app_module.app.config.update(
            PROXY_AUTH_KEY=TEST_PROXY_KEY,
            RATE_LIMITING_ENABLED=False,
            PAYLOAD_DEBUG_LOGGING=False,
            STREAM_CHUNK_DEBUG_LOGGING=False,
        )
        app_module._MODEL_CONFIG_CACHE = None

    def tearDown(self):
        app_module.app.config.update(self._saved_config)
        app_module._MODEL_CONFIG_CACHE = None

    @staticmethod
    def _auth_headers():
        return {"Authorization": f"Bearer {TEST_PROXY_KEY}"}

    @staticmethod
    def _model():
        return {
            "provider": "openrouter",
            "api_key": "unit-openrouter-key",
            "endpoint": "http://127.0.0.1/openrouter",
            "original_model_name": "provider/model",
        }

    def _request_json(self, model="or-unit", *, stream=False):
        return {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "stream": stream,
        }

    def test_models_requires_bearer_auth(self):
        self.assertEqual(401, self.client.get("/v1/models").status_code)
        self.assertEqual(
            401,
            self.client.get(
                "/v1/models", headers={"Authorization": "Basic unit-test"}
            ).status_code,
        )
        self.assertEqual(
            401,
            self.client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong-key"}
            ).status_code,
        )
        authorized = self.client.get("/v1/models", headers=self._auth_headers())
        self.assertEqual(200, authorized.status_code)
        self.assertEqual("list", authorized.get_json()["object"])

    def test_models_endpoint_exposes_only_the_five_public_provider_families(self):
        response = self.client.get("/v1/models", headers=self._auth_headers())
        self.assertEqual(200, response.status_code)
        entries = response.get_json()["data"]
        self.assertTrue(entries)
        owned_by = {entry["owned_by"] for entry in entries}
        self.assertEqual(PUBLIC_PROVIDERS, owned_by)
        self.assertTrue(all(entry["object"] == "model" for entry in entries))

    def test_non_stream_chat_uses_mocked_transport_and_returns_provider_json(self):
        upstream = {
            "id": "chat-unit",
            "model": "provider/model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }
        fake_response = _HttpResponse(upstream)
        with (
            patch.object(app_module, "models", {"or-unit": self._model()}),
            patch.object(app_module, "_MODEL_CONFIG_CACHE", None),
            patch.object(
                app_module, "_make_request", return_value=(fake_response, None)
            ) as request_mock,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._request_json(),
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("hello", response.get_json()["choices"][0]["message"]["content"])
        request_mock.assert_called_once()
        self.assertFalse(request_mock.call_args.kwargs["stream"])
        self.assertEqual(
            "provider/model",
            request_mock.call_args.args[1]["model"],
        )
        self.assertEqual(
            "Bearer unit-openrouter-key",
            request_mock.call_args.args[2]["Authorization"],
        )

    def test_stream_chat_uses_mocked_transport_and_preserves_sse_protocol(self):
        lines = [
            b'data: {"id":"stream-unit","model":"provider/model","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            b"\n",
            b'data: {"id":"stream-unit","model":"provider/model","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n',
            b"\n",
            b'data: {"id":"stream-unit","model":"provider/model","choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        fake_response = _HttpResponse(lines=lines)
        with (
            patch.object(app_module, "models", {"or-unit": self._model()}),
            patch.object(app_module, "_MODEL_CONFIG_CACHE", None),
            patch.object(
                app_module, "_make_request", return_value=(fake_response, None)
            ) as request_mock,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._request_json(stream=True),
            )

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.mimetype.startswith("text/event-stream"))
        payloads = _sse_payloads(response.data)
        self.assertEqual("hello", _sse_text(payloads))
        self.assertEqual("stop", payloads[-1]["choices"][0]["finish_reason"])
        self.assertTrue(response.data.endswith(b"data: [DONE]\n\n"))
        request_mock.assert_called_once()
        self.assertTrue(request_mock.call_args.kwargs["stream"])

    def test_anthropic_stream_converter_maps_text_and_stop_reason_offline(self):
        events = [
            b"event: message_start\n",
            b'data: {"type":"message_start","message":{"role":"assistant"}}\n',
            b"\n",
            b"event: content_block_delta\n",
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n',
            b"\n",
            b"event: message_delta\n",
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n',
            b"\n",
        ]
        config = {
            "SMOOTH_QUEUE_TIMEOUT": 0.01,
            "SMOOTH_MAX_CHUNK_SIZE": 3,
            "SMOOTH_FIRST_CHUNK_IMMEDIATE": True,
            "STREAM_CHUNK_DEBUG_LOGGING": False,
            "REPLACE_CHARACTERS_ENABLED": False,
            "STRIP_CHARACTERS_ENABLED": False,
        }

        chunks = list(
            anthropic_stream_converter(
                iter(events),
                "claude-unit",
                config,
                logger=None,
                debug_status=False,
            )
        )
        payloads = _sse_payloads(chunks)

        self.assertEqual("hello", _sse_text(payloads))
        self.assertEqual("stop", payloads[-1]["choices"][0]["finish_reason"])
        self.assertEqual("data: [DONE]\n\n", chunks[-1])

    def test_unknown_model_returns_client_error_without_transport_call(self):
        with (
            patch.object(app_module, "models", {"or-unit": self._model()}),
            patch.object(app_module, "_MODEL_CONFIG_CACHE", None),
            patch.object(app_module, "_make_request") as request_mock,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._request_json(model="does-not-exist"),
            )

        self.assertEqual(404, response.status_code)
        self.assertEqual(404, response.get_json()["error"]["code"])
        request_mock.assert_not_called()

    def test_upstream_error_is_returned_as_json_without_a_network_call(self):
        error_response = Response(
            json.dumps(
                {
                    "error": {
                        "message": "synthetic upstream failure",
                        "type": "proxy_error",
                        "code": 502,
                    }
                }
            ),
            status=502,
            mimetype="application/json",
        )
        with (
            patch.object(app_module, "models", {"or-unit": self._model()}),
            patch.object(app_module, "_MODEL_CONFIG_CACHE", None),
            patch.object(
                app_module, "_make_request", return_value=(None, error_response)
            ),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._request_json(),
            )

        self.assertEqual(502, response.status_code)
        self.assertEqual("synthetic upstream failure", response.get_json()["error"]["message"])

    def test_invalid_content_type_is_rejected_before_transport(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers=self._auth_headers(),
            data="not-json",
            content_type="text/plain",
        )
        self.assertEqual(415, response.status_code)


class StatsPrivacyTests(unittest.TestCase):
    def test_stats_never_expose_request_or_response_text(self):
        tracker = app_module.StatsTracker()
        input_secret = "unit-private-input-7e9f"
        output_secret = "unit-private-output-3a1c"
        req_id = tracker.start_request(
            "or-unit",
            input_tokens=2,
            input_text=input_secret,
            last_input=input_secret,
        )
        tracker.finish_request(req_id, output_tokens=3, output_text=output_secret)

        for stats in (tracker.get_stats(), tracker.get_stats(since=0)):
            serialized = json.dumps(stats)
            self.assertNotIn(input_secret, serialized)
            self.assertNotIn(output_secret, serialized)
            self.assertNotIn("input_text", serialized)
            self.assertNotIn("last_input", serialized)
            self.assertNotIn("output_text", serialized)

        history = tracker.get_stats()["request_history"]
        self.assertEqual(1, len(history))
        self.assertEqual("or-unit", history[0]["model"])
        self.assertNotIn("input_text", history[0])
        self.assertNotIn("output_text", history[0])

    def test_stats_route_is_privacy_safe_for_the_same_tracker(self):
        original_key = app_module.app.config.get("PROXY_AUTH_KEY")
        app_module.app.config["PROXY_AUTH_KEY"] = TEST_PROXY_KEY
        tracker = app_module.StatsTracker()
        input_secret = "unit-route-private-input"
        output_secret = "unit-route-private-output"
        req_id = tracker.start_request(
            "or-unit", input_text=input_secret, last_input=input_secret
        )
        tracker.finish_request(req_id, output_text=output_secret)

        original_tracker = app_module.stats_tracker
        app_module.stats_tracker = tracker
        try:
            response = app_module.app.test_client().get(
                "/stats", headers={"Authorization": f"Bearer {TEST_PROXY_KEY}"}
            )
        finally:
            app_module.stats_tracker = original_tracker
            app_module.app.config["PROXY_AUTH_KEY"] = original_key

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertNotIn(input_secret, body)
        self.assertNotIn(output_secret, body)
        self.assertNotIn("input_text", body)
        self.assertNotIn("output_text", body)

    def test_disabled_payload_logging_does_not_format_sensitive_payload(self):
        fake_response = _HttpResponse({"ok": True})
        secret = "unit-log-private-value"
        with app_module.app.app_context():
            original_flag = app_module.app.config.get("PAYLOAD_DEBUG_LOGGING")
            app_module.app.config["PAYLOAD_DEBUG_LOGGING"] = False
            try:
                with (
                    patch.object(app_module, "_http_post", return_value=fake_response),
                    patch.object(app_module, "_debug") as debug_mock,
                ):
                    result, error = app_module._make_request(
                        "http://127.0.0.1/unit",
                        {"secret": secret},
                        {"Content-Type": "application/json"},
                        timeout=1,
                    )
            finally:
                app_module.app.config["PAYLOAD_DEBUG_LOGGING"] = original_flag

        self.assertIs(fake_response, result)
        self.assertIsNone(error)
        self.assertNotIn(secret, repr(debug_mock.call_args_list))


if __name__ == "__main__":
    unittest.main()
