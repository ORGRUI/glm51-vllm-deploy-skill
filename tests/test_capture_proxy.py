from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PROXY_PATH = ROOT / "merge-quant-serve" / "scripts" / "capture_proxy.py"

spec = importlib.util.spec_from_file_location("capture_proxy", CAPTURE_PROXY_PATH)
assert spec is not None
assert spec.loader is not None
capture_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_proxy)


class FakeRequest:
    method = "POST"
    path = "/v1/chat/completions"
    headers = {"content-type": "application/json"}


def test_normalizes_messages_prompt_and_history_tool_call_arguments():
    payload = {
        "messages": [{"tool_calls": [{"function": {"arguments": {"city": "北京"}}}]}],
        "prompt": [{"tool_calls": [{"function": {"arguments": ["a", "b"]}}]}],
        "history": [{"tool_calls": [{"function": {"arguments": {"ok": True}}}]}],
    }

    count = capture_proxy.normalize_tool_call_arguments_in_json(payload)

    assert count == 3
    assert payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"北京"}'
    )
    assert payload["prompt"][0]["tool_calls"][0]["function"]["arguments"] == (
        '["a","b"]'
    )
    assert payload["history"][0]["tool_calls"][0]["function"]["arguments"] == (
        '{"ok":true}'
    )


def test_keeps_existing_string_tool_call_arguments():
    payload = {
        "messages": [{"tool_calls": [{"function": {"arguments": '{"city":"北京"}'}}]}],
        "history": [{"tool_calls": [{"function": {"arguments": "already-string"}}]}],
    }

    count = capture_proxy.normalize_tool_call_arguments_in_json(payload)

    assert count == 0
    assert (
        payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        == '{"city":"北京"}'
    )
    assert (
        payload["history"][0]["tool_calls"][0]["function"]["arguments"]
        == "already-string"
    )


def test_rewrite_defaults_max_tokens_and_disables_thinking_when_omitted():
    payload = {
        "model": "glm51",
        "messages": [{"role": "user", "content": "1+1"}],
    }

    body, summary = capture_proxy.rewrite_request_body(
        FakeRequest(),
        json.dumps(payload).encode("utf-8"),
        force_temperature=None,
        default_max_tokens=8192,
        mask_replacement_char=False,
        normalize_tool_call_arguments=False,
        disable_thinking=True,
    )

    rewritten = json.loads(body)
    assert rewritten["max_tokens"] == 8192
    assert rewritten["chat_template_kwargs"]["enable_thinking"] is False
    assert summary["forwarded_max_tokens"] == 8192
    assert summary["forwarded_enable_thinking"] is False


def test_rewrite_keeps_explicit_enable_thinking():
    payload = {
        "model": "glm51",
        "messages": [{"role": "user", "content": "1+1"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }

    body, summary = capture_proxy.rewrite_request_body(
        FakeRequest(),
        json.dumps(payload).encode("utf-8"),
        force_temperature=None,
        default_max_tokens=8192,
        mask_replacement_char=False,
        normalize_tool_call_arguments=False,
        disable_thinking=True,
    )

    rewritten = json.loads(body)
    assert rewritten["chat_template_kwargs"]["enable_thinking"] is True
    assert rewritten["max_tokens"] == 8192
    assert "forwarded_enable_thinking" not in summary
