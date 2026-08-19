"""Anthropic Claude Code OAuth path in `repo2rlenv.llm`.

The anthropic provider bypasses LiteLLM and calls the Messages API directly
with a Bearer OAuth token. These tests pin that contract: correct auth
headers, the Claude Code identity system block, a bare model id, and a clean
failure when no token is resolved. Network is fully mocked.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

import repo2rlenv.llm as llm
from repo2rlenv.spec.input import LLMSpec


def test_anthropic_uses_oauth_bearer_no_apikey() -> None:
    spec = LLMSpec(provider="anthropic", model="claude-sonnet-4-6")
    token = "sk-ant-oat01-abc"
    fake_response = json.dumps(
        {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
    ).encode()

    with mock.patch.dict(
        os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": token}, clear=True
    ), mock.patch("urllib.request.urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = fake_response
        resp = llm.complete(spec, system="be helpful", user="hello", max_tokens=16)

    # (a) response parsing
    assert resp.content == "hi"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 3
    assert resp.cost_usd == 0.0

    # (b) OAuth headers, no x-api-key
    req = m.call_args.args[0]
    assert req.get_header("Authorization") == f"Bearer {token}"
    assert req.get_header("Anthropic-beta") == "oauth-2025-04-20"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    assert req.get_header("X-api-key") is None

    # (c) request body shape
    body = json.loads(req.data)
    assert body["system"][0] == {
        "type": "text",
        "text": "You are Claude Code, Anthropic's official CLI for Claude.",
    }
    assert body["system"][1]["text"] == "be helpful"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_missing_oauth_token_raises() -> None:
    spec = LLMSpec(provider="anthropic", model="claude-sonnet-4-6")
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
            llm.complete(spec, system="be helpful", user="hello", max_tokens=16)
