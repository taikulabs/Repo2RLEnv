"""LiteLLM wrapper — single entry point across providers, with cost tracking.

The pipelines call `complete(input, prompt)`; we resolve credentials from
either the LLMSpec hint or the provider-default env var, dispatch, then use
LiteLLM's `completion_cost()` to attach a USD estimate. The Anthropic
provider is the exception: it bypasses LiteLLM and calls the Messages API
directly using a Claude Code OAuth token (Bearer auth), with cost 0.0.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from repo2rlenv.auth import resolve_claude_oauth_token, resolve_llm_api_key
from repo2rlenv.spec.input import LLMSpec

logger = logging.getLogger(__name__)

_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# Models that reject any `temperature` value (forced default). Add patterns
# as new releases land. Probed empirically via scripts/probe_llm_routes.py.
_NO_TEMPERATURE_RE = re.compile(
    r"(claude-opus-4-7|claude-opus-4-8|gpt-5(\.|-|$)|gpt-6|o1-|o3-|o4-)",
    re.IGNORECASE,
)


def _supports_temperature(model: str) -> bool:
    return _NO_TEMPERATURE_RE.search(model) is None


@dataclass(slots=True)
class LLMResponse:
    content: str
    usage: dict | None = None
    cost_usd: float = 0.0  # cost of THIS call, in USD (best-effort)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _is_failover_eligible(exc: BaseException) -> bool:
    """True for transient provider errors worth retrying on a fallback model.

    LiteLLM raises specific subclasses; we match on class names so we don't
    have to import the symbols (some live in nested submodules and shift
    between versions).
    """
    # HTTPError is a subclass of URLError, so check it first for the status code.
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    # Retry on: 5xx upstream, rate limits, network blips, timeouts.
    # Don't retry on: 4xx bad-request, auth errors, not-found, content filter.
    return name in {
        "InternalServerError",  # 5xx incl. Anthropic 529 Overloaded
        "RateLimitError",  # 429
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "APIError",  # generic upstream error
    }


def _do_complete_anthropic_oauth(
    spec: LLMSpec,
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    """Anthropic path via Claude Code OAuth, bypassing LiteLLM.

    The LiteLLM SDK forces the credential into the `x-api-key` header, which
    Anthropic rejects for OAuth tokens (litellm issue #19618). We therefore
    call the Messages API directly with `Authorization: Bearer <token>` and
    the OAuth beta header. urllib errors propagate so `complete()` can fail
    over. The token is never logged.
    """
    token = resolve_claude_oauth_token(spec.oauth_token_env)
    if token is None:
        raise RuntimeError(
            "no Claude Code OAuth token resolved for provider 'anthropic'. "
            "Set CLAUDE_CODE_OAUTH_TOKEN (run 'claude setup-token')."
        )

    # The first system block MUST be the Claude Code identity verbatim;
    # Anthropic rejects OAuth calls otherwise.
    system_blocks: list[dict] = [{"type": "text", "text": _CLAUDE_CODE_IDENTITY}]
    if system:
        system_blocks.append({"type": "text", "text": system})

    body: dict = {
        "model": spec.model,  # bare id, not spec.qualified_name
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user}],
    }
    if _supports_temperature(spec.model):
        body["temperature"] = temperature

    url = spec.endpoint or "https://api.anthropic.com/v1/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=spec.timeout_sec) as resp:
        payload = json.loads(resp.read())

    content = "".join(
        b.get("text", "")
        for b in payload.get("content", [])
        if b.get("type") == "text"
    )
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("input_tokens", 0) or 0
    completion_tokens = usage.get("output_tokens", 0) or 0

    # Subscription OAuth usage has no per-token billing, so cost is always 0.0.
    # This means the bootstrap max_spend_usd cap never trips for Claude —
    # intentional for OAuth/subscription auth.
    return LLMResponse(
        content=content,
        usage=usage or None,
        cost_usd=0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _do_complete(
    spec: LLMSpec,
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    """One non-fallback chat-completion call. Internal helper for `complete()`."""
    if spec.provider == "anthropic":
        return _do_complete_anthropic_oauth(
            spec,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    import litellm  # type: ignore[import-untyped]

    api_key = resolve_llm_api_key(spec.provider, spec.api_key_env)
    if api_key is None:
        raise RuntimeError(
            f"no API key resolved for provider {spec.provider!r}. "
            f"Set {spec.api_key_env or 'the provider-default env var'}."
        )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": spec.qualified_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "timeout": spec.timeout_sec,
    }
    # Newer reasoning-focused models (Opus 4.7+, GPT-5+) reject `temperature`.
    if _supports_temperature(spec.model):
        kwargs["temperature"] = temperature
    if spec.endpoint:
        kwargs["api_base"] = spec.endpoint

    if spec.provider == "huggingface" and spec.endpoint is None:
        kwargs.setdefault("api_base", "https://router.huggingface.co/v1")

    response = litellm.completion(**kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""

    usage_obj = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage_obj is not None:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

    cost_usd = 0.0
    try:
        cost_usd = float(litellm.completion_cost(completion_response=response))
    except Exception as exc:
        logger.debug("completion_cost failed for %s: %s", spec.qualified_name, exc)

    return LLMResponse(
        content=content,
        usage=dict(usage_obj) if usage_obj else None,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def complete(
    spec: LLMSpec,
    *,
    system: str | None = None,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    _depth: int = 0,
) -> LLMResponse:
    """Single chat-completion call with automatic fallback on transient errors.

    Calls `spec.qualified_name`. On 5xx / 429 / network / timeout errors, if
    `spec.fallback` is set, retries with the fallback model recursively (up
    to 3 levels deep, then re-raises). 4xx errors (bad model, auth, etc.)
    are NOT retried — those signal config bugs, not transient failures.

    Honors `LLMSpec.endpoint` for self-hosted backends.
    """
    try:
        return _do_complete(
            spec,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        if _depth >= 3 or spec.fallback is None or not _is_failover_eligible(exc):
            raise
        logger.warning(
            "primary LLM %s failed with %s; falling back to %s",
            spec.qualified_name,
            type(exc).__name__,
            spec.fallback.qualified_name,
        )
        return complete(
            spec.fallback,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            _depth=_depth + 1,
        )
