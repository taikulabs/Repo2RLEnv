from unittest.mock import patch
from types import SimpleNamespace
from repo2rlenv.pipelines import _synthesis
from repo2rlenv.spec.input import LLMSpec


def _llm():
    return LLMSpec(provider="anthropic", model="claude-sonnet-4-6")


def test_synthesize_returns_body_and_cost():
    resp = SimpleNamespace(content="**Title:** X\n## Description\nA real ten word problem statement here now.", cost_usd=0.02)
    with patch.object(_synthesis, "complete", return_value=resp):
        body, cost = _synthesis.synthesize_problem_statement(_llm(), "src")
    assert body.startswith("**Title:**")
    assert cost == 0.02


def test_synthesize_rejects_too_short():
    resp = SimpleNamespace(content="too short", cost_usd=0.01)
    with patch.object(_synthesis, "complete", return_value=resp):
        body, cost = _synthesis.synthesize_problem_statement(_llm(), "src")
    assert body is None


def test_synthesize_swallows_llm_error():
    with patch.object(_synthesis, "complete", side_effect=RuntimeError("500")):
        body, cost = _synthesis.synthesize_problem_statement(_llm(), "src")
    assert body is None
    assert cost == 0.0
