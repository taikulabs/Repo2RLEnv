"""Shared LLM problem-statement synthesis.

Rewrites a fix commit / PR body into a leak-free bug report (symptom only,
no fix approach / file names / SHAs). Used by commit_runtime and pr_to_env.
"""
from __future__ import annotations

import logging
import re

from repo2rlenv.llm import complete
from repo2rlenv.spec.input import LLMSpec

logger = logging.getLogger(__name__)

SYNTH_SYSTEM = """You are writing a GitHub bug report for an AI coding agent to fix.

You are given the commit message (and maybe a linked issue) that FIXED a bug.
Rewrite it into a clear problem statement describing ONLY the observed problem
and expected behavior — the symptom, as a user would report it BEFORE any fix
existed.

STRICT RULES:
- Describe the symptom + expected vs actual behavior. Include a short
  reproduction if one is evident.
- Do NOT describe the solution, the fix approach, or which functions/files/tests
  to change. Do NOT mention "fix", "patch", commit SHAs, PR/issue numbers,
  file names, test names, "Signed-off-by", or changelog bullets.
- Output exactly: a `**Title:**` line, then a `## Description` section. Markdown
  allowed. Nothing else. Keep it concise (2-6 sentences)."""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def synthesize_problem_statement(
    llm: LLMSpec,
    source_text: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str | None, float]:
    """Return (cleaned_body_or_None, cost_usd). None on failure / < 10 words."""
    try:
        resp = complete(
            llm, system=SYNTH_SYSTEM, user=source_text,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as exc:
        logger.warning("synthesis failed: %s", exc)
        return None, 0.0
    body = (resp.content or "").strip()
    if _word_count(body) < 10:
        return None, resp.cost_usd
    return body, resp.cost_usd
