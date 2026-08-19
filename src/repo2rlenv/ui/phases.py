"""Shared bootstrap-phase model used by the bootstrap and generation views."""

from __future__ import annotations

from dataclasses import dataclass

from repo2rlenv.ui.theme import GLYPH

PHASES = ["clone", "pull", "sandbox", "agent", "commit", "push"]

PHASE_GLYPH = {
    "clone": GLYPH.PHASE_CLONE,
    "pull": GLYPH.PHASE_PULL,
    "sandbox": GLYPH.PHASE_SANDBOX,
    "agent": GLYPH.PHASE_AGENT,
    "commit": GLYPH.PHASE_COMMIT,
    "push": GLYPH.PHASE_PUSH,
}


@dataclass
class PhaseState:
    name: str
    status: str = "pending"  # pending | running | done | skipped | failed
    detail: str = ""
    started_at: float = 0.0
    duration_sec: float = 0.0
