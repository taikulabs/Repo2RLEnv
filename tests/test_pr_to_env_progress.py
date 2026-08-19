from __future__ import annotations


def test_phases_shared_model_importable():
    from repo2rlenv.ui.phases import PHASES, PHASE_GLYPH, PhaseState

    assert PHASES == ["clone", "pull", "sandbox", "agent", "commit", "push"]
    assert set(PHASE_GLYPH) == set(PHASES)  # one icon per phase name
    st = PhaseState(name="clone")
    assert st.status == "pending" and st.duration_sec == 0.0


def test_bootstrap_view_reuses_shared_phases():
    from repo2rlenv.ui import phases
    from repo2rlenv.ui.views import bootstrap

    assert bootstrap.PhaseState is phases.PhaseState
    assert bootstrap._PHASES is phases.PHASES
