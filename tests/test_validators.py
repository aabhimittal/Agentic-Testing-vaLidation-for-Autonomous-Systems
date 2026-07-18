from atlas import (
    Always,
    Eventually,
    Never,
    RespondsWithin,
    SafetyEnvelope,
    rollout,
)

from .helpers import Walker, make_walk


def trace_for(goal=5, **kw):
    return rollout(Walker(), make_walk(goal=goal, **kw))


def test_always_passes_when_invariant_holds():
    result = Always("fuel_positive", lambda s: s["fuel"] > 0).check(trace_for())
    assert result.passed


def test_always_reports_each_violating_timestep():
    result = Always("pos_below_3", lambda s: s["pos"] < 3).check(trace_for(goal=5))
    assert not result.passed
    # states with pos 3, 4, 5 violate -> timesteps 3, 4, 5
    assert [v.timestep for v in result.violations] == [3, 4, 5]


def test_never_flags_forbidden_state():
    result = Never("at_4", lambda s: s["pos"] == 4).check(trace_for(goal=5))
    assert not result.passed
    assert result.violations[0].timestep == 4


def test_eventually_passes_when_goal_reached():
    assert Eventually("arrived", lambda s: s["pos"] >= 5).check(trace_for()).passed


def test_eventually_fails_on_truncated_trace():
    trace = trace_for(goal=50, max_steps=10)
    result = Eventually("arrived", lambda s: s["pos"] >= 50).check(trace)
    assert not result.passed
    assert result.violations[0].timestep == -1


def test_responds_within_passes_inside_window():
    # trigger: pos >= 2 (first true at t=2); response: pos >= 4 (t=4) -> window 2 ok
    v = RespondsWithin("resp", lambda s: s["pos"] >= 2, lambda s: s["pos"] >= 4, window=2)
    assert v.check(trace_for(goal=5)).passed


def test_responds_within_fails_outside_window():
    v = RespondsWithin("resp", lambda s: s["pos"] == 0, lambda s: s["pos"] >= 4, window=2)
    result = v.check(trace_for(goal=5))
    assert not result.passed
    assert result.violations[0].timestep == 0


def test_responds_within_grace_when_scenario_completes_first():
    # Trigger fires on the final delivered state; window extends past a
    # completed (non-truncated) trace -> inconclusive, not a failure.
    v = RespondsWithin("resp", lambda s: s["pos"] == 5, lambda s: False, window=3)
    assert v.check(trace_for(goal=5)).passed


def test_responds_within_no_grace_on_truncated_trace():
    trace = trace_for(goal=50, max_steps=10)
    v = RespondsWithin("resp", lambda s: s["pos"] == 9, lambda s: False, window=5)
    assert not v.check(trace).passed


def test_safety_envelope_bounds():
    env = SafetyEnvelope("env", {"pos": (0, 3), "missing_key": (0, 1)})
    result = env.check(trace_for(goal=5))
    assert not result.passed
    assert all("pos" in v.message for v in result.violations)


def test_safety_envelope_open_sided():
    env = SafetyEnvelope("env", {"fuel": (0, None)})
    assert env.check(trace_for()).passed
