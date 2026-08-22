"""Industrial edge cases: the states a lab demo never produces but a fleet
eventually will — NaN/inf from a bad sensor, overflow from an integrator,
empty/partial state, zero-step and non-terminating episodes, aliasing of the
observation dict, and pathological trace shapes for the compressor.
"""

import math

from atlas import (
    Action,
    Always,
    Eventually,
    Finite,
    Never,
    Observation,
    SafetyEnvelope,
    Scenario,
    Step,
    Trace,
    compress_trace,
    rollout,
)

from .helpers import Walker, make_walk


class Idle:
    """Agent that reads nothing and does nothing — for worlds without pos/goal."""

    def reset(self):
        pass

    def act(self, obs):
        return Action("noop")


def trace_with(states, final=None):
    """Build a Trace whose observed states are `states`."""
    steps = [Step(Observation(dict(s), t), Action("noop")) for t, s in enumerate(states)]
    return Trace("edge", steps=steps, final_state=dict(final if final is not None else states[-1]))


# --- non-finite values: the silent safety gap --------------------------------

def test_nan_slips_past_naive_comparison_but_finite_catches_it():
    t = trace_with([{"v": float("nan")}])
    # The trap: NaN > 10 and NaN < 0 are both False, so a bound/forbidden check
    # waves it through. This documents *why* Finite exists.
    assert Never("over", lambda s: s["v"] > 10).check(t).passed
    # Finite makes the corruption an explicit, located failure.
    assert not Finite("finite", ["v"]).check(t).passed


def test_safety_envelope_flags_non_finite():
    for bad in (float("nan"), float("inf"), float("-inf")):
        t = trace_with([{"x": 1.0}, {"x": bad}])
        res = SafetyEnvelope("env", {"x": (0.0, 10.0)}).check(t)
        assert not res.passed
        assert res.violations[0].timestep == 1


def test_finite_without_keys_scans_all_numeric_values():
    t = trace_with([{"a": 1, "b": 2.0, "label": "ok"}, {"a": float("inf"), "b": 2.0}])
    res = Finite("all").check(t)
    assert not res.passed and res.violations[0].message.startswith("a=")


def test_finite_passes_on_clean_numeric_trace():
    t = trace_with([{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}])
    assert Finite("all").check(t).passed
    assert Finite("some", ["a"]).check(t).passed


# --- missing / empty state ---------------------------------------------------

def test_validators_tolerate_missing_and_empty_state():
    t = trace_with([{}, {"other": 1}])
    # Bounds on an absent key are simply skipped, no KeyError.
    assert SafetyEnvelope("env", {"x": (0, 1)}).check(t).passed
    assert Finite("f", ["x"]).check(t).passed
    # Predicate validators must use .get to stay crash-free on partial state.
    assert Always("nonneg", lambda s: s.get("x", 0) >= 0).check(t).passed


# --- degenerate episode lengths ----------------------------------------------

def test_zero_max_steps_produces_empty_but_valid_trace():
    scenario = make_walk(goal=5, max_steps=0)
    t = rollout(Walker(), scenario)
    assert len(t) == 0 and t.truncated  # never got to act, goal not met
    # Eventually sees only the (initial=final) state and correctly fails.
    assert not Eventually("reach", lambda s: s["pos"] >= 5).check(t).passed


def test_already_satisfied_initial_state_terminates_immediately():
    t = rollout(Walker(), make_walk(goal=0))  # pos already == goal
    assert len(t) == 0 and not t.truncated
    assert Eventually("reach", lambda s: s["pos"] >= 0).check(t).passed


def test_non_terminating_agent_is_truncated_not_hung():
    # Agent that never makes progress: done never holds -> clean truncation.
    stuck = Scenario(
        name="stuck",
        initial_state={"pos": 0, "goal": 5, "fuel": 100},
        dynamics=lambda s, a: dict(s),  # no-op dynamics
        done=lambda s: s["pos"] >= s["goal"],
        max_steps=8,
    )
    t = rollout(Walker(), stuck)
    assert t.truncated and len(t) == 8


# --- overflow to infinity ----------------------------------------------------

def test_integrator_overflow_to_inf_is_caught():
    def blow_up(state, action):
        return {"e": state["e"] * 10.0}

    scenario = Scenario("overflow", {"e": 1e307}, blow_up, lambda s: False, max_steps=5)
    t = rollout(Idle(), scenario)
    assert any(math.isinf(s["e"]) for s in t.states())  # overflowed within budget
    assert not Finite("finite", ["e"]).check(t).passed


# --- observation aliasing ----------------------------------------------------

def test_agent_mutating_its_observation_cannot_corrupt_ground_truth():
    class Vandal:
        def reset(self):
            pass

        def act(self, obs):
            obs.state["pos"] = 9999  # try to poison shared state
            obs.state["injected"] = True
            return Action("move", {"dx": 1})

    t = rollout(Vandal(), make_walk(goal=3, max_steps=10))
    # rollout hands the agent a fresh copy each step, so poisoning it cannot
    # reach the environment: ground truth progressed normally to the goal and
    # the injected key never entered real state.
    assert t.final_state["pos"] == 3 and len(t) == 3
    assert "injected" not in t.final_state


# --- compressor under pathological shapes ------------------------------------

def test_compression_collapses_long_constant_runs():
    # 500 identical no-op steps must run-length-encode to a tiny payload.
    states = [{"x": 0} for _ in range(500)]
    c = compress_trace(trace_with(states))
    # First step is emitted in full; the remaining 499 identical lines collapse
    # into one run-length tag.
    assert "x499" in c.text
    assert c.compressed_tokens * 10 < c.raw_tokens  # >10x on a constant trace


def test_compression_handles_unicode_and_weird_keys():
    t = trace_with([{"θ": 0.0, "key with spaces": 1}, {"θ": 1.5, "key with spaces": 1}])
    c = compress_trace(t)
    assert "θ=1.5" in c.text            # changed key emitted in the delta
    assert c.compressed_tokens >= 1
