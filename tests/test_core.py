from atlas import rollout

from .helpers import Walker, make_walk


def test_rollout_terminates_at_goal():
    trace = rollout(Walker(), make_walk(goal=5))
    assert len(trace) == 5
    assert trace.final_state["pos"] == 5
    assert not trace.truncated


def test_rollout_truncates_at_max_steps():
    trace = rollout(Walker(), make_walk(goal=50, max_steps=10))
    assert len(trace) == 10
    assert trace.truncated


def test_trace_states_includes_final():
    trace = rollout(Walker(), make_walk(goal=3))
    assert len(trace.states()) == len(trace) + 1
    assert trace.states()[-1] == trace.final_state


def test_scenario_with_initial_state_is_independent_copy():
    s = make_walk()
    v = s.with_initial_state({**s.initial_state, "pos": 2}, suffix="shifted")
    assert v.initial_state["pos"] == 2
    assert s.initial_state["pos"] == 0
    assert v.name != s.name


def test_observations_are_snapshots_not_references():
    trace = rollout(Walker(), make_walk(goal=3))
    positions = [s.observation.state["pos"] for s in trace]
    assert positions == [0, 1, 2]
