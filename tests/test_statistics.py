import pytest

from atlas import (
    Dropout,
    Eventually,
    FaultInjector,
    evaluate_stochastic,
    wilson_lower_bound,
    z_for,
)

from .helpers import Walker, make_walk


# --- Wilson bound ------------------------------------------------------------

def test_wilson_bounds_are_honest_at_the_extremes():
    assert wilson_lower_bound(0, 10) == 0.0            # no evidence of success
    assert wilson_lower_bound(10, 10) < 1.0            # 10/10 is not certainty
    assert wilson_lower_bound(10, 10) > wilson_lower_bound(8, 10)
    assert wilson_lower_bound(80, 100) > wilson_lower_bound(8, 10)  # tighter with more n


def test_wilson_edge_cases():
    assert wilson_lower_bound(0, 0) == 0.0
    assert 0.0 <= wilson_lower_bound(1, 3) <= 1.0


def test_z_for_picks_nearest_level():
    assert z_for(0.95) == pytest.approx(1.6449)
    assert z_for(0.93) == z_for(0.95)  # nearest tabulated


# --- evaluate_stochastic -----------------------------------------------------

def test_deterministic_pass_agent_clears_the_gate():
    result = evaluate_stochastic(
        agent_factory=lambda seed: Walker(),
        scenario=make_walk(goal=5),
        validators=[Eventually("reach", lambda s: s["pos"] >= 5)],
        trials=20,
        required=0.8,
    )
    assert result.pass_rate == 1.0
    assert not result.flaky and result.passed
    assert result.failing_seeds == ()


def test_deterministic_fail_agent_fails_the_gate():
    result = evaluate_stochastic(
        agent_factory=lambda seed: Walker(),
        scenario=make_walk(goal=999, max_steps=5),  # unreachable
        validators=[Eventually("reach", lambda s: s["pos"] >= 999)],
        trials=10,
        required=0.5,
    )
    assert result.pass_rate == 0.0 and not result.passed and not result.flaky


def _flaky_factory(seed):
    # Heavy, seeded command dropout: some seeds reach the goal within budget,
    # some truncate -> genuinely flaky across the seed sweep.
    return FaultInjector(actuator_faults=[Dropout(0.6)], seed=seed).wrap(Walker())


def test_flaky_agent_is_detected_and_gated_out():
    scenario = make_walk(goal=4, max_steps=6)
    validators = [Eventually("reach", lambda s: s["pos"] >= 4)]
    result = evaluate_stochastic(_flaky_factory, scenario, validators, trials=40, required=0.95)
    assert result.flaky                       # mixed outcomes observed
    assert 0 < result.passes < result.trials
    assert not result.passed                  # confidence bound below 0.95
    assert len(result.failing_seeds) > 0
    assert "flaky" in result.summary()


def test_evaluation_is_reproducible():
    scenario = make_walk(goal=4, max_steps=6)
    validators = [Eventually("reach", lambda s: s["pos"] >= 4)]
    a = evaluate_stochastic(_flaky_factory, scenario, validators, trials=25, base_seed=100)
    b = evaluate_stochastic(_flaky_factory, scenario, validators, trials=25, base_seed=100)
    assert a.passes == b.passes and a.failing_seeds == b.failing_seeds


def test_zero_trials_rejected():
    with pytest.raises(ValueError):
        evaluate_stochastic(lambda s: Walker(), make_walk(), [], trials=0)
