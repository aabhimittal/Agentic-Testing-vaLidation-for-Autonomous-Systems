import pytest

from atlas import (
    BudgetExceeded,
    ScenarioCost,
    TokenBudget,
    compress_trace,
    estimate_tokens,
    rollout,
    select_scenarios,
)

from .helpers import Walker, make_walk


def test_estimate_tokens():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("") == 1  # never zero


def test_budget_charge_and_ledger():
    b = TokenBudget(limit=100)
    b.charge(30, label="a")
    b.charge(30, label="a")
    assert b.spent == 60
    assert b.remaining == 40
    assert b.ledger == {"a": 60}


def test_budget_hard_limit():
    b = TokenBudget(limit=10)
    with pytest.raises(BudgetExceeded):
        b.charge(11, label="too_big")
    assert b.spent == 0  # failed charge must not partially apply


def test_compress_trace_shrinks_and_is_self_contained():
    trace = rollout(Walker(), make_walk(goal=15, max_steps=40))
    c = compress_trace(trace)
    assert c.compressed_tokens < c.raw_tokens
    assert c.ratio > 1.3
    # First step emitted in full: all initial keys present in the text.
    assert "goal=15" in c.text
    assert "fuel=100" in c.text
    assert "delta-encoded" in c.text


def test_compress_trace_run_length_encodes_repeats():
    # Never-done scenario: walker reaches the goal then emits "stop" with an
    # unchanged state every step -> identical delta lines collapse via xN.
    scenario = make_walk(goal=3, max_steps=12)
    scenario.done = lambda s: False
    trace = rollout(Walker(), scenario)
    text = compress_trace(trace).text
    assert "[] -> stop x8" in text


def test_compress_empty_trace():
    trace = rollout(Walker(), make_walk(goal=0))  # done immediately
    c = compress_trace(trace)
    assert len(trace) == 0
    assert "final[" in c.text


def test_select_scenarios_orders_by_value_density():
    costs = [
        ScenarioCost("cheap_flaky", expected_tokens=100, historical_failure_rate=0.5),
        ScenarioCost("pricey_flaky", expected_tokens=1000, historical_failure_rate=0.6),
        ScenarioCost("cheap_stable", expected_tokens=100, historical_failure_rate=0.01),
    ]
    chosen = select_scenarios(costs, TokenBudget(limit=250))
    assert chosen == ["cheap_flaky", "cheap_stable"]  # pricey one never fits


def test_select_scenarios_skips_then_fits_later():
    costs = [
        ScenarioCost("huge", expected_tokens=900, historical_failure_rate=0.9),
        ScenarioCost("small", expected_tokens=50, historical_failure_rate=0.1),
    ]
    # huge has best density but doesn't fit; small still gets selected
    chosen = select_scenarios(costs, TokenBudget(limit=100))
    assert chosen == ["small"]
