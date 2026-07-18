from atlas import Never, ScenarioFuzzer

from .helpers import Walker, make_walk


def brittle_validator():
    # Walker starting left of 0 is "unsafe": pretend negative positions are a
    # restricted zone. Fuzzing pos around 0 should find a negative start.
    return Never("no_negative_pos", lambda s: s["pos"] < 0)


def test_fuzzer_finds_counterexample():
    fuzzer = ScenarioFuzzer(mutable_keys=["pos"], max_iterations=100, seed=3)
    report = fuzzer.falsify(Walker(), make_walk(goal=5), [brittle_validator()])
    assert report.falsified
    cx = report.counterexamples[0]
    assert cx.validator == "no_negative_pos"
    assert cx.scenario.initial_state["pos"] < 0
    assert "pos" in cx.mutated_keys


def test_fuzzer_is_deterministic_under_seed():
    def run():
        f = ScenarioFuzzer(mutable_keys=["pos"], max_iterations=100, seed=3)
        return f.falsify(Walker(), make_walk(goal=5), [brittle_validator()])

    a, b = run(), run()
    assert a.iterations == b.iterations
    assert a.counterexamples[0].scenario.initial_state == b.counterexamples[0].scenario.initial_state


def test_fuzzer_reports_clean_when_unfalsifiable():
    fuzzer = ScenarioFuzzer(mutable_keys=["pos"], max_iterations=20, seed=1)
    always_safe = Never("impossible", lambda s: False)
    report = fuzzer.falsify(Walker(), make_walk(goal=5), [always_safe])
    assert not report.falsified
    assert report.iterations == 20


def test_fuzzer_ignores_non_numeric_and_bool_keys():
    fuzzer = ScenarioFuzzer(mutable_keys=["name_like", "flag"], max_iterations=5, seed=0)
    scenario = make_walk()
    scenario.initial_state["name_like"] = "abc"
    scenario.initial_state["flag"] = True
    report = fuzzer.falsify(Walker(), scenario, [Never("none", lambda s: False)])
    assert not report.falsified  # and no TypeError from perturbing a str/bool
