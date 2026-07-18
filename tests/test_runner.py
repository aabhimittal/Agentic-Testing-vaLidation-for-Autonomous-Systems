import json

from atlas import (
    Always,
    Eventually,
    TestRunner,
    TokenBudget,
    determinism,
    to_json,
    to_markdown,
)

from .helpers import Walker, make_walk


def build_runner(budget=None):
    return TestRunner(
        agent=Walker(),
        validators=[
            Always("fuel_positive", lambda s: s["fuel"] > 0),
            Eventually("arrived", lambda s: s["pos"] >= s["goal"]),
        ],
        relations=[determinism()],
        budget=budget,
    )


def test_suite_passes_end_to_end():
    suite = build_runner().run([make_walk(goal=3), make_walk(goal=7)])
    assert suite.passed
    assert len(suite.reports) == 2
    assert all(r.compression_ratio >= 1.0 for r in suite.reports)


def test_suite_records_failures():
    suite = build_runner().run([make_walk(goal=50, max_steps=10)])  # can't arrive
    assert not suite.passed
    names = [v.validator for r in suite.failures for v in r.validations if not v.passed]
    assert names == ["arrived"]


def test_budget_skips_scenarios_and_marks_suite_failed():
    suite = build_runner(budget=TokenBudget(limit=60)).run(
        [make_walk(goal=3), make_walk(goal=30, max_steps=35)]
    )
    assert len(suite.reports) == 1
    assert suite.skipped_scenarios == ["walk_to_30"]
    assert not suite.passed  # skipped work is not silent success
    assert suite.tokens_spent <= 60


def test_markdown_and_json_reports():
    suite = build_runner().run([make_walk(goal=3), make_walk(goal=50, max_steps=10)])
    md = to_markdown(suite)
    assert "FAILED" in md
    assert "walk_to_3" in md and "walk_to_50" in md
    assert "goal predicate never satisfied" in md

    payload = json.loads(to_json(suite))
    assert payload["passed"] is False
    assert len(payload["scenarios"]) == 2
    failing = next(s for s in payload["scenarios"] if not s["passed"])
    assert failing["violations"][0]["validator"] == "arrived"


def test_run_one_bypasses_budget():
    runner = build_runner(budget=TokenBudget(limit=1))
    trace, report = runner.run_one(make_walk(goal=3))
    assert len(trace) == 3
    assert report.passed
    assert runner.budget.spent == 0
