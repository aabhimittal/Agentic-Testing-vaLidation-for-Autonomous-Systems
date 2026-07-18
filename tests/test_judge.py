from atlas import (
    KeywordJudge,
    PredicateJudge,
    SemanticValidator,
    TestRunner,
    TokenBudget,
    rollout,
)
from atlas.judge import _parse_verdict

from .helpers import Walker, make_walk


def a_trace():
    return rollout(Walker(), make_walk(goal=5))


def test_predicate_judge_pass_and_fail():
    yes = PredicateJudge(lambda crit, trace: "move" in trace)
    no = PredicateJudge(lambda crit, trace: "teleport" in trace)
    t = compress_text()
    assert yes.evaluate("c", t).passed
    fail = no.evaluate("c", t)
    assert not fail.passed and fail.score == 0.0
    assert yes.evaluate("c", t).tokens_used > 0


def compress_text():
    from atlas import compress_trace

    return compress_trace(a_trace()).text


def test_keyword_judge_required_and_forbidden():
    t = compress_text()
    assert KeywordJudge(required=["move"], forbidden=["crash"]).evaluate("c", t).passed
    missing = KeywordJudge(required=["nonexistent"]).evaluate("c", t)
    assert not missing.passed and "missing required" in missing.rationale
    forbidden = KeywordJudge(forbidden=["move"]).evaluate("c", t)
    assert not forbidden.passed and "found forbidden" in forbidden.rationale


def test_semantic_validator_passes_and_charges_budget():
    budget = TokenBudget(limit=10_000)
    v = SemanticValidator(
        name="mentions_movement",
        criterion="the agent should move",
        judge=PredicateJudge(lambda c, t: "move" in t),
        budget=budget,
    )
    result = v.check(a_trace())
    assert result.passed
    assert budget.spent > 0
    assert budget.ledger["judge:mentions_movement"] == budget.spent
    assert v.last_judgment is not None


def test_semantic_validator_reports_violation_with_rationale():
    v = SemanticValidator(
        name="reaches_mars",
        criterion="the agent should reach mars",
        judge=PredicateJudge(lambda c, t: "mars" in t, rationale="no mars in trace"),
    )
    result = v.check(a_trace())
    assert not result.passed
    assert "judge rejected" in result.violations[0].message
    assert "no mars in trace" in result.violations[0].message


def test_semantic_validator_plugs_into_runner():
    runner = TestRunner(
        agent=Walker(),
        validators=[
            SemanticValidator(
                "mentions_move",
                "agent moves",
                PredicateJudge(lambda c, t: "move" in t),
            )
        ],
    )
    suite = runner.run([make_walk(goal=3)])
    assert suite.passed


def test_parse_verdict_variants():
    assert _parse_verdict('{"passed": true, "score": 1, "rationale": "ok"}')["passed"] is True
    embedded = _parse_verdict('Sure!\n{"passed": false, "score": 0.2, "rationale": "meh"} done')
    assert embedded["passed"] is False and embedded["score"] == 0.2
    garbage = _parse_verdict("not json at all")
    assert garbage["passed"] is False and "unparseable" in garbage["rationale"]
