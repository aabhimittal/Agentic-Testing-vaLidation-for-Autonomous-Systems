# ATLAS — Agentic Testing & vaLidation for Autonomous Systems

A zero-dependency Python framework for testing autonomous agents when there is
**no oracle for the "correct" action** — combining four complementary
approaches in one harness:

| Approach | Module | What it catches |
|---|---|---|
| Temporal-logic trace validation | `atlas.validators` | Safety invariants, forbidden states, liveness, *bounded response* (noticing a hazard but reacting too late) |
| Metamorphic testing for agents | `atlas.metamorphic` | Non-determinism, state leaks, coordinate-frame sensitivity, hard-coded direction, resource non-monotonicity, dict-order dependence, coupling to irrelevant inputs |
| Adversarial falsification | `atlas.adversarial` | The scenarios you *didn't* write: seeded, reproducible fuzzing of initial conditions until a validator breaks |
| Token-budgeted execution | `atlas.tokens` | Runaway LLM-evaluation cost: delta-compressed traces, hard token budgets, cost-aware scenario selection, pluggable tokenizer (heuristic or `tiktoken`) |
| LLM-as-judge validation | `atlas.judge` | Fuzzy, oracle-free properties ("behaved cautiously", "recovered gracefully"): a judge scores the trace and plugs in as an ordinary validator, charging its tokens to the budget |

Everything is deterministic, dependency-free, and runs in milliseconds.

## Why these approaches

**Agents have no expected-output assertion.** You can't `assert action == correct_action`
because nobody knows the correct action. ATLAS tests *properties of behavior*
instead:

- **Temporal validators** check the recorded trace: `Never(battery <= 0)`,
  `Eventually(delivered)`, and — the interesting one — `RespondsWithin`:
  *whenever* battery is low, the agent must be charging *within N steps*.
  Plain invariants can't express "noticed the hazard but reacted too late."
- **Metamorphic relations** need no oracle at all: they transform a scenario
  and assert a relation between the two runs. The built-in catalogue:
  `determinism`, `translation_invariance`, `irrelevant_key_invariance`,
  `mirror_symmetry` (reflect the world → every decision must reflect; catches
  hard-coded directions), `resource_monotonicity` (more of a resource must not
  make the agent slower or fail), and `key_order_invariance` (dict iteration
  order must not leak into decisions). Violations expose brittle policies
  without labeling a single "correct" answer.
- **The fuzzer treats your validators as an objective** and perturbs initial
  conditions until one breaks — falsification-based testing (from
  cyber-physical systems) applied to agent policies. Every counterexample is
  reproducible from `(seed, scenario, validator)`.
- **An LLM judge handles what predicates can't**: `SemanticValidator` wraps a
  `Judge` and a natural-language criterion, renders the trace with the same
  compressor the budget accounts for, and returns an ordinary `ValidationResult`
  — so fuzzy properties sit in the same suite as temporal ones. `KeywordJudge`
  / `PredicateJudge` are deterministic and offline (use them in CI);
  `AnthropicJudge` calls a real model when you install `anthropic`.

**Token optimization is a first-class concern.** Modern agent test suites route
traces through LLM judges and summarizers, where cost scales with tokens:

- `compress_trace` delta-encodes states (only changed keys per step) and
  run-length-encodes repeated actions — typically **3–10× smaller** before a
  trace ever reaches a model, with the first step kept in full so the encoding
  is self-contained.
- `TokenBudget` is a hard, per-label-accounted ceiling shared by the whole
  suite; when it runs dry, remaining scenarios are recorded as **skipped, not
  silently dropped** — and a suite with skips does not report success.
- `select_scenarios` spends a tight budget where bugs are likely: greedy
  knapsack over *expected failures per token* from historical failure rates.
- Token counting is a pluggable backend. The default is a dependency-free
  ~4-chars/token heuristic; `pip install ".[tiktoken]"` then
  `set_default_tokenizer(TiktokenTokenizer())` for exact BPE counts, or pass any
  object with `count(text) -> int`.

## Quick start

```bash
pip install -e ".[dev]"
pytest                              # framework test suite
python examples/drone_delivery.py   # full worked example
```

```python
from atlas import (
    Action, Never, Eventually, RespondsWithin, Scenario,
    ScenarioFuzzer, TestRunner, TokenBudget, determinism,
)

class MyAgent:
    def reset(self): ...
    def act(self, obs) -> Action: ...

scenario = Scenario(
    name="nominal",
    initial_state={"pos": 0, "battery": 60},
    dynamics=my_dynamics,          # pure (state, action) -> state
    done=lambda s: s["delivered"],
    max_steps=120,
)

runner = TestRunner(
    agent=MyAgent(),
    validators=[
        Never("battery_dead", lambda s: s["battery"] <= 0),
        RespondsWithin("low_batt", trigger=is_low, response=is_charging, window=10),
        Eventually("delivered", lambda s: s["delivered"]),
    ],
    relations=[determinism()],
    budget=TokenBudget(limit=4000),
)
suite = runner.run([scenario])

# Then hunt for the scenarios you didn't think of:
fuzz = ScenarioFuzzer(mutable_keys=["pos", "battery"], seed=7).falsify(
    MyAgent(), scenario, runner.validators
)
```

Add a semantic, oracle-free property with an LLM judge (offline judge shown;
swap in `AnthropicJudge(model="claude-sonnet-5")` for a real model):

```python
from atlas import SemanticValidator, KeywordJudge

budget = TokenBudget(limit=4000)
mission = SemanticValidator(
    name="mission_complete",
    criterion="The run should end with the package delivered and the drone still operational.",
    judge=KeywordJudge(required=["delivered=True"], forbidden=["battery=0 "]),
    budget=budget,        # judge tokens are charged here, alongside trace compression
)
runner = TestRunner(agent=MyAgent(), validators=[mission], budget=budget)
```

`atlas.report.to_markdown(suite)` renders a PR-ready report;
`to_json(suite)` feeds dashboards.

## Layout

```
atlas/
  core.py         Agent protocol, Scenario, Trace, rollout
  validators.py   Always / Never / Eventually / RespondsWithin / SafetyEnvelope
  metamorphic.py  determinism, translation/mirror/order invariance, monotonicity
  adversarial.py  ScenarioFuzzer (seeded falsification)
  tokens.py       TokenBudget, compress_trace, select_scenarios, tokenizers
  judge.py        Judge protocol, Keyword/Predicate/Anthropic judges, SemanticValidator
  runner.py       TestRunner -> SuiteReport
  report.py       Markdown / JSON rendering
examples/drone_delivery.py   end-to-end worked example
tests/                       pytest suite
```

## Design notes

- Scenario dynamics must be **pure functions** of `(state, action)`; this is
  what makes metamorphic transforms sound and fuzz counterexamples replayable.
- Temporal properties are evaluated over **finite recorded traces**, not a
  model — cheap, exact about what actually happened, and every violation
  carries the timestep where it occurred.
- A truncated trace (hit `max_steps`) fails `Eventually` and gets no grace
  period from `RespondsWithin`: running out the clock is not a pass.

## License

MIT — see [LICENSE](LICENSE).
