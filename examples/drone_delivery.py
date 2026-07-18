"""End-to-end ATLAS example: validating a delivery-drone agent.

A drone on a 1-D corridor must reach a goal position before its battery dies,
detouring to a charging pad when low. The greedy policy here is simple on
purpose — the point is what the harness catches:

- safety: battery must never go negative, position stays in the corridor
- bounded response: once battery is low, the drone must be charging within
  10 steps
- liveness: the package is eventually delivered
- metamorphic: decisions must be deterministic and translation-invariant
- adversarial: the fuzzer perturbs start position / battery looking for a
  configuration that strands the drone

Run: python examples/drone_delivery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import (
    Action,
    Always,
    Eventually,
    KeywordJudge,
    Never,
    Observation,
    RespondsWithin,
    SafetyEnvelope,
    Scenario,
    ScenarioFuzzer,
    SemanticValidator,
    TestRunner,
    TokenBudget,
    determinism,
    key_order_invariance,
    resource_monotonicity,
    to_markdown,
    translation_invariance,
)

LOW_BATTERY = 25


class DeliveryDrone:
    """Greedy policy: head for the goal; if battery is low, head for the pad."""

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        s = obs.state
        if s["delivered"]:
            return Action("hover")
        if s["battery"] <= LOW_BATTERY and not s["charging"]:
            return self._move_toward(s["pos"], s["pad_pos"], reason="charge")
        if s["charging"] and s["battery"] < 90:
            return Action("charge")
        return self._move_toward(s["pos"], s["goal_pos"], reason="deliver")

    @staticmethod
    def _move_toward(pos: float, target: float, reason: str) -> Action:
        if pos < target:
            return Action("move", {"dx": 1})
        if pos > target:
            return Action("move", {"dx": -1})
        return Action("charge") if reason == "charge" else Action("drop")


def dynamics(state: dict, action: Action) -> dict:
    s = dict(state)
    if action.name == "move":
        s["pos"] += action.params["dx"]
        s["battery"] -= 2
    elif action.name == "charge":
        s["battery"] = min(100, s["battery"] + 10)
    elif action.name == "drop":
        if s["pos"] == s["goal_pos"]:
            s["delivered"] = True
        s["battery"] -= 1
    elif action.name == "hover":
        s["battery"] -= 1
    s["charging"] = s["pos"] == s["pad_pos"]
    return s


def make_scenario(start: int = 0, battery: int = 60, goal: int = 12, pad: int = 3) -> Scenario:
    return Scenario(
        name=f"delivery[start={start},battery={battery}]",
        initial_state={
            "pos": start, "battery": battery, "goal_pos": goal,
            "pad_pos": pad, "delivered": False, "charging": start == pad,
        },
        dynamics=dynamics,
        done=lambda s: s["delivered"],
        max_steps=120,
    )


VALIDATORS = [
    Never("battery_depleted", lambda s: s["battery"] <= 0),
    SafetyEnvelope("corridor_bounds", {"pos": (-5, 50)}),
    RespondsWithin(
        "low_battery_response",
        trigger=lambda s: s["battery"] <= LOW_BATTERY and not s["delivered"],
        response=lambda s: s["charging"] or s["delivered"],
        window=10,
    ),
    Eventually("package_delivered", lambda s: s["delivered"]),
]


def main() -> int:
    agent = DeliveryDrone()
    scenarios = [
        make_scenario(),                    # comfortable margin
        make_scenario(start=8, battery=30), # starts low, must detour
        make_scenario(start=1, battery=95, goal=20),
    ]

    budget = TokenBudget(limit=4000)

    # An LLM-as-judge validator for a fuzzy, outcome-level property. KeywordJudge
    # runs fully offline so the example needs no API key; swap in
    #   AnthropicJudge(model="claude-sonnet-5")
    # for real semantic evaluation of free-form criteria (reads ANTHROPIC_API_KEY).
    mission_complete = SemanticValidator(
        name="mission_complete",
        criterion="The mission should end with the package delivered and the "
                  "drone still operational (battery not depleted).",
        judge=KeywordJudge(required=["delivered=True"], forbidden=["battery=0 "]),
        budget=budget,
    )

    runner = TestRunner(
        agent=agent,
        validators=list(VALIDATORS) + [mission_complete],
        relations=[
            determinism(),
            translation_invariance(("pos", "goal_pos", "pad_pos")),
            resource_monotonicity("battery", delta=20),
            key_order_invariance(),
        ],
        budget=budget,
    )
    suite = runner.run(scenarios)
    print(to_markdown(suite))

    # Fuzz the tight-margin scenario: larger perturbations can push the drone
    # past the point of no return (low battery, far from the pad).
    fuzzer = ScenarioFuzzer(
        mutable_keys=["pos", "battery"], max_iterations=60, relative_step=0.6, seed=7
    )
    fuzz = fuzzer.falsify(agent, make_scenario(start=8, battery=30), VALIDATORS)
    if fuzz.falsified:
        cx = fuzz.counterexamples[0]
        print(f"Fuzzer counterexample after {cx.iterations_used} iteration(s): "
              f"validator `{cx.validator}` fails from initial state "
              f"{cx.scenario.initial_state} (seed={fuzz.seed})")
    else:
        print(f"Fuzzer: no counterexample in {fuzz.iterations} iterations (seed={fuzz.seed})")

    return 0 if suite.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
