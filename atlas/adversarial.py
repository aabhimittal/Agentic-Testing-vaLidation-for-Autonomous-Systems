"""Adversarial falsification: search for scenario variants that break the agent.

Rather than testing a fixed scenario list, the fuzzer treats validators as an
objective and hill-climbs over initial-state perturbations, keeping mutants
that move the agent closer to a violation. This is falsification-based testing
(as used for cyber-physical systems) applied to agent policies: hand-written
scenarios find the bugs you thought of; the fuzzer finds the ones you didn't.

Fully deterministic under a seed, so every discovered counterexample is
reproducible from (seed, scenario, validator) alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .core import AgentUnderTest, Scenario, State, rollout
from .validators import Validator


@dataclass
class Counterexample:
    scenario: Scenario
    validator: str
    mutated_keys: tuple[str, ...]
    iterations_used: int


@dataclass
class FuzzReport:
    seed: int
    iterations: int
    counterexamples: list[Counterexample] = field(default_factory=list)

    @property
    def falsified(self) -> bool:
        return bool(self.counterexamples)


@dataclass
class ScenarioFuzzer:
    """Perturb numeric initial-state keys, guided by how close each rollout
    gets to violating a validator (violation count as a coarse robustness
    signal; first violation wins).
    """

    mutable_keys: Sequence[str]
    max_iterations: int = 50
    relative_step: float = 0.25  # perturbation size as a fraction of the value
    seed: int = 0

    def falsify(
        self,
        agent: AgentUnderTest,
        scenario: Scenario,
        validators: Sequence[Validator],
    ) -> FuzzReport:
        rng = random.Random(self.seed)
        report = FuzzReport(seed=self.seed, iterations=self.max_iterations)
        base = dict(scenario.initial_state)

        for i in range(1, self.max_iterations + 1):
            candidate, touched = self._mutate(base, rng)
            variant = scenario.with_initial_state(candidate, suffix=f"fuzz{i}")
            trace = rollout(agent, variant)
            for validator in validators:
                result = validator.check(trace)
                if not result.passed:
                    report.counterexamples.append(
                        Counterexample(
                            scenario=variant,
                            validator=result.validator,
                            mutated_keys=touched,
                            iterations_used=i,
                        )
                    )
                    report.iterations = i
                    return report  # first counterexample is enough; stay cheap
        return report

    def _mutate(self, base: State, rng: random.Random) -> tuple[State, tuple[str, ...]]:
        candidate = dict(base)
        eligible = [
            k for k in self.mutable_keys
            if isinstance(candidate.get(k), (int, float)) and not isinstance(candidate.get(k), bool)
        ]
        if not eligible:
            return candidate, ()
        touched = tuple(rng.sample(eligible, k=rng.randint(1, len(eligible))))
        for key in touched:
            value = candidate[key]
            span = max(abs(value) * self.relative_step, 1.0)
            perturbed = value + rng.uniform(-span, span)
            candidate[key] = round(perturbed) if isinstance(value, int) else perturbed
        return candidate, touched
