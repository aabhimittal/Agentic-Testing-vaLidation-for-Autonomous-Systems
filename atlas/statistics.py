"""Statistical validation for stochastic agents and environments.

A single passing run proves nothing about a policy that samples actions, or one
tested under randomized faults (see ``atlas.faults``): it may pass 80% of the
time and fail catastrophically the rest. This module runs a scenario across many
seeds and reports whether the *true* success rate clears a threshold with
statistical confidence — not just the point estimate from one lucky sample.

The confidence bound is the Wilson score lower bound, which — unlike the naive
``passes / trials`` — stays sensible for small samples and rates near 0 or 1
(the regime safety cases care about). A suite gate should test
``result.lower_bound >= required``, never the raw ``pass_rate``.

Nothing here needs numpy/scipy: the z-scores for common confidence levels are
tabulated and the arithmetic is closed-form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .core import AgentUnderTest, Scenario, rollout
from .validators import Validator

# z for a one-sided normal tail at common confidence levels.
_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}

# Factory: given a trial seed, produce the agent to test. Lets callers vary a
# stochastic policy's RNG, or pair with a per-seed FaultInjector.
AgentFactory = Callable[[int], AgentUnderTest]


def z_for(confidence: float) -> float:
    """One-sided z-score for ``confidence`` (nearest tabulated level)."""
    if confidence in _Z:
        return _Z[confidence]
    return min(_Z.items(), key=lambda kv: abs(kv[0] - confidence))[1]


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion.

    Robust where the normal approximation collapses: 10/10 successes yields a
    bound well below 1.0 (honest about small samples), and 0 successes yields a
    bound of exactly 0.0.
    """
    if trials <= 0:
        return 0.0
    z = z_for(confidence)
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = phat + z2 / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * trials)) / trials)
    return max(0.0, (center - margin) / denom)


@dataclass
class StochasticResult:
    """Outcome of evaluating a scenario across many seeded trials."""

    trials: int
    passes: int
    pass_rate: float
    lower_bound: float
    confidence: float
    required: float
    flaky: bool  # some trials passed and some failed
    failing_seeds: tuple[int, ...]

    @property
    def passed(self) -> bool:
        """Gate on the confidence bound, not the point estimate."""
        return self.lower_bound >= self.required

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        flag = " (flaky)" if self.flaky else ""
        return (
            f"[{verdict}] {self.passes}/{self.trials} passed{flag}; "
            f"rate={self.pass_rate:.3f}, {int(self.confidence * 100)}% lower "
            f"bound={self.lower_bound:.3f} vs required {self.required:.3f}"
        )


def evaluate_stochastic(
    agent_factory: AgentFactory,
    scenario: Scenario,
    validators: Sequence[Validator],
    trials: int = 30,
    required: float = 0.95,
    confidence: float = 0.95,
    base_seed: int = 0,
) -> StochasticResult:
    """Run ``scenario`` ``trials`` times, each with a fresh agent from
    ``agent_factory(seed)``, and require every validator to pass for a trial to
    count as a success.

    Seeds are ``base_seed .. base_seed + trials - 1`` so the whole evaluation is
    reproducible. Returns a :class:`StochasticResult` whose ``passed`` gates on
    the Wilson lower bound.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    passes = 0
    failing: list[int] = []
    for i in range(trials):
        seed = base_seed + i
        agent = agent_factory(seed)
        trace = rollout(agent, scenario)
        ok = all(v.check(trace).passed for v in validators)
        if ok:
            passes += 1
        else:
            failing.append(seed)
    rate = passes / trials
    return StochasticResult(
        trials=trials,
        passes=passes,
        pass_rate=rate,
        lower_bound=wilson_lower_bound(passes, trials, confidence),
        confidence=confidence,
        required=required,
        flaky=0 < passes < trials,
        failing_seeds=tuple(failing),
    )
