"""Trace validators: safety invariants and a bounded temporal logic.

Instead of full LTL model checking, ATLAS evaluates temporal properties over
finite recorded traces — cheap, exact on what actually happened, and every
violation carries the timestep where it occurred so failures are debuggable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable

from .core import State, Trace

Predicate = Callable[[State], bool]


def _is_finite_number(v: object) -> bool:
    """True for a real, finite int/float (bools and non-numbers excluded)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


@dataclass(frozen=True)
class Violation:
    validator: str
    timestep: int  # -1 means "over the whole trace"
    message: str


@dataclass
class ValidationResult:
    validator: str
    passed: bool
    violations: list[Violation] = field(default_factory=list)


@runtime_checkable
class Validator(Protocol):
    name: str

    def check(self, trace: Trace) -> ValidationResult: ...


def _result(name: str, violations: list[Violation]) -> ValidationResult:
    return ValidationResult(validator=name, passed=not violations, violations=violations)


@dataclass
class Always:
    """Safety invariant: ``predicate`` must hold in every state of the trace."""

    name: str
    predicate: Predicate

    def check(self, trace: Trace) -> ValidationResult:
        violations = [
            Violation(self.name, t, f"invariant violated at step {t}")
            for t, state in enumerate(trace.states())
            if not self.predicate(state)
        ]
        return _result(self.name, violations)


@dataclass
class Never:
    """Forbidden-state property: ``predicate`` must hold in no state."""

    name: str
    predicate: Predicate

    def check(self, trace: Trace) -> ValidationResult:
        violations = [
            Violation(self.name, t, f"forbidden state entered at step {t}")
            for t, state in enumerate(trace.states())
            if self.predicate(state)
        ]
        return _result(self.name, violations)


@dataclass
class Eventually:
    """Liveness on a finite trace: ``predicate`` must hold in some state.

    A truncated trace (max_steps hit) that never satisfies the predicate is a
    genuine failure: the agent had its full step budget and did not deliver.
    """

    name: str
    predicate: Predicate

    def check(self, trace: Trace) -> ValidationResult:
        if any(self.predicate(s) for s in trace.states()):
            return _result(self.name, [])
        return _result(
            self.name,
            [Violation(self.name, -1, "goal predicate never satisfied in trace")],
        )


@dataclass
class RespondsWithin:
    """Bounded response: whenever ``trigger`` holds, ``response`` must hold
    within ``window`` subsequent states. Catches agents that notice a hazard
    (low battery, obstacle) but react too late — a class of bug plain
    invariants miss entirely.
    """

    name: str
    trigger: Predicate
    response: Predicate
    window: int

    def check(self, trace: Trace) -> ValidationResult:
        states = trace.states()
        violations: list[Violation] = []
        for t, state in enumerate(states):
            if not self.trigger(state):
                continue
            horizon = states[t : t + self.window + 1]
            if not any(self.response(s) for s in horizon):
                if t + self.window >= len(states) and not trace.truncated:
                    # Scenario completed before the window could elapse:
                    # inconclusive, not a failure. Truncated traces don't get
                    # this grace — the agent simply ran out its budget.
                    continue
                violations.append(
                    Violation(
                        self.name,
                        t,
                        f"trigger at step {t} not answered within {self.window} steps",
                    )
                )
        return _result(self.name, violations)


@dataclass
class SafetyEnvelope:
    """Numeric bounds on state variables: {key: (min, max)}. ``None`` skips a side."""

    name: str
    bounds: dict[str, tuple[float | None, float | None]]

    def check(self, trace: Trace) -> ValidationResult:
        violations: list[Violation] = []
        for t, state in enumerate(trace.states()):
            for key, (lo, hi) in self.bounds.items():
                if key not in state:
                    continue
                v = state[key]
                # A non-finite value (NaN/inf) makes `v < lo` and `v > hi` both
                # False, so a naive bound check waves corruption through. In a
                # safety envelope that is exactly the case we must flag.
                if not _is_finite_number(v):
                    violations.append(
                        Violation(self.name, t, f"{key}={v} is not a finite number at step {t}")
                    )
                    continue
                if lo is not None and v < lo:
                    violations.append(
                        Violation(self.name, t, f"{key}={v} below minimum {lo} at step {t}")
                    )
                if hi is not None and v > hi:
                    violations.append(
                        Violation(self.name, t, f"{key}={v} above maximum {hi} at step {t}")
                    )
        return _result(self.name, violations)


@dataclass
class Finite:
    """Every listed numeric key must stay a finite number in every state.

    NaN/inf leaking into state — a divide-by-zero in a controller, an
    uninitialized sensor, an overflowed integrator — is a silent failure mode:
    ordinary comparison-based checks pass right through it. This validator makes
    non-finite state an explicit, located failure. With no ``keys``, it checks
    every numeric value present in each state.
    """

    name: str
    keys: Iterable[str] | None = None

    def check(self, trace: Trace) -> ValidationResult:
        violations: list[Violation] = []
        for t, state in enumerate(trace.states()):
            if self.keys is None:
                items = [(k, v) for k, v in state.items() if isinstance(v, (int, float))
                         and not isinstance(v, bool)]
            else:
                items = [(k, state[k]) for k in self.keys if k in state]
            for key, v in items:
                if not _is_finite_number(v):
                    violations.append(
                        Violation(self.name, t, f"{key}={v} is non-finite at step {t}")
                    )
        return _result(self.name, violations)
