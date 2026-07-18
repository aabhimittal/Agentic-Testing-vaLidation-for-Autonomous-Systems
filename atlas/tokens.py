"""Token optimization for agent testing pipelines.

Agent test suites increasingly route traces through LLMs (semantic judges,
failure triage, report summarization), where cost scales with tokens, not
CPU. This module keeps that spend flat:

- ``TokenBudget``: a hard per-run token ceiling shared across the suite.
- ``compress_trace``: lossless-in-substance delta encoding of a trace —
  only changed state keys per step, run-length encoded repeated actions.
  Typical traces compress 3-10x before ever reaching a model.
- ``CostAwareSelector``: greedy knapsack that orders scenarios by expected
  failures per token, so a tight budget is spent where bugs are likely.

Token counts use the ~4 chars/token heuristic; swap in a real tokenizer via
``estimate_tokens``'s ``chars_per_token`` if exactness matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .core import Trace


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Cheap token estimate; deliberately dependency-free."""
    return max(1, round(len(text) / chars_per_token))


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push spend past the budget's hard limit."""


@dataclass
class TokenBudget:
    """A hard token ceiling with per-label accounting."""

    limit: int
    spent: int = 0
    ledger: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def can_afford(self, tokens: int) -> bool:
        return self.spent + tokens <= self.limit

    def charge(self, tokens: int, label: str = "unlabeled") -> None:
        if not self.can_afford(tokens):
            raise BudgetExceeded(
                f"charge of {tokens} tokens for {label!r} exceeds budget "
                f"({self.spent}/{self.limit} spent)"
            )
        self.spent += tokens
        self.ledger[label] = self.ledger.get(label, 0) + tokens


@dataclass
class CompressedTrace:
    text: str
    raw_tokens: int
    compressed_tokens: int

    @property
    def ratio(self) -> float:
        return self.raw_tokens / max(1, self.compressed_tokens)


def _render_full(trace: Trace) -> str:
    lines = [f"trace {trace.scenario_name} ({len(trace)} steps)"]
    for step in trace:
        state = " ".join(f"{k}={v}" for k, v in sorted(step.observation.state.items()))
        lines.append(f"t={step.observation.timestep} state[{state}] -> {step.action}")
    final = " ".join(f"{k}={v}" for k, v in sorted(trace.final_state.items()))
    lines.append(f"final[{final}] truncated={trace.truncated}")
    return "\n".join(lines)


def compress_trace(trace: Trace) -> CompressedTrace:
    """Delta + run-length encode a trace for LLM consumption.

    Step lines carry only the state keys that changed since the previous
    step; consecutive identical (delta, action) lines collapse into one line
    with an ``xN`` repeat count. The first step is always emitted in full so
    the encoding is self-contained.
    """
    raw = _render_full(trace)
    lines = [f"trace {trace.scenario_name} ({len(trace)} steps, delta-encoded)"]

    prev_state: dict | None = None
    pending: str | None = None
    pending_count = 0

    def flush() -> None:
        nonlocal pending, pending_count
        if pending is not None:
            lines.append(pending if pending_count == 1 else f"{pending} x{pending_count}")
        pending, pending_count = None, 0

    for step in trace:
        state = step.observation.state
        if prev_state is None:
            delta = dict(sorted(state.items()))
        else:
            delta = {k: v for k, v in sorted(state.items()) if prev_state.get(k) != v}
            gone = [k for k in prev_state if k not in state]
            for k in gone:
                delta[k] = "<removed>"
        rendered = " ".join(f"{k}={v}" for k, v in delta.items())
        line = f"[{rendered}] -> {step.action}"
        if line == pending:
            pending_count += 1
        else:
            flush()
            pending, pending_count = line, 1
        prev_state = state

    flush()
    if prev_state is not None:
        final_delta = {
            k: v for k, v in sorted(trace.final_state.items()) if prev_state.get(k) != v
        }
    else:
        final_delta = dict(sorted(trace.final_state.items()))
    rendered = " ".join(f"{k}={v}" for k, v in final_delta.items())
    lines.append(f"final[{rendered}] truncated={trace.truncated}")

    text = "\n".join(lines)
    return CompressedTrace(
        text=text,
        raw_tokens=estimate_tokens(raw),
        compressed_tokens=estimate_tokens(text),
    )


@dataclass(frozen=True)
class ScenarioCost:
    """Prior knowledge about a scenario used for budget-aware selection."""

    name: str
    expected_tokens: int
    historical_failure_rate: float  # 0.0..1.0, from previous runs

    @property
    def value_density(self) -> float:
        """Expected failures surfaced per token spent."""
        return self.historical_failure_rate / max(1, self.expected_tokens)


def select_scenarios(costs: Sequence[ScenarioCost], budget: TokenBudget) -> list[str]:
    """Greedy knapsack: highest value-density first, while the budget allows.

    Scenarios that don't fit are skipped, not queued — later cheaper
    scenarios may still fit. Ties break by name for deterministic ordering.
    """
    chosen: list[str] = []
    remaining = budget.remaining
    ranked = sorted(costs, key=lambda c: (-c.value_density, c.name))
    for cost in ranked:
        if cost.expected_tokens <= remaining:
            chosen.append(cost.name)
            remaining -= cost.expected_tokens
    return chosen
