"""LLM-as-judge: semantic validation of traces against natural-language
criteria.

Some agent properties resist a Python predicate — "the drone behaved
cautiously near obstacles", "the plan is coherent", "recovery was graceful".
A ``Judge`` evaluates a rendered trace against a criterion string and returns a
structured ``Judgment``. ``SemanticValidator`` adapts any judge into the
ordinary ``Validator`` interface so judged properties sit alongside temporal
ones in the same suite — and charges the judge's token cost to the shared
``TokenBudget``.

Backends:
- ``KeywordJudge`` / ``PredicateJudge``: deterministic, offline, dependency
  free. Use these in CI so the suite never needs a network or API key.
- ``AnthropicJudge``: real LLM judging via the optional ``anthropic`` package,
  reading the API key from the environment. Lazily imported.

Judges are *advisory by design*: they can be wrong, so keep hard safety
properties in deterministic validators and reserve judges for the fuzzy,
oracle-free questions only a model can answer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, runtime_checkable

from .core import Trace
from .tokens import TokenBudget, Tokenizer, compress_trace, estimate_tokens
from .validators import ValidationResult, Violation


@dataclass
class Judgment:
    """A judge's verdict on one (criterion, trace) pair."""

    passed: bool
    score: float  # 0.0 (fails criterion) .. 1.0 (fully satisfies)
    rationale: str
    tokens_used: int = 0


@runtime_checkable
class Judge(Protocol):
    def evaluate(self, criterion: str, rendered_trace: str) -> Judgment: ...


@dataclass
class PredicateJudge:
    """Deterministic judge driven by a Python predicate over (criterion, trace).
    The offline stand-in for a real LLM in tests and CI."""

    predicate: Callable[[str, str], bool]
    rationale: str = "predicate judge"

    def evaluate(self, criterion: str, rendered_trace: str) -> Judgment:
        ok = self.predicate(criterion, rendered_trace)
        tokens = estimate_tokens(criterion) + estimate_tokens(rendered_trace)
        return Judgment(
            passed=ok,
            score=1.0 if ok else 0.0,
            rationale=self.rationale,
            tokens_used=tokens,
        )


@dataclass
class KeywordJudge:
    """Passes iff every keyword in ``required`` appears in the rendered trace
    and none in ``forbidden`` do. A crude but genuinely useful offline judge
    for 'the trace should/shouldn't contain X' checks."""

    required: Sequence[str] = ()
    forbidden: Sequence[str] = ()

    def evaluate(self, criterion: str, rendered_trace: str) -> Judgment:
        missing = [k for k in self.required if k not in rendered_trace]
        present = [k for k in self.forbidden if k in rendered_trace]
        ok = not missing and not present
        parts = []
        if missing:
            parts.append(f"missing required: {missing}")
        if present:
            parts.append(f"found forbidden: {present}")
        rationale = "; ".join(parts) if parts else "all keyword constraints met"
        satisfied = len(self.required) - len(missing)
        total = max(1, len(self.required) + len(self.forbidden))
        score = (satisfied + (len(self.forbidden) - len(present))) / total
        tokens = estimate_tokens(rendered_trace)
        return Judgment(passed=ok, score=round(score, 3), rationale=rationale, tokens_used=tokens)


_JUDGE_SYSTEM = (
    "You are a strict evaluator of autonomous-agent execution traces. "
    "You are given a CRITERION and a delta-encoded TRACE. Decide whether the "
    "trace satisfies the criterion. Respond with ONLY a JSON object: "
    '{"passed": bool, "score": number in [0,1], "rationale": string}. '
    "No prose outside the JSON."
)


class AnthropicJudge:
    """Real LLM judge backed by the Anthropic Messages API (optional).

    Requires ``pip install anthropic`` and an API key (``ANTHROPIC_API_KEY`` by
    default). The model id is a constructor argument so you pick the
    capability/cost tradeoff; a mid-tier model is usually the right judge.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str | None = None,
        max_tokens: int = 512,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without anthropic
            raise ImportError(
                "AnthropicJudge requires the 'anthropic' package; "
                "install it with `pip install anthropic`."
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "AnthropicJudge needs an API key: set ANTHROPIC_API_KEY or pass api_key."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def evaluate(self, criterion: str, rendered_trace: str) -> Judgment:
        user = f"CRITERION:\n{criterion}\n\nTRACE:\n{rendered_trace}"
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        verdict = _parse_verdict(text)
        tokens = _usage_tokens(response) or (
            estimate_tokens(_JUDGE_SYSTEM + user, tokenizer=self.tokenizer)
            + estimate_tokens(text, tokenizer=self.tokenizer)
        )
        return Judgment(
            passed=bool(verdict.get("passed", False)),
            score=float(verdict.get("score", 0.0)),
            rationale=str(verdict.get("rationale", "")),
            tokens_used=tokens,
        )


def _parse_verdict(text: str) -> dict:
    """Extract the JSON verdict from a model reply, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"passed": False, "score": 0.0, "rationale": f"unparseable judge reply: {text[:120]!r}"}


def _usage_tokens(response) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)


@dataclass
class SemanticValidator:
    """Adapts a ``Judge`` + criterion into a ``Validator``.

    Renders the trace with ATLAS's compressor (so the judge reads the same
    token-efficient form the budget accounts for) and, if a ``budget`` is
    supplied, charges the judge's token cost — semantic validation that
    respects the suite's ceiling.
    """

    name: str
    criterion: str
    judge: Judge
    budget: TokenBudget | None = None
    tokenizer: Tokenizer | None = None
    last_judgment: Judgment | None = field(default=None, init=False)

    def check(self, trace: Trace) -> ValidationResult:
        rendered = compress_trace(trace, self.tokenizer).text
        judgment = self.judge.evaluate(self.criterion, rendered)
        self.last_judgment = judgment
        if self.budget is not None and judgment.tokens_used:
            self.budget.charge(judgment.tokens_used, label=f"judge:{self.name}")
        if judgment.passed:
            return ValidationResult(self.name, True, [])
        return ValidationResult(
            self.name,
            False,
            [
                Violation(
                    self.name,
                    -1,
                    f"judge rejected (score={judgment.score:.2f}): {judgment.rationale}",
                )
            ],
        )
