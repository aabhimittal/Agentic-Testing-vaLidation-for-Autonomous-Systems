"""Test runner: rolls agents through scenarios, applies validators and
metamorphic relations, and charges every trace against a shared token budget.

When the budget runs dry mid-suite, remaining scenarios are recorded as
skipped rather than silently dropped — a suite that quietly truncates reads
as "covered everything" when it didn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .core import AgentUnderTest, Scenario, Trace, rollout
from .metamorphic import MetamorphicRelation, MetamorphicResult
from .tokens import TokenBudget, compress_trace
from .validators import ValidationResult, Validator


@dataclass
class ScenarioReport:
    scenario: str
    trace_steps: int
    truncated: bool
    validations: list[ValidationResult] = field(default_factory=list)
    metamorphic: list[MetamorphicResult] = field(default_factory=list)
    tokens_charged: int = 0
    compression_ratio: float = 1.0

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.validations) and all(
            m.passed for m in self.metamorphic
        )


@dataclass
class SuiteReport:
    reports: list[ScenarioReport] = field(default_factory=list)
    skipped_scenarios: list[str] = field(default_factory=list)
    tokens_spent: int = 0
    token_limit: int = 0

    @property
    def passed(self) -> bool:
        return not self.skipped_scenarios and all(r.passed for r in self.reports)

    @property
    def failures(self) -> list[ScenarioReport]:
        return [r for r in self.reports if not r.passed]


@dataclass
class TestRunner:
    __test__ = False  # keep pytest from collecting this as a test class

    agent: AgentUnderTest
    validators: Sequence[Validator] = ()
    relations: Sequence[MetamorphicRelation] = ()
    budget: TokenBudget | None = None

    def run(self, scenarios: Sequence[Scenario]) -> SuiteReport:
        suite = SuiteReport(token_limit=self.budget.limit if self.budget else 0)
        for scenario in scenarios:
            trace = rollout(self.agent, scenario)
            compressed = compress_trace(trace)

            if self.budget is not None:
                if not self.budget.can_afford(compressed.compressed_tokens):
                    suite.skipped_scenarios.append(scenario.name)
                    continue
                self.budget.charge(compressed.compressed_tokens, label=scenario.name)

            report = ScenarioReport(
                scenario=scenario.name,
                trace_steps=len(trace),
                truncated=trace.truncated,
                tokens_charged=compressed.compressed_tokens,
                compression_ratio=compressed.ratio,
            )
            report.validations = [v.check(trace) for v in self.validators]
            report.metamorphic = [r.check(self.agent, scenario) for r in self.relations]
            suite.reports.append(report)

        suite.tokens_spent = self.budget.spent if self.budget else 0
        return suite

    def run_one(self, scenario: Scenario) -> tuple[Trace, ScenarioReport]:
        """Roll out a single scenario and validate it, bypassing the budget.
        Useful for debugging a specific failure interactively."""
        trace = rollout(self.agent, scenario)
        compressed = compress_trace(trace)
        report = ScenarioReport(
            scenario=scenario.name,
            trace_steps=len(trace),
            truncated=trace.truncated,
            tokens_charged=0,
            compression_ratio=compressed.ratio,
        )
        report.validations = [v.check(trace) for v in self.validators]
        report.metamorphic = [r.check(self.agent, scenario) for r in self.relations]
        return trace, report
