"""ATLAS — Agentic Testing & vaLidation for Autonomous Systems."""

from .adversarial import Counterexample, FuzzReport, ScenarioFuzzer
from .core import Action, AgentUnderTest, Observation, Scenario, Step, Trace, rollout
from .metamorphic import (
    MetamorphicRelation,
    MetamorphicResult,
    determinism,
    irrelevant_key_invariance,
    same_actions,
    translation_invariance,
)
from .report import to_json, to_markdown
from .runner import ScenarioReport, SuiteReport, TestRunner
from .tokens import (
    BudgetExceeded,
    CompressedTrace,
    ScenarioCost,
    TokenBudget,
    compress_trace,
    estimate_tokens,
    select_scenarios,
)
from .validators import (
    Always,
    Eventually,
    Never,
    RespondsWithin,
    SafetyEnvelope,
    ValidationResult,
    Validator,
    Violation,
)

__version__ = "0.1.0"

__all__ = [
    "Action", "AgentUnderTest", "Observation", "Scenario", "Step", "Trace", "rollout",
    "Always", "Never", "Eventually", "RespondsWithin", "SafetyEnvelope",
    "ValidationResult", "Validator", "Violation",
    "MetamorphicRelation", "MetamorphicResult", "determinism",
    "translation_invariance", "irrelevant_key_invariance", "same_actions",
    "ScenarioFuzzer", "FuzzReport", "Counterexample",
    "TokenBudget", "BudgetExceeded", "CompressedTrace", "ScenarioCost",
    "compress_trace", "estimate_tokens", "select_scenarios",
    "TestRunner", "ScenarioReport", "SuiteReport",
    "to_markdown", "to_json",
]
