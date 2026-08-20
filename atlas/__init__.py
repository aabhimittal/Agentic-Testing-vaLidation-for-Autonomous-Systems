"""ATLAS — Agentic Testing & vaLidation for Autonomous Systems."""

from .adversarial import Counterexample, FuzzReport, ScenarioFuzzer
from .core import Action, AgentUnderTest, Observation, Scenario, Step, Trace, rollout
from .faults import (
    Bias,
    Delay,
    DropSensor,
    Dropout,
    FaultInjector,
    GaussianNoise,
    Saturate,
    StuckActuator,
    StuckSensor,
)
from .judge import (
    AnthropicJudge,
    Judge,
    Judgment,
    KeywordJudge,
    PredicateJudge,
    SemanticValidator,
)
from .metamorphic import (
    MetamorphicRelation,
    MetamorphicResult,
    actions_related_by,
    determinism,
    irrelevant_key_invariance,
    key_order_invariance,
    mirror_symmetry,
    resource_monotonicity,
    same_actions,
    translation_invariance,
)
from .report import to_json, to_markdown
from .runner import ScenarioReport, SuiteReport, TestRunner
from .statistics import (
    StochasticResult,
    evaluate_stochastic,
    wilson_lower_bound,
    z_for,
)
from .tokens import (
    BudgetExceeded,
    CompressedTrace,
    HeuristicTokenizer,
    ScenarioCost,
    TiktokenTokenizer,
    TokenBudget,
    Tokenizer,
    compress_trace,
    estimate_tokens,
    get_default_tokenizer,
    select_scenarios,
    set_default_tokenizer,
)
from .validators import (
    Always,
    Eventually,
    Finite,
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
    "Always", "Never", "Eventually", "RespondsWithin", "SafetyEnvelope", "Finite",
    "ValidationResult", "Validator", "Violation",
    "FaultInjector", "StuckSensor", "GaussianNoise", "Bias", "DropSensor",
    "Dropout", "Delay", "StuckActuator", "Saturate",
    "evaluate_stochastic", "StochasticResult", "wilson_lower_bound", "z_for",
    "MetamorphicRelation", "MetamorphicResult", "determinism",
    "translation_invariance", "irrelevant_key_invariance", "same_actions",
    "mirror_symmetry", "resource_monotonicity", "key_order_invariance",
    "actions_related_by",
    "ScenarioFuzzer", "FuzzReport", "Counterexample",
    "TokenBudget", "BudgetExceeded", "CompressedTrace", "ScenarioCost",
    "compress_trace", "estimate_tokens", "select_scenarios",
    "Tokenizer", "HeuristicTokenizer", "TiktokenTokenizer",
    "get_default_tokenizer", "set_default_tokenizer",
    "Judge", "Judgment", "PredicateJudge", "KeywordJudge", "AnthropicJudge",
    "SemanticValidator",
    "TestRunner", "ScenarioReport", "SuiteReport",
    "to_markdown", "to_json",
]
