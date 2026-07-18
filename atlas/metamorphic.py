"""Metamorphic testing for agents.

Autonomous agents rarely have an oracle for "the correct action" — but we can
still state how the agent's behaviour must *relate* across transformed inputs.
A MetamorphicRelation transforms a scenario and asserts a relation between the
original and follow-up traces. No ground truth required.

Built-in relations:
- determinism: same scenario twice -> identical action sequence
- translation_invariance: shifting all coordinates by a constant offset must
  not change the action sequence (agents should reason relatively)
- irrelevant_key_invariance: adding a state key the agent must ignore must
  not change behaviour (catches accidental coupling / prompt-injection-like
  sensitivity in learned policies)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .core import Action, AgentUnderTest, Scenario, State, Trace, rollout

ScenarioTransform = Callable[[Scenario], Scenario]
TraceRelation = Callable[[Trace, Trace], bool]


@dataclass
class MetamorphicResult:
    relation: str
    scenario: str
    passed: bool
    detail: str = ""


@dataclass
class MetamorphicRelation:
    """A (transform, relation) pair checked against a scenario."""

    name: str
    transform: ScenarioTransform
    relation: TraceRelation
    description: str = ""

    def check(self, agent: AgentUnderTest, scenario: Scenario) -> MetamorphicResult:
        source = rollout(agent, scenario)
        follow_up = rollout(agent, self.transform(scenario))
        passed = self.relation(source, follow_up)
        detail = "" if passed else (
            f"relation violated: source ran {len(source)} steps "
            f"({_sig(source)!r}...), follow-up ran {len(follow_up)} steps "
            f"({_sig(follow_up)!r}...)"
        )
        return MetamorphicResult(self.name, scenario.name, passed, detail)


def _sig(trace: Trace, n: int = 5) -> str:
    return ",".join(str(a) for a in trace.actions()[:n])


def same_actions(a: Trace, b: Trace) -> bool:
    return a.actions() == b.actions()


def determinism() -> MetamorphicRelation:
    return MetamorphicRelation(
        name="determinism",
        transform=lambda s: s.with_initial_state(s.initial_state, suffix="rerun"),
        relation=same_actions,
        description="Two identical runs must produce identical action sequences.",
    )


def translation_invariance(
    coordinate_keys: tuple[str, ...], offset: float = 7.0
) -> MetamorphicRelation:
    """Shift every listed coordinate key by ``offset``; behaviour must not change."""

    def transform(s: Scenario) -> Scenario:
        shifted: State = {
            k: (v + offset if k in coordinate_keys else v)
            for k, v in s.initial_state.items()
        }

        def shifted_dynamics(state: State, action: Action) -> State:
            # Unshift, apply the original dynamics, reshift — so the world is
            # the same world, just expressed in translated coordinates.
            local = {k: (v - offset if k in coordinate_keys else v) for k, v in state.items()}
            nxt = s.dynamics(local, action)
            return {k: (v + offset if k in coordinate_keys else v) for k, v in nxt.items()}

        def shifted_done(state: State) -> bool:
            local = {k: (v - offset if k in coordinate_keys else v) for k, v in state.items()}
            return s.done(local)

        return Scenario(
            name=f"{s.name}::translated",
            initial_state=shifted,
            dynamics=shifted_dynamics,
            done=shifted_done,
            max_steps=s.max_steps,
            tags=s.tags,
        )

    return MetamorphicRelation(
        name="translation_invariance",
        transform=transform,
        relation=same_actions,
        description="Translating all coordinates must leave decisions unchanged.",
    )


def irrelevant_key_invariance(key: str, value: Any = 42) -> MetamorphicRelation:
    """Inject a state key the agent has no business reading."""

    def transform(s: Scenario) -> Scenario:
        seeded = dict(s.initial_state)
        seeded[key] = value

        def dynamics(state: State, action: Action) -> State:
            nxt = s.dynamics({k: v for k, v in state.items() if k != key}, action)
            nxt[key] = value
            return nxt

        return Scenario(
            name=f"{s.name}::+{key}",
            initial_state=seeded,
            dynamics=dynamics,
            done=lambda st: s.done({k: v for k, v in st.items() if k != key}),
            max_steps=s.max_steps,
            tags=s.tags,
        )

    return MetamorphicRelation(
        name=f"irrelevant_key_invariance[{key}]",
        transform=transform,
        relation=same_actions,
        description="An irrelevant state key must not alter decisions.",
    )
