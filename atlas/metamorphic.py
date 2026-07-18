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


ActionMap = Callable[[Action], Action]


def actions_related_by(mapper: ActionMap) -> TraceRelation:
    """Relation: applying ``mapper`` to every follow-up action recovers the
    source action sequence. The building block for symmetry relations."""

    def relation(source: Trace, follow_up: Trace) -> bool:
        return [mapper(a) for a in follow_up.actions()] == source.actions()

    return relation


def mirror_symmetry(
    coordinate_keys: tuple[str, ...], action_mirror: ActionMap
) -> MetamorphicRelation:
    """Reflect the world about the origin on the listed coordinate keys; every
    decision must be the mirror image of the original run.

    ``action_mirror`` maps an action to its reflection (e.g. move +1 -> move
    -1). Catches policies that hard-code a direction instead of reasoning
    toward the goal — a bug plain determinism can never surface.
    """

    def flip(state: State) -> State:
        return {k: (-v if k in coordinate_keys else v) for k, v in state.items()}

    def transform(s: Scenario) -> Scenario:
        def dynamics(state: State, action: Action) -> State:
            # State is stored in the mirrored frame; unflip, advance with the
            # action expressed in the original frame, reflect the result back.
            nxt = s.dynamics(flip(state), action_mirror(action))
            return flip(nxt)

        return Scenario(
            name=f"{s.name}::mirrored",
            initial_state=flip(s.initial_state),
            dynamics=dynamics,
            done=lambda st: s.done(flip(st)),
            max_steps=s.max_steps,
            tags=s.tags,
        )

    return MetamorphicRelation(
        name="mirror_symmetry",
        transform=transform,
        relation=actions_related_by(action_mirror),
        description="Reflecting the world must reflect every decision.",
    )


def resource_monotonicity(resource_key: str, delta: float = 10.0) -> MetamorphicRelation:
    """Give the agent *more* of a resource; it must not take strictly longer to
    reach the goal. More budget making an agent slower (or fail to finish where
    it previously succeeded) is a robustness bug. Vacuously satisfied when the
    original run never terminated (nothing to be monotone against).
    """

    def transform(s: Scenario) -> Scenario:
        richer = dict(s.initial_state)
        base = richer.get(resource_key, 0)
        richer[resource_key] = base + delta
        return s.with_initial_state(richer, suffix=f"more_{resource_key}")

    def relation(source: Trace, follow_up: Trace) -> bool:
        if source.truncated:
            return True  # original never finished; monotonicity is vacuous
        return len(follow_up) <= len(source)

    return MetamorphicRelation(
        name=f"resource_monotonicity[{resource_key}]",
        transform=transform,
        relation=relation,
        description="More of a resource must not slow the agent down.",
    )


def key_order_invariance() -> MetamorphicRelation:
    """Rebuild every state dict in reversed key order; decisions must not
    change. Catches agents that accidentally depend on dict iteration order
    (e.g. serializing state into a prompt and letting key order leak in).
    """

    def reorder(state: State) -> State:
        return dict(reversed(list(state.items())))

    def transform(s: Scenario) -> Scenario:
        def dynamics(state: State, action: Action) -> State:
            return reorder(s.dynamics(reorder(state), action))

        return Scenario(
            name=f"{s.name}::reordered",
            initial_state=reorder(s.initial_state),
            dynamics=dynamics,
            done=s.done,
            max_steps=s.max_steps,
            tags=s.tags,
        )

    return MetamorphicRelation(
        name="key_order_invariance",
        transform=transform,
        relation=same_actions,
        description="State-dict key order must not affect decisions.",
    )
