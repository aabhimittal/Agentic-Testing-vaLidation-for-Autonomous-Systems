"""Core abstractions: agents, scenarios, traces.

An agent under test is anything with an ``act(observation) -> Action`` method.
A Scenario pairs an initial state with a dynamics function; the runner rolls
the agent through it and records a Trace, which validators then inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

State = dict[str, Any]


@dataclass(frozen=True)
class Action:
    """A named action with optional parameters, e.g. Action("move", {"dx": 1})."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class Observation:
    """What the agent sees at a timestep. ``state`` is a flat key/value dict."""

    state: State
    timestep: int


@dataclass(frozen=True)
class Step:
    """One transition: the observation the agent saw and the action it took."""

    observation: Observation
    action: Action


@dataclass
class Trace:
    """The full recorded execution of one scenario."""

    scenario_name: str
    steps: list[Step] = field(default_factory=list)
    final_state: State = field(default_factory=dict)
    truncated: bool = False  # hit max_steps without the scenario terminating

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def states(self) -> list[State]:
        """All observed states plus the final state."""
        return [s.observation.state for s in self.steps] + [self.final_state]

    def actions(self) -> list[Action]:
        return [s.action for s in self.steps]


@runtime_checkable
class AgentUnderTest(Protocol):
    """Minimal interface a system under test must expose."""

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...


# Dynamics: (state, action) -> next state. Termination: state -> done?
Dynamics = Callable[[State, Action], State]
DonePredicate = Callable[[State], bool]


@dataclass
class Scenario:
    """A reproducible test situation for an agent.

    ``dynamics`` must be a pure function of (state, action) so scenarios are
    deterministic and metamorphic transforms stay sound.
    """

    name: str
    initial_state: State
    dynamics: Dynamics
    done: DonePredicate
    max_steps: int = 100
    tags: tuple[str, ...] = ()

    def with_initial_state(self, state: State, suffix: str = "variant") -> "Scenario":
        """A copy of this scenario starting from a different state."""
        return Scenario(
            name=f"{self.name}::{suffix}",
            initial_state=dict(state),
            dynamics=self.dynamics,
            done=self.done,
            max_steps=self.max_steps,
            tags=self.tags,
        )


def rollout(agent: AgentUnderTest, scenario: Scenario) -> Trace:
    """Run ``agent`` through ``scenario`` and record the trace."""
    agent.reset()
    state = dict(scenario.initial_state)
    trace = Trace(scenario_name=scenario.name)
    for t in range(scenario.max_steps):
        if scenario.done(state):
            trace.final_state = state
            return trace
        obs = Observation(state=dict(state), timestep=t)
        action = agent.act(obs)
        trace.steps.append(Step(observation=obs, action=action))
        state = scenario.dynamics(state, action)
    trace.final_state = state
    trace.truncated = not scenario.done(state)
    return trace
