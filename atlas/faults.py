"""Fault injection for autonomous-agent testing (FMEA-style).

Real deployments do not hand an agent clean observations and perfectly executed
actions. Sensors freeze, drift, and go noisy; commands are dropped, delayed, or
clipped by actuator limits. A policy that is correct on nominal traces can still
be unsafe the first time a sensor sticks. This module injects those faults
*deterministically* (seeded), so a fault-tolerance run is as reproducible as any
other ATLAS test.

Design: a fault does not alter the environment's ground-truth dynamics — it sits
between the environment and the agent. ``SensorFault`` corrupts what the agent
*observes*; ``ActuatorFault`` corrupts the action *before* it reaches the world.
``FaultInjector.wrap(agent)`` returns an ordinary ``AgentUnderTest`` you can pass
straight to ``rollout`` / ``TestRunner``, so every validator, metamorphic
relation, and the fuzzer all work unchanged under fault conditions.

    injector = FaultInjector(
        sensor_faults=[StuckSensor("battery"), GaussianNoise("pos", sigma=0.5)],
        actuator_faults=[Dropout(0.1), Saturate("dx", -1, 1)],
        seed=7,
    )
    trace = rollout(injector.wrap(agent), scenario)   # validators still apply
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .core import Action, AgentUnderTest, Observation, State

NOOP = Action("noop")


@runtime_checkable
class SensorFault(Protocol):
    """Corrupts the observed state the agent reasons over."""

    def reset(self) -> None: ...

    def corrupt(self, state: State, rng: random.Random) -> State: ...


@runtime_checkable
class ActuatorFault(Protocol):
    """Corrupts the action on its way to the environment."""

    def reset(self) -> None: ...

    def corrupt(self, action: Action, rng: random.Random) -> Action: ...


# --- sensor faults -----------------------------------------------------------

@dataclass
class StuckSensor:
    """A sensor that freezes: after ``at_step`` the observed ``key`` never
    changes again (a classic stuck-at fault). Ground truth keeps moving, so the
    agent is now flying blind on that channel."""

    key: str
    at_step: int = 0
    _frozen: object = field(default=None, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._frozen = None
        self._seen = 0

    def corrupt(self, state: State, rng: random.Random) -> State:
        if self.key not in state:
            return state
        out = dict(state)
        if self._seen >= self.at_step:
            if self._frozen is None:
                self._frozen = state[self.key]
            out[self.key] = self._frozen
        self._seen += 1
        return out


@dataclass
class GaussianNoise:
    """Additive zero-mean Gaussian noise on a numeric sensor channel."""

    key: str
    sigma: float = 1.0

    def reset(self) -> None:
        pass

    def corrupt(self, state: State, rng: random.Random) -> State:
        if self.key not in state or not _is_number(state[self.key]):
            return state
        out = dict(state)
        out[self.key] = state[self.key] + rng.gauss(0.0, self.sigma)
        return out


@dataclass
class Bias:
    """Constant offset (miscalibration/drift) on a numeric sensor channel."""

    key: str
    delta: float

    def reset(self) -> None:
        pass

    def corrupt(self, state: State, rng: random.Random) -> State:
        if self.key not in state or not _is_number(state[self.key]):
            return state
        out = dict(state)
        out[self.key] = state[self.key] + self.delta
        return out


@dataclass
class DropSensor:
    """Intermittent sensor dropout: with probability ``prob`` the channel reads
    a fixed ``fallback`` (default 0) instead of its true value."""

    key: str
    prob: float = 0.1
    fallback: object = 0

    def reset(self) -> None:
        pass

    def corrupt(self, state: State, rng: random.Random) -> State:
        if self.key not in state or rng.random() >= self.prob:
            return state
        out = dict(state)
        out[self.key] = self.fallback
        return out


# --- actuator faults ---------------------------------------------------------

@dataclass
class Dropout:
    """Command dropout: with probability ``prob`` the action is lost and the
    environment receives a no-op instead."""

    prob: float = 0.1
    noop: Action = NOOP

    def reset(self) -> None:
        pass

    def corrupt(self, action: Action, rng: random.Random) -> Action:
        return self.noop if rng.random() < self.prob else action


@dataclass
class Delay:
    """Control latency: every command is delayed by ``steps`` timesteps. The
    first ``steps`` actions the environment sees are no-ops while the buffer
    fills."""

    steps: int = 1
    noop: Action = NOOP
    _buffer: list = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        self._buffer = [self.noop] * max(0, self.steps)

    def corrupt(self, action: Action, rng: random.Random) -> Action:
        if self.steps <= 0:
            return action
        self._buffer.append(action)
        return self._buffer.pop(0)


@dataclass
class StuckActuator:
    """The actuator jams on its first command and repeats it forever."""

    _frozen: Action | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._frozen = None

    def corrupt(self, action: Action, rng: random.Random) -> Action:
        if self._frozen is None:
            self._frozen = action
        return self._frozen


@dataclass
class Saturate:
    """Actuator limit: clamp a numeric action parameter into ``[lo, hi]``.
    Models a command that asks for more travel/torque than the hardware gives."""

    param: str
    lo: float | None = None
    hi: float | None = None

    def reset(self) -> None:
        pass

    def corrupt(self, action: Action, rng: random.Random) -> Action:
        if self.param not in action.params or not _is_number(action.params[self.param]):
            return action
        v = action.params[self.param]
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        if v == action.params[self.param]:
            return action
        params = dict(action.params)
        params[self.param] = v
        return Action(action.name, params)


# --- composition -------------------------------------------------------------

@dataclass
class FaultInjector:
    """Wraps an agent so its observations and actions pass through a fixed,
    seeded fault chain. Deterministic: the same ``seed`` yields the same faults
    on every run, so failures are reproducible from ``(seed, scenario)``."""

    sensor_faults: Sequence[SensorFault] = ()
    actuator_faults: Sequence[ActuatorFault] = ()
    seed: int = 0

    def wrap(self, agent: AgentUnderTest) -> "_FaultyAgent":
        return _FaultyAgent(agent, self.sensor_faults, self.actuator_faults, self.seed)


class _FaultyAgent:
    """An ``AgentUnderTest`` that applies sensor faults before delegating and
    actuator faults after. Not constructed directly — use ``FaultInjector``."""

    def __init__(
        self,
        inner: AgentUnderTest,
        sensor_faults: Sequence[SensorFault],
        actuator_faults: Sequence[ActuatorFault],
        seed: int,
    ) -> None:
        self._inner = inner
        self._sensor = list(sensor_faults)
        self._actuator = list(actuator_faults)
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._inner.reset()
        self._rng = random.Random(self._seed)
        for f in self._sensor:
            f.reset()
        for f in self._actuator:
            f.reset()

    def act(self, observation: Observation) -> Action:
        state = observation.state
        for f in self._sensor:
            state = f.corrupt(state, self._rng)
        seen = Observation(state=state, timestep=observation.timestep)
        action = self._inner.act(seen)
        for f in self._actuator:
            action = f.corrupt(action, self._rng)
        return action


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
