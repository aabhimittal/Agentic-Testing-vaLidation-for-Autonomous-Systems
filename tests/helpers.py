"""Shared toy world for tests: a 1-D walker that steps toward a goal."""

from atlas import Action, Observation, Scenario


class Walker:
    """Deterministic agent: step +1 until pos == goal, then stop."""

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        if obs.state["pos"] < obs.state["goal"]:
            return Action("move", {"dx": 1})
        return Action("stop")


class FlakyWalker(Walker):
    """Non-deterministic: alternates action names across calls at the goal.
    Exists to be caught by the determinism metamorphic relation."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        # Deliberately does NOT reset self.calls: state leaks across episodes.
        pass

    def act(self, obs: Observation) -> Action:
        self.calls += 1
        if self.calls % 2 == 0 and obs.state["pos"] < obs.state["goal"]:
            return Action("creep", {"dx": 1})
        return super().act(obs)


def walk_dynamics(state: dict, action: Action) -> dict:
    s = dict(state)
    if action.name in ("move", "creep"):
        s["pos"] += action.params["dx"]
        s["fuel"] = s.get("fuel", 100) - 1
    return s


def make_walk(goal: int = 5, pos: int = 0, max_steps: int = 20) -> Scenario:
    return Scenario(
        name=f"walk_to_{goal}",
        initial_state={"pos": pos, "goal": goal, "fuel": 100},
        dynamics=walk_dynamics,
        done=lambda s: s["pos"] >= s["goal"],
        max_steps=max_steps,
    )
