from atlas import (
    Action,
    Bias,
    Delay,
    DropSensor,
    Dropout,
    FaultInjector,
    GaussianNoise,
    Observation,
    Saturate,
    StuckActuator,
    StuckSensor,
    rollout,
)
from atlas.faults import NOOP

from .helpers import Walker, make_walk


class Probe:
    """Records exactly what it observed and always drives dx=3."""

    def __init__(self):
        self.seen = []

    def reset(self):
        self.seen = []

    def act(self, obs: Observation) -> Action:
        self.seen.append(dict(obs.state))
        return Action("move", {"dx": 3})


# --- sensor faults corrupt observation, not ground truth ---------------------

def test_stuck_sensor_freezes_observation_but_not_dynamics():
    probe = Probe()
    wrapped = FaultInjector(sensor_faults=[StuckSensor("pos")]).wrap(probe)
    trace = rollout(wrapped, make_walk(goal=6, max_steps=10))
    # The agent always saw the frozen initial pos...
    assert all(s["pos"] == 0 for s in probe.seen)
    # ...but the real trajectory (recorded observations) actually advanced.
    real_positions = [step.observation.state["pos"] for step in trace]
    assert real_positions == sorted(real_positions) and real_positions[-1] > 0


def test_gaussian_noise_perturbs_only_target_key():
    probe = Probe()
    wrapped = FaultInjector(sensor_faults=[GaussianNoise("pos", sigma=2.0)], seed=1).wrap(probe)
    rollout(wrapped, make_walk(goal=4))
    first = probe.seen[0]
    assert first["pos"] != 0  # noise moved it off the true value
    assert first["goal"] == 4 and first["fuel"] == 100  # untouched channels


def test_bias_and_dropsensor_are_deterministic_per_seed():
    def observed(seed):
        p = Probe()
        rollout(
            FaultInjector(sensor_faults=[Bias("pos", 5.0), DropSensor("fuel", prob=0.5)], seed=seed).wrap(p),
            make_walk(goal=4),
        )
        return p.seen

    assert observed(3) == observed(3)          # reproducible
    assert observed(3)[0]["pos"] == 5.0         # bias applied to true 0


# --- actuator faults corrupt the action the environment receives -------------

def test_dropout_replaces_some_actions_with_noop_and_is_seeded():
    def run(seed):
        return rollout(
            FaultInjector(actuator_faults=[Dropout(0.5)], seed=seed).wrap(Walker()),
            make_walk(goal=5, max_steps=40),
        )

    t1, t2 = run(11), run(11)
    assert [a.name for a in t1.actions()] == [a.name for a in t2.actions()]  # deterministic
    assert any(a == NOOP for a in t1.actions())  # some commands were dropped


def test_delay_emits_noops_while_buffer_fills():
    trace = rollout(
        FaultInjector(actuator_faults=[Delay(steps=2)]).wrap(Walker()),
        make_walk(goal=5, max_steps=40),
    )
    assert trace.actions()[0] == NOOP and trace.actions()[1] == NOOP
    assert any(a.name == "move" for a in trace.actions())


def test_saturate_clamps_actuator_parameter():
    trace = rollout(
        FaultInjector(actuator_faults=[Saturate("dx", lo=-1, hi=1)]).wrap(Probe()),
        make_walk(goal=6, max_steps=10),
    )
    assert all(a.params.get("dx") == 1 for a in trace.actions() if a.name == "move")


def test_stuck_actuator_repeats_first_command_forever():
    class Hesitant(Walker):
        # Waits one tick before driving — a jammed actuator freezes on that
        # idle command, so the vehicle never proceeds.
        def reset(self):
            self._first = True

        def act(self, obs):
            if self._first:
                self._first = False
                return Action("wait")
            return super().act(obs)

    trace = rollout(
        FaultInjector(actuator_faults=[StuckActuator()]).wrap(Hesitant()),
        make_walk(goal=3, max_steps=10),
    )
    assert trace.truncated  # stuck on "wait": no progress
    assert {str(a) for a in trace.actions()} == {"wait"}


def test_wrap_leaves_original_agent_and_scenario_reusable():
    agent = Walker()
    scenario = make_walk(goal=5)
    clean = rollout(agent, scenario)
    rollout(FaultInjector(actuator_faults=[Dropout(0.9)], seed=2).wrap(agent), scenario)
    again = rollout(agent, scenario)  # unwrapped agent unaffected
    assert [str(a) for a in clean.actions()] == [str(a) for a in again.actions()]
