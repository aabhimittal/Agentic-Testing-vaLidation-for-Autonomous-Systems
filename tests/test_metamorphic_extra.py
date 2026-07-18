from atlas import (
    Action,
    key_order_invariance,
    mirror_symmetry,
    resource_monotonicity,
)

from .helpers import (
    BiWalker,
    OrderSensitiveWalker,
    Walker,
    make_walk,
    mirror_move,
)


# --- mirror_symmetry ---------------------------------------------------------

def test_mirror_symmetry_passes_for_symmetric_agent():
    rel = mirror_symmetry(("pos", "goal"), mirror_move)
    assert rel.check(BiWalker(), make_walk(goal=5)).passed


def test_mirror_symmetry_catches_hardcoded_direction():
    # Walker only ever moves +1; in the reflected world it should move -1.
    rel = mirror_symmetry(("pos", "goal"), mirror_move)
    result = rel.check(Walker(), make_walk(goal=5))
    assert not result.passed
    assert "relation violated" in result.detail


def test_mirror_transform_negates_coordinates():
    rel = mirror_symmetry(("pos", "goal"), mirror_move)
    mirrored = rel.transform(make_walk(goal=5, pos=2))
    assert mirrored.initial_state["pos"] == -2
    assert mirrored.initial_state["goal"] == -5
    assert mirrored.initial_state["fuel"] == 100  # untouched


# --- resource_monotonicity ---------------------------------------------------

def test_resource_monotonicity_passes_when_resource_ignored():
    # Walker ignores fuel: more fuel -> identical length -> monotone.
    rel = resource_monotonicity("fuel", delta=50)
    assert rel.check(Walker(), make_walk(goal=5)).passed


def test_resource_monotonicity_vacuous_when_source_truncated():
    # Goal unreachable within budget -> source truncated -> vacuously true.
    rel = resource_monotonicity("fuel", delta=50)
    assert rel.check(Walker(), make_walk(goal=999, max_steps=5)).passed


def test_resource_monotonicity_catches_dawdling():
    class Dawdler(Walker):
        # Wastes a step up front whenever fuel is plentiful.
        def reset(self):
            self._wasted = False

        def act(self, obs):
            if obs.state["fuel"] > 120 and not self._wasted:
                self._wasted = True
                return Action("idle")
            return super().act(obs)

    rel = resource_monotonicity("fuel", delta=50)  # 100 -> 150 crosses 120
    result = rel.check(Dawdler(), make_walk(goal=5))
    assert not result.passed


# --- key_order_invariance ----------------------------------------------------

def test_key_order_invariance_passes_for_normal_agent():
    assert key_order_invariance().check(Walker(), make_walk(goal=4)).passed


def test_key_order_invariance_catches_iteration_order_dependence():
    result = key_order_invariance().check(OrderSensitiveWalker(), make_walk(goal=4))
    assert not result.passed


def test_key_order_transform_reverses_keys():
    reordered = key_order_invariance().transform(make_walk(goal=4))
    assert list(reordered.initial_state) == list(reversed(list(make_walk(goal=4).initial_state)))
