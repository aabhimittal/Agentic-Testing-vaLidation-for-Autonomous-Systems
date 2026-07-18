from atlas import determinism, irrelevant_key_invariance, translation_invariance

from .helpers import FlakyWalker, Walker, make_walk


def test_determinism_passes_for_pure_agent():
    assert determinism().check(Walker(), make_walk()).passed


def test_determinism_catches_state_leak_across_episodes():
    result = determinism().check(FlakyWalker(), make_walk())
    assert not result.passed
    assert "relation violated" in result.detail


def test_translation_invariance_passes_for_relative_reasoner():
    rel = translation_invariance(("pos", "goal"), offset=100.0)
    assert rel.check(Walker(), make_walk()).passed


def test_translation_transform_actually_shifts_state():
    rel = translation_invariance(("pos", "goal"), offset=100.0)
    shifted = rel.transform(make_walk(goal=5))
    assert shifted.initial_state["pos"] == 100.0
    assert shifted.initial_state["goal"] == 105.0
    assert shifted.initial_state["fuel"] == 100  # non-coordinate key untouched


def test_irrelevant_key_invariance():
    rel = irrelevant_key_invariance("weather", value="storm")
    assert rel.check(Walker(), make_walk()).passed
    assert rel.transform(make_walk()).initial_state["weather"] == "storm"


class WeatherSensitiveWalker(Walker):
    """Reads a key it shouldn't: freezes whenever 'weather' is present."""

    def act(self, obs):
        if "weather" in obs.state:
            from atlas import Action

            return Action("stop")
        return super().act(obs)


def test_irrelevant_key_invariance_catches_coupling():
    rel = irrelevant_key_invariance("weather", value="storm")
    assert not rel.check(WeatherSensitiveWalker(), make_walk()).passed
