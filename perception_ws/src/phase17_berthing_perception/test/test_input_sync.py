"""点云缺路降级选择回归。"""

from phase17_berthing_perception.input_sync import select_fresh_inputs


def test_three_fresh_inputs_are_usable_when_forward_is_stale() -> None:
    selected = select_fresh_inputs(
        {"front": 2_000, "left": 1_950, "right": 1_900, "back": 1_850, "forward": 800},
        minimum_inputs=3,
        maximum_age_ns=350,
    )
    assert selected is not None
    names, anchor, span = selected
    assert names == ("back", "front", "left", "right")
    assert anchor == 2_000
    assert span == 150


def test_two_fresh_inputs_are_rejected() -> None:
    assert select_fresh_inputs(
        {"front": 2_000, "left": 1_900, "right": 1_000},
        minimum_inputs=3,
        maximum_age_ns=350,
    ) is None
