"""真实危险接近中的稀疏回波应建立对象，背景表面不得同时被放行。"""

import json
from pathlib import Path

import pytest

from phase16z_s3_runtime.s3_track_semantics import motion_supported_semantic_probability

FIXTURE = json.loads((Path(__file__).parent / "fixtures/recorded_sparse_semantics.json").read_text(encoding="utf-8"))


def test_recorded_moving_surface_can_confirm_before_near_collision():
    item = FIXTURE["moving_surface"]["input"]
    features = item["features"]
    assert features["recent_entity_evidence_count"] == 0
    assert features["recent_line_fit_rms_m"] > .2
    probability, supported = motion_supported_semantic_probability(
        features, item["model_probability"], supported_probability=.9,
        classification=item["estimate"]["classification"])
    assert supported
    assert probability == pytest.approx(.9)


@pytest.mark.parametrize("example", FIXTURE["background_surfaces"])
def test_recorded_background_motion_is_not_enough_to_confirm_boat(example):
    item = example["input"]
    probability, supported = motion_supported_semantic_probability(
        item["features"], item["model_probability"], supported_probability=.9,
        classification=item["estimate"]["classification"])
    assert not supported
    assert probability == pytest.approx(item["model_probability"])
