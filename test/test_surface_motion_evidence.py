"""验证部分船体回波及静态、错误关联、本船边缘的运动出生反例。"""

import numpy as np
import pytest

from phase16z_s3_runtime.s3_candidate_repair import CandidateObservation
from phase16z_s3_runtime.s3_track_semantics import TrackEvidenceHistory, motion_supported_semantic_probability


def evaluate(velocity, *, jitter=.45, radius=.65, entity=True, ego_distance=10.):
    history = TrackEvidenceHistory()
    times = np.arange(13) * .2
    for index, t in enumerate(times):
        history.add_observation(CandidateObservation(
            stamp_ns=round((1. + t) * 1e9), observation_id=str(index),
            x_m=velocity[0] * t, y_m=velocity[1] * t + (-1.) ** index * jitter,
            radius_m=radius, support=12, source_sensors=("left", "back"),
            position_std_m=.2, range_m=ego_distance, confidence=.5, entity_evidence=entity))
    stamp = round((1. + times[-1]) * 1e9)
    features = history._recent_motion(stamp)
    features.update(history._recent_shape(stamp))
    features["track_distance_to_ego"] = ego_distance
    return features, motion_supported_semantic_probability(features, .1, supported_probability=.9)


def test_moving_partial_surface_does_not_require_monotone_centroids():
    features, (probability, supported) = evaluate((1.4, .3))
    assert features["recent_line_fit_rms_m"] > .2
    assert features["recent_path_efficiency"] < .9
    assert supported and probability >= .9


@pytest.mark.parametrize("velocity,jitter,radius,entity,distance", [
    ((0., 0.), .45, .65, True, 10.),
    ((1.4, .3), 2.5, .2, True, 10.),
    ((1.4, .3), .45, .65, False, 10.),
    ((1.4, .3), .45, .65, True, 3.6),
])
def test_static_jitter_bad_association_water_and_attached_echo_are_rejected(velocity, jitter, radius, entity, distance):
    _, (probability, supported) = evaluate(velocity, jitter=jitter, radius=radius, entity=entity, ego_distance=distance)
    assert not supported and probability == .1
