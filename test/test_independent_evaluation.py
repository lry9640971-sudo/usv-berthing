"""防止验收把有文件、未知接触或错误泊位编号当成完成。"""

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from official_evaluation import VELOCITY_REFERENCE, completion_snapshot, evaluate
import json


def inputs(bay_id):
    manifest = {"dock_pose": [10., 20., 0., 0., 0., math.pi / 2.],
                "static_obstacle_names": ["pillar0"], "dynamic_obstacle_names": ["boat0"]}
    monitor = {"truth_stamp_sim_s": 40., "truth_position_xyyaw": [10. - (3. + 6. * (bay_id - 1)), 21.5, 0.],
               "truth_speed_mps": .05,
               "counts": {"contact_stream_messages": 100}, "events": []}
    completion = {"task_completed": True, "bay_id": bay_id, "max_single_plan_duration_s": .1}
    status = {"elapsed_sim_s": 40.}
    return manifest, monitor, completion, status


@pytest.mark.parametrize("bay_id", (1, 2, 3))
def test_three_official_bay_numbers_transform_to_correct_target(bay_id):
    values = inputs(bay_id)
    result = evaluate(*values, 60., "CONTROLLER_COMPLETION")
    assert result["independent_final_position_error_m"] == pytest.approx(0.)
    assert result["official_single_run_passed"]
    assert result["contact_speed_evidence"] == "NOT_OBSERVED"


def test_missing_completion_speed_and_excessive_observed_contact_reject():
    values = inputs(1)
    values[1]["truth_speed_mps"] = None
    assert not evaluate(*values, 60., "CONTROLLER_COMPLETION")["official_single_run_passed"]
    values[1]["truth_speed_mps"] = .05
    values[1]["events"] = [{"category": "berth", "first_speed_mps": .21}]
    assert not evaluate(*values, 60., "CONTROLLER_COMPLETION")["gates"]["contact_speed"]


def test_false_completion_and_unknown_contact_speed_cannot_pass():
    values = inputs(1)
    values[1]["events"] = [{"category": "berth", "first_speed_mps": None}]
    values[2]["task_completed"] = False
    result = evaluate(*values, 60., "COMPLETION_REJECTED")
    assert not result["gates"]["controller_reported_completion"]
    assert not result["gates"]["contact_speed"]


def test_control_cycle_overrun_rejects_otherwise_successful_result():
    values = inputs(1)
    values[1]["events"] = [{"category": "berth", "first_speed_mps": .1}]
    values[3]["max_control_cycle_s"] = .51
    result = evaluate(*values, 60., "CONTROLLER_COMPLETION")
    assert not result["gates"]["maximum_planning_and_control_s"]
    assert not result["official_single_run_passed"]


def test_completion_uses_same_instant_truth_and_not_later_shutdown_drift():
    values = inputs(1)
    values[2]["stamp_sim_s"] = 40.
    values[1]["completion_truth"] = {"report_stamp_sim_s": 40., "truth_stamp_sim_s": 40.01,
                                      "position_xy_m": [7., 21.5], "speed_mps": .05, "velocity_reference": VELOCITY_REFERENCE}
    values[1]["truth_position_xyyaw"] = [9., 22., 0.]
    values[1]["truth_speed_mps"] = .4
    result = evaluate(*values, 60., "CONTROLLER_COMPLETION")
    assert result["official_single_run_passed"]
    assert result["independent_final_position_error_m"] == pytest.approx(0.)
    assert result["independent_completion_speed_mps"] == pytest.approx(.05)


@pytest.mark.parametrize("snapshot", (None,
    {"report_stamp_sim_s": 39., "truth_stamp_sim_s": 39., "position_xy_m": [7., 21.5], "speed_mps": .05},
    {"report_stamp_sim_s": 40., "truth_stamp_sim_s": 40.2, "position_xy_m": [7., 21.5], "speed_mps": .05}))
def test_missing_or_wrong_instant_truth_cannot_borrow_final_position(snapshot):
    values = inputs(1)
    values[2]["stamp_sim_s"] = 40.
    values[1]["completion_truth"] = snapshot
    result = evaluate(*values, 60., "CONTROLLER_COMPLETION")
    assert not result["gates"]["independent_position"]
    assert not result["gates"]["independent_completion_speed"]


def test_current_speed_labels_distinguish_actual_braking_from_previous_interval_average():
    fixture = json.loads((Path(__file__).parent/'fixtures/completion_velocity_contract_cases.json').read_text(encoding='utf-8'))
    for case in fixture:
        snapshot = completion_snapshot(case['truth_rows'], case['report']['stamp_sim_s'])
        monitor = {'completion_truth': snapshot, 'truth_stamp_sim_s': snapshot['truth_stamp_sim_s'],
                   'counts': {'contact_stream_messages': 100}, 'events': []}
        result = evaluate(case['manifest'], monitor, case['report'], {}, 800., 'CONTROLLER_COMPLETION')
        assert snapshot['speed_mps'] == pytest.approx(case['expected_current_speed_mps'], abs=1e-9)
        assert result['gates']['independent_completion_speed'] is case['expected_speed_pass']
        if case['source_case'].endswith('012827'):
            assert snapshot['preceding_0p2s_average_speed_mps'] > .20
            assert snapshot['speed_mps'] < .20


def test_instantaneous_speed_requires_actual_bracketing_samples_and_cannot_borrow_average():
    assert completion_snapshot([(39.8, 0., 0.), (40., .04, 0.)], 40.) is None
    values = inputs(1)
    values[2]['stamp_sim_s'] = 40.
    values[1]['completion_truth'] = {'report_stamp_sim_s': 40., 'truth_stamp_sim_s': 40.,
                                    'position_xy_m': [7., 21.5], 'speed_mps': .05}
    assert not evaluate(*values, 60., 'CONTROLLER_COMPLETION')['gates']['independent_completion_speed']
