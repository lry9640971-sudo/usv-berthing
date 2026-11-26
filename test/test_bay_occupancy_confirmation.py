"""单帧噪声、持续占用和失效不能混淆，物理规划入口仍使用原完整地图。"""

import json
from pathlib import Path

from competition_user.closed_loop.dynamics import VesselState
from competition_user.closed_loop.mission import Bay, Mission, MissionState
from test_rebuilt_mission import payload


def selected_mission():
    mission = Mission()
    mission.accept_bays(payload(1.), 1.)
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 1., can_complete=False)
    return mission


def test_recorded_single_frame_pulses_preserve_actual_final_bay():
    data = json.loads((Path(__file__).parent / "fixtures/bay1_single_occupancy_pulses_483.json").read_text(encoding="utf-8"))
    initial = data["initial"]
    mission = Mission()
    bay = initial["selected_bay"]
    mission.bay = Bay(bay["bay_id"], tuple(bay["entry_pose"]), tuple(bay["final_pose"]))
    mission.state = MissionState(initial["state"])
    mission.bay_stamp = initial["bay_measurement_stamp_sim_s"]
    mission.geometry_stamp = initial["geometry_stamp_sim_s"]
    vessel = VesselState(**initial["vessel"])
    occupied = 0
    for p in data["payloads"]:
        now = max(initial["stamp_sim_s"], p["stamp_ns"] / 1e9 + .05)
        occupied += p["candidates"][0]["occupancy_status"] == "OCCUPIED"
        mission.accept_bays(p, now)
        mission.update(vessel, now, can_complete=False)
        assert mission.bay.bay_id == 1
        assert mission.state == MissionState.FINAL
        assert mission.guidance(vessel).feedback == "terminal"
    assert occupied == 2
    assert not mission.pending_occupancy
    # 真实持续占用的2号仍被确认并排除，不能把三个泊位都强行留成FREE。
    assert mission.occupied_measurements[2][1] == 2
    assert 2 not in [bay.bay_id for bay in mission.free_bays]


def test_duplicate_or_old_packet_cannot_confirm_occupancy():
    mission = selected_mission()
    p = payload(1.1, ("OCCUPIED", "FREE", "FREE"))
    mission.accept_bays(p, 1.1)
    mission.accept_bays(p, 1.2)
    mission.accept_bays(payload(1.05, ("OCCUPIED", "FREE", "FREE")), 1.2)
    assert mission.bay.bay_id == 1
    assert mission.occupied_measurements[1][1] == 1
    assert mission.bay_stamp == 1.


def test_unknown_breaks_streak_without_erasing_unresolved_occupancy():
    mission = selected_mission()
    for stamp, status in ((1.1, "OCCUPIED"), (1.2, "UNKNOWN"), (1.3, "OCCUPIED")):
        mission.accept_bays(payload(stamp, (status, "FREE", "FREE")), stamp)
        assert mission.bay.bay_id == 1
        assert mission.pending_occupancy
    mission.accept_bays(payload(1.4, ("OCCUPIED", "FREE", "FREE")), 1.4)
    assert mission.bay is None
    assert mission.reason == "BAY_OCCUPIED"


def test_degraded_geometry_cannot_supply_second_occupied_measurement():
    mission = selected_mission()
    mission.accept_bays(payload(1.1, ("OCCUPIED", "FREE", "FREE")), 1.1)
    mission.accept_bays(payload(1.2, ("OCCUPIED", "FREE", "FREE"), status="DEGRADED"), 1.2)
    assert mission.occupied_measurements[1][1] == 0
    assert mission.pending_occupancy
    mission.accept_bays(payload(1.3, ("OCCUPIED", "FREE", "FREE")), 1.3)
    assert mission.bay.bay_id == 1
    assert mission.occupied_measurements[1][1] == 1


def test_pending_occupancy_forbids_completion_and_free_clears_it_immediately():
    mission = selected_mission()
    vessel = VesselState(1.5, 3., .12, .18, 0., .13)
    mission.accept_bays(payload(1.1, ("OCCUPIED", "FREE", "FREE")), 1.1)
    mission.check_completion(vessel, 1.1, True)
    assert mission.state == MissionState.FINAL
    mission.accept_bays(payload(1.2, ("UNKNOWN", "FREE", "FREE")), 1.2)
    mission.check_completion(vessel, 1.2, True)
    assert mission.state == MissionState.FINAL
    mission.accept_bays(payload(1.3), 1.3)
    mission.check_completion(vessel, 1.3, True)
    assert mission.state == MissionState.COMPLETE


def test_a_new_bay_cannot_be_chosen_from_unconfirmed_occupied_packet():
    mission = Mission()
    mission.accept_bays(payload(1., ("OCCUPIED", "FREE", "FREE")), 1.)
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 1.)
    assert mission.bay.bay_id == 2


def test_unconfirmed_occupancy_never_refreshes_the_original_free_lifetime():
    mission = selected_mission()
    for stamp in (1.1, 2., 3., 4.01):
        mission.accept_bays(payload(stamp, ("UNKNOWN" if stamp != 1.1 else "OCCUPIED", "UNKNOWN", "UNKNOWN")), stamp)
    assert mission.bay_stamp == 1.
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 4.01)
    assert mission.bay is None
    assert mission.reason == "BAY_CONFIRMATION_EXPIRED"


def test_occupied_measurements_beyond_original_freshness_cannot_be_paired():
    mission = selected_mission()
    mission.accept_bays(payload(1.1, ("OCCUPIED", "UNKNOWN", "UNKNOWN")), 1.1)
    mission.accept_bays(payload(4.11, ("OCCUPIED", "UNKNOWN", "UNKNOWN")), 4.11)
    assert mission.occupied_measurements[1][1] == 1
    assert mission.bay.bay_id == 1
