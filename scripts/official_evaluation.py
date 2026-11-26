"""从独立真值和实际接触记录给出验收结果；缺失证据不能当成通过。"""

import math
import numpy as np


VELOCITY_REFERENCE = "centered_40ms_instantaneous_quadratic"


def completion_snapshot(truth_rows, stamp):
    """只评估完成时速度；过去区间平均不能替代减速后的当前速度。"""
    rows = sorted({row[0]: tuple(row) for row in truth_rows}.values())
    if not rows or rows[-1][0] < stamp+.04-1e-9:
        return None
    selected = [row for row in rows if abs(row[0]-stamp) <= .04+1e-9]
    if len(selected) < 3 or not selected[0][0] <= stamp <= selected[-1][0]:
        return None
    times = np.asarray([row[0]-stamp for row in selected])
    xy = np.asarray([row[1:3] for row in selected])
    velocity = np.polyfit(times, xy, 2)[1]
    closest = min(rows, key=lambda row: abs(row[0]-stamp))
    preceding = [row for row in rows if closest[0]-.20 <= row[0] <= closest[0]]
    average = None
    if len(preceding) >= 2:
        average = math.hypot(preceding[-1][1]-preceding[0][1], preceding[-1][2]-preceding[0][2])/(preceding[-1][0]-preceding[0][0])
    return {"report_stamp_sim_s": stamp, "truth_stamp_sim_s": closest[0],
            "position_xy_m": list(closest[1:3]), "speed_mps": float(np.linalg.norm(velocity)),
            "velocity_xy_mps": velocity.tolist(), "velocity_reference": VELOCITY_REFERENCE,
            "velocity_samples": len(selected), "preceding_0p2s_average_speed_mps": average}


def evaluate(manifest, monitor, completion, status, elapsed_wall, reason):
    snapshot = monitor.get("completion_truth")
    report_stamp = completion.get("stamp_sim_s")
    if snapshot is not None and report_stamp is not None:
        if (not math.isclose(snapshot["report_stamp_sim_s"], report_stamp, abs_tol=1e-6, rel_tol=0.)
                or abs(snapshot["truth_stamp_sim_s"] - report_stamp) > .1):
            snapshot = None
    events = monitor.get("events", [])
    collision_counts = {key: sum(row["category"] == key for row in events) for key in ("static", "dynamic")}
    totals = {"static": len(manifest["static_obstacle_names"]), "dynamic": len(manifest["dynamic_obstacle_names"])}
    contact_stream_present = monitor.get("counts", {}).get("contact_stream_messages", 0) > 0
    success = {key: max(0., 1. - collision_counts[key] / totals[key]) if totals[key] and contact_stream_present else None for key in totals}
    bay = status.get("selected_bay") or {}
    bay_id = completion.get("bay_id", bay.get("bay_id"))
    target = None
    position_error = None
    if bay_id in (1, 2, 3):
        # 官方三泊位的标识目标来自原始模型；生产仍只使用精化感知目标。
        dock = manifest["dock_pose"]
        local_x, local_y = 1.5, 3. + 6. * (bay_id - 1)
        c, s = math.cos(dock[5]), math.sin(dock[5])
        target = [dock[0] + c * local_x - s * local_y, dock[1] + s * local_x + c * local_y]
        truth = (snapshot["position_xy_m"] if snapshot is not None else
                 monitor.get("truth_position_xyyaw") if "stamp_sim_s" not in completion else None)
        if truth:
            position_error = math.hypot(truth[0] - target[0], truth[1] - target[1])
    contacts = [row for row in events if row["category"] == "berth"]
    contact_speeds = [row.get("first_speed_mps") for row in contacts]
    # 平台 v1.1 3.2.2 同时记录完成时速度；总方案另约束实际发生的接触瞬间。
    # 不强行制造一次接触；没接触明确标未观测，不能据此伪造接触速度。
    completion_speed = (snapshot["speed_mps"] if snapshot is not None and snapshot.get("velocity_reference") == VELOCITY_REFERENCE else
                        monitor.get("truth_speed_mps") if completion.get("task_completed") is True and "stamp_sim_s" not in completion else None)
    contact_pass = all(value is not None and value <= .20 for value in contact_speeds)
    completion_speed_pass = completion_speed is not None and completion_speed <= .20
    timing = max(completion.get("max_single_plan_duration_s", 0.), status.get("max_single_plan_duration_s", 0.),
                 completion.get("max_control_cycle_s", 0.), status.get("max_control_cycle_s", 0.))
    truth_present = monitor.get("truth_stamp_sim_s") is not None
    gates = {"controller_reported_completion": completion.get("task_completed") is True,
             "independent_position": position_error is not None and position_error <= .20,
             "static_avoidance": truth_present and contact_stream_present and success["static"] == 1.,
             "dynamic_avoidance": truth_present and contact_stream_present and success["dynamic"] is not None and success["dynamic"] >= .95,
             "contact_speed": contact_pass,
             "independent_completion_speed": completion_speed_pass,
             "maximum_planning_and_control_s": timing <= .50,
             "runtime_wall_s": elapsed_wall <= 900.}
    return {"stop_reason": reason, "official_single_run_passed": all(gates.values()) and not manifest.get("diagnostic_initial_pose_override", False), "gates": gates,
            "collision_episodes": collision_counts, "total_obstacles": totals, "avoidance_success_fraction": success,
            "berth_contact_speed_mps": contact_speeds, "contact_speed_evidence": "OBSERVED" if contacts else "NOT_OBSERVED",
            "independent_completion_speed_mps": completion_speed,
            "completion_velocity_reference": VELOCITY_REFERENCE,
            "legacy_preceding_0p2s_speed_mps": snapshot.get("preceding_0p2s_average_speed_mps") if snapshot else None,
            "independent_target_xy_m": target, "independent_final_position_error_m": position_error,
            "elapsed_wall_s": elapsed_wall, "elapsed_sim_s": status.get("elapsed_sim_s"),
            "maximum_planning_and_control_s": timing, "controller_report": completion,
            "monitor_counts": monitor.get("counts", {}), "selected_bay": bay,
            "evaluation_scope": "single_run_independent_evaluation_requires_five_frozen_version_runs"}
