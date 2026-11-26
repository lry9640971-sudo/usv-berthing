"""管理单套官方诊断或验收；只清理本轮进程，失败先保存原因和证据。"""

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from official_evaluation import evaluate
from owned_processes import terminate_process_group


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "acceptance", "capture", "actuator", "terminal_probe"), default="diagnostic")
    parser.add_argument("--max-wall", type=float, default=180.)
    parser.add_argument("--diagnostic-pose", type=float, nargs=6)
    parser.add_argument("--actuator-period", type=float, default=.05)
    parser.add_argument("--actuator-yaw-brake", type=float, default=6.)
    parser.add_argument("--sensor-trace", action="store_true")
    parser.add_argument("--headless-rendering", action="store_true")
    args = parser.parse_args()
    if args.diagnostic_pose is not None and args.mode not in ("capture", "terminal_probe"):
        parser.error("位姿隔离仅允许感知捕获或近泊位闭环诊断，不能进入官方验收")
    if args.mode == "terminal_probe" and args.diagnostic_pose is None:
        parser.error("近泊位诊断须明确给出位姿覆盖，结果不能计正式验收")
    root = Path(__file__).resolve().parent.parent
    case = f"{args.mode}_{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    out = root / "runtime_output" / case
    out.mkdir(parents=True, exist_ok=False)
    (out / "controller").mkdir()
    (out / "perception").mkdir()
    # 每轮 Gazebo 服务隔离，防止同名世界的旧发现记录接到错误实例。
    os.environ["GZ_PARTITION"] = "usv_"+case
    handles, processes = {}, {}
    active = True
    controller_started = None
    controller_finished = None
    reason = "STARTING"
    input_failed_since = None
    motion = deque()

    def stop(*_):
        nonlocal active, reason
        active, reason = False, "USER_OR_SUPERVISOR_SIGNAL"

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(name, command):
        handles[name] = (out / f"{name}.log").open("w", encoding="utf-8")
        processes[name] = subprocess.Popen(command, stdout=handles[name], stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=root)
        atomic_json(out / "owned_processes.json", {key: {"pid": proc.pid, "process_group": proc.pid}
                                                   for key, proc in processes.items()})

    def cleanup(name):
        process = processes.get(name)
        if process is None:
            return
        residual = terminate_process_group(process.pid)
        process.poll()
        if residual:
            atomic_json(out / f"cleanup_{name}_residual.json", residual)

    def control_world(paused):
        response = subprocess.run(["gz", "service", "-s", f"/world/{manifest['world_name']}/control",
                                   "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
                                   "--timeout", "3000", "--req", "pause: " + ("true" if paused else "false")],
                                  capture_output=True, text=True, encoding="utf-8", timeout=6.)
        with (out / "world_control.log").open("a", encoding="utf-8") as handle:
            handle.write(response.stdout + response.stderr)
        if response.returncode != 0 or "data: true" not in response.stdout:
            raise RuntimeError("世界控制服务未确认成功")

    manifest = {}
    try:
        subprocess.run([sys.executable, str(root / "scripts/prepare_scene.py"), str(out), str(args.seed)], check=True)
        manifest = read_json(out / "scene/manifest.json")
        if args.diagnostic_pose is not None:
            manifest["official_initial_pose"] = manifest["wamv_pose"]
            manifest["wamv_pose"] = args.diagnostic_pose
            manifest["diagnostic_initial_pose_override"] = True
            atomic_json(out / "scene/manifest.json", manifest)
        glx = subprocess.run(["glxinfo", "-B"], capture_output=True, text=True, encoding="utf-8", timeout=15.)
        (out / "graphics.log").write_text(glx.stdout + glx.stderr, encoding="utf-8")
        # GPU 厂商不是官方验收条件；探测只记录，实际雷达接线和运行预算仍独立检查。
        if glx.returncode:
            print(json.dumps({"stage": "GRAPHICS_PROBE_UNAVAILABLE", "return_code": glx.returncode}, ensure_ascii=False), flush=True)
        atomic_json(out / "run_config.json", {"mode": args.mode, "seed": args.seed, "root": str(root),
                                              "max_wall_s": args.max_wall, "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
                                              "graphics_probe_return_code": glx.returncode,
                                              "graphics_requested_backend": os.environ.get("GALLIUM_DRIVER"),
                                              "headless_rendering": args.headless_rendering,
                                              "graphics_renderer_probe": next((line.split(":", 1)[1].strip() for line in glx.stdout.splitlines() if line.startswith("OpenGL renderer string:")), None),
                                              "official_sensor_configuration": "original",
                                              "gz_partition": os.environ["GZ_PARTITION"],
                                              "started_wall": datetime.now().isoformat()})
        print(json.dumps({"output": str(out), "stage": "START_SIMULATION", "seed": args.seed}, ensure_ascii=False), flush=True)
        atomic_json(root / "runtime_output/active_run.json", {"output": str(out), "supervisor_pid": os.getpid()})
        launch("simulation", ["ros2", "launch", str(root / "scripts/official_scene.launch.py"),
                              "manifest:=" + str(out / "scene/manifest.json"),
                              "headless_rendering:=" + ("true" if args.headless_rendering else "false")])
        launch("resources", [sys.executable, "-u", str(root / "scripts/resource_monitor.py"),
                             str(out), str(os.getpid())])
        deadline = time.monotonic() + 90.
        while active:
            if processes["simulation"].poll() is not None:
                raise RuntimeError("仿真启动进程退出")
            services = subprocess.run(["gz", "service", "-l"], capture_output=True, text=True,
                                      encoding="utf-8", timeout=6.)
            if f"/world/{manifest['world_name']}/control" in services.stdout:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("仿真世界服务启动超时")
            time.sleep(1.)
        if not active:
            raise RuntimeError("启动被中断")
        launch("monitor", [sys.executable, "-u", str(root / "scripts/official_monitor.py"), str(out)]
               + (["--sensor-trace"] if args.sensor_trace else []))
        launch("perception", ["ros2", "launch", "phase17_perception_stack", "full_perception.launch.py",
                              "enable_bev:=false", "enable_berthing:=true", "enable_dynamic_speed:=true",
                              "enable_gps_velocity:=true", "gps_velocity_ros_topic:=/wamv/sensors/gps/gps/velocity",
                              "gps_velocity_gz_topic:=/world/" + manifest["world_name"] + "/model/wamv/link/wamv/gps_wamv_link/sensor/navsat/navsat",
                              "use_sim_time:=true", "receipt_dir:=" + str(out / "perception")])
        time.sleep(2.)
        controller_started = time.monotonic()
        if args.mode == "actuator":
            launch("actuator", [sys.executable, "-u", str(root / "scripts/actuator_probe.py"), str(out / "actuator"),
                                "--control-period-s", str(args.actuator_period),
                                "--yaw-brake-s", str(args.actuator_yaw_brake)])
        elif args.mode != "capture":
            launch("controller", ["ros2", "run", "competition_user", "berthing_node", "--ros-args",
                                  "-p", "runtime_output_dir:=" + str(out / "controller")])
        control_world(False)
        while active:
            now = time.monotonic()
            elapsed = now - controller_started
            monitor = read_json(out / "monitor.json")
            status = read_json(out / "controller/runtime_status.json")
            completion = read_json(out / "controller/completion.json")
            probe = read_json(out / "actuator/probe_result.json")
            exited = [key for key, process in processes.items() if process.poll() is not None]
            reason = "RUNNING"
            if exited:
                reason = "PROCESS_EXIT:" + ",".join(exited)
            elif args.mode == "actuator" and probe.get("probe_completed"):
                reason = "ACTUATOR_PROBE_COMPLETED"
            elif completion:
                reason = "CONTROLLER_COMPLETION" if completion.get("task_completed") is True else "COMPLETION_REJECTED"
            elif elapsed >= (900. if args.mode == "acceptance" else args.max_wall):
                reason = "WALL_TIME_LIMIT"
            elif elapsed>15. and (monitor.get("truth_position_xyyaw") is None
                    or any(monitor.get("lidar", {}).get(name, {}).get("count", 0)<2 for name in ("back", "forward", "front", "left", "right"))
                    or monitor.get("counts", {}).get("contact_stream_messages", 0)==0):
                reason = "SIMULATION_INPUT_WIRING_NOT_READY"
            elif args.mode != "capture" and (any(row["category"] == "static" for row in monitor.get("events", []))
                    or sum(row["category"] == "dynamic" for row in monitor.get("events", [])) > .05 * len(manifest["dynamic_obstacle_names"])):
                reason = "NON_BERTH_COLLISION"
            elif args.mode != "capture" and max(status.get("max_single_plan_duration_s", 0.), status.get("max_control_cycle_s", 0.)) > .50:
                reason = "PLANNING_OR_CONTROL_BUDGET_EXCEEDED"
            if args.mode == "diagnostic" and reason == "RUNNING" and elapsed > 25.:
                if status.get("input_status") not in ("READY", None):
                    input_failed_since = input_failed_since or now
                    if now - input_failed_since > 12.:
                        reason = "PERSISTENT_INPUT_HOLD:" + status["input_status"]
                else:
                    input_failed_since = None
                pose = monitor.get("truth_position_xyyaw")
                stamp = monitor.get("truth_stamp_sim_s")
                if pose is not None and stamp is not None:
                    if not motion or stamp > motion[-1][0]:
                        motion.append((stamp, *pose[:2]))
                    while motion and stamp - motion[0][0] > 40.:
                        motion.popleft()
                    if motion and stamp - motion[0][0] > 39. and status.get("state") in ("NAVIGATE", "APPROACH", "FINAL"):
                        maximum = max(math.hypot(row[1] - motion[0][1], row[2] - motion[0][2]) for row in motion)
                        if maximum < .8:
                            reason = "PERSISTENT_NAVIGATION_STALL"
            atomic_json(out / "supervisor_status.json", {"stage": reason, "elapsed_controller_wall_s": elapsed,
                                                        "controller": status, "monitor_truth": monitor.get("truth_position_xyyaw"),
                                                        "truth_stamp_sim_s": monitor.get("truth_stamp_sim_s"),
                                                        "contact_events": monitor.get("events", [])})
            if reason != "RUNNING":
                break
            time.sleep(1.)
        controller_finished = time.monotonic()
    except Exception as error:
        reason = "SUPERVISOR_ERROR:" + repr(error)
        (out / "supervisor_error.log").write_text(reason, encoding="utf-8")
        print(reason, flush=True)
    finally:
        controller_finished = controller_finished or time.monotonic()
        # 零推力发布时仿真仍在运行，之后暂停以保存终点真实状态。
        cleanup("controller")
        cleanup("actuator")
        if manifest and processes.get("simulation") and processes["simulation"].poll() is None:
            try:
                control_world(True)
            except Exception:
                pass
        cleanup("perception")
        cleanup("monitor")
        cleanup("simulation")
        cleanup("resources")
        for handle in handles.values():
            handle.close()
        elapsed = 0. if controller_started is None else controller_finished - controller_started
        monitor = read_json(out / "monitor.json")
        completion = read_json(out / "controller/completion.json")
        status = read_json(out / "controller/runtime_status.json")
        if manifest:
            result = evaluate(manifest, monitor, completion, status, elapsed, reason)
        else:
            result = {"official_single_run_passed": False, "stop_reason": reason}
        result["mode"] = args.mode
        if args.mode == "terminal_probe":
            result["official_single_run_passed"] = False
            result["evaluation_scope"] = "isolated_terminal_probe_not_official_acceptance"
        result["process_return_codes"] = {key: process.poll() for key, process in processes.items()}
        atomic_json(out / "result.json", result)
        print(json.dumps({"output": str(out), "result": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
