#!/usr/bin/env bash
set -Eeo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-${root}/runtime_output/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${out}/perception" "${out}/controller"
source /opt/ros/jazzy/setup.bash
source "${root}/perception_ws/install/setup.bash"
source "${root}/control_ws/install/setup.bash"
perception_pid=''
controller_pid=''
cleanup() {
  set +e
  for pid in "${controller_pid}" "${perception_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${controller_pid}" "${perception_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM
setsid ros2 launch phase17_perception_stack full_perception.launch.py \
  enable_bev:=false enable_berthing:=true enable_dynamic_speed:=true \
  receipt_dir:="${out}/perception" >"${out}/perception.log" 2>&1 & perception_pid=$!
sleep 3
setsid ros2 run competition_user berthing_node --ros-args \
  -p runtime_output_dir:="${out}/controller" & controller_pid=$!
wait "${controller_pid}"
