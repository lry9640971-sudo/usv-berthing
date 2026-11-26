#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "${USV_SIM_SETUP:?请设置官方仿真 install/setup.bash 路径}"
source "${root}/perception_ws/install/setup.bash"
source "${root}/control_ws/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-71}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=6
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
python3 -u "${root}/scripts/official_supervisor.py" "$@"
