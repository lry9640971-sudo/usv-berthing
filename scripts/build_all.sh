#!/usr/bin/env bash
set -Eeo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
cd "${root}/perception_ws"
colcon build --symlink-install --parallel-workers 2
source "${root}/perception_ws/install/setup.bash"
cd "${root}/control_ws"
colcon build --packages-select competition_user --symlink-install
echo "构建完成"
