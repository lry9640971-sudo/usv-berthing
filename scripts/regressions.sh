#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "${root}/perception_ws/install/setup.bash"
python3 "${root}/scripts/run_regressions.py" "${1:-delivery}"
