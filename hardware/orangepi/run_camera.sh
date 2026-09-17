#!/usr/bin/env bash
set -euo pipefail

# 在香橙派上运行；不安装依赖、不设置底盘地址，也不下发任何运动命令。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${ROBOT_NAV_CAMERA_PYTHON:-python3}" -m robot_nav.adapters.orangepi.server \
    --color-width "${ROBOT_NAV_COLOR_WIDTH:-640}" \
    --color-height "${ROBOT_NAV_COLOR_HEIGHT:-480}" \
    --depth-width "${ROBOT_NAV_DEPTH_WIDTH:-640}" \
    --depth-height "${ROBOT_NAV_DEPTH_HEIGHT:-480}" \
    --fps "${ROBOT_NAV_CAMERA_FPS:-30}" \
    --idle-timeout-s "${ROBOT_NAV_CAMERA_IDLE_S:-900}" \
    "$@"
