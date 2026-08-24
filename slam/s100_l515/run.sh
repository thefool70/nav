#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "用法: slam/s100_l515/run.sh python -m robot_nav s100-l515 --slam ..." >&2
    exit 2
fi

robot_nav_slam_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
robot_nav_slam_pids=()

stop_slam_processes() {
    local process_id
    for process_id in "${robot_nav_slam_pids[@]}"; do
        kill "$process_id" 2>/dev/null || true
    done
    wait "${robot_nav_slam_pids[@]}" 2>/dev/null || true
}
trap stop_slam_processes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

camera_arguments=(
    enable_color:=true
    enable_depth:=true
    enable_sync:=true
    align_depth.enable:=true
    pointcloud.enable:=true
    pointcloud.allow_no_texture_points:=true
    rgb_camera.color_profile:=640x480x30
    depth_module.depth_profile:=640x480x30
)
camera_serial="${ROBOT_NAV_L515_SERIAL:-}"
previous_argument=""
for argument in "$@"; do
    if [[ "$previous_argument" == "--camera-serial" ]]; then
        camera_serial="$argument"
        break
    fi
    if [[ "$argument" == --camera-serial=* ]]; then
        camera_serial="${argument#*=}"
        break
    fi
    previous_argument="$argument"
done
if [[ -n "$camera_serial" ]]; then
    camera_serial="${camera_serial#_}"
    camera_arguments+=("serial_no:=_${camera_serial}")
fi

ros2 launch realsense2_camera rs_launch.py "${camera_arguments[@]}" &
robot_nav_slam_pids+=("$!")

ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
    --ros-args \
    --params-file "$robot_nav_slam_dir/pointcloud_to_laserscan.yaml" \
    --remap cloud_in:=/camera/camera/depth/color/points \
    --remap scan:=/scan &
robot_nav_slam_pids+=("$!")

ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="$robot_nav_slam_dir/slam_toolbox.yaml" &
robot_nav_slam_pids+=("$!")

"$@"
