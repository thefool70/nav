#!/usr/bin/env bash
set -euo pipefail

# 仅在随车笔记本执行，调用其原有发布器；不复制或修改相机实现。
exec "${ROBOT_NAV_CAMERA_PYTHON:-python3}" \
    "${ROBOT_NAV_CAMERA_PUBLISHER:-/home/hri/data_publisher/rgbd_pose_publisher.py}" "$@"
