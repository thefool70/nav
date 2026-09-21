#!/usr/bin/env bash
set -euo pipefail

# 开发机运行。SSH 负责认证/加密，相机转发 Unix 套接字，底盘转发 TCP，不解析 Hermes 控制命令。
# 密码由 SSH 交互读取；也可以使用已有 SSH 公钥认证。
onboard_host="${ROBOT_NAV_ONBOARD_HOST:-hri@10.113.45.27}"
hermes_host="${ROBOT_NAV_HERMES_HOST:-192.168.11.1}"
hermes_port="${ROBOT_NAV_HERMES_PORT:-1448}"
local_hermes_port="${ROBOT_NAV_LOCAL_HERMES_PORT:-11448}"
local_camera_socket="${ROBOT_NAV_LOCAL_CAMERA_SOCKET:-/tmp/robot-nav-camera.sock}"
remote_camera_socket="${ROBOT_NAV_REMOTE_CAMERA_SOCKET:-/tmp/ngd_frames.ipc}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
identity="${ROBOT_NAV_SSH_IDENTITY:-$project_root/data/hermes/id_ed25519}"
known_hosts="${ROBOT_NAV_SSH_KNOWN_HOSTS:-$project_root/data/hermes/known_hosts}"
ssh_options=(-F "${ROBOT_NAV_SSH_CONFIG:-/dev/null}")
if [[ -f "$identity" ]]; then
    ssh_options+=(-i "$identity" -o IdentitiesOnly=yes)
fi
if [[ -f "$known_hosts" ]]; then
    ssh_options+=(-o "UserKnownHostsFile=$known_hosts" -o StrictHostKeyChecking=yes)
fi
if [[ "${ROBOT_NAV_SSH_BATCH_MODE:-0}" == "1" ]]; then
    ssh_options+=(-o BatchMode=yes)
fi

exec ssh "${ssh_options[@]}" -N -T \
    -o ConnectTimeout=10 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=2 \
    -L "127.0.0.1:${local_hermes_port}:${hermes_host}:${hermes_port}" \
    -L "${local_camera_socket}:${remote_camera_socket}" \
    "$onboard_host"
