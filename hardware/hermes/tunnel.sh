#!/usr/bin/env bash
set -euo pipefail

# 开发机运行：转发设备、实时导航，或随车录制后下载回放。
# 密码由 SSH 交互读取；也可以使用已有 SSH 公钥认证。
mode="${1:-}"
case "$mode" in
    "") ;;
    --run|--record) shift ;;
    -h|--help)
        echo "用法: bash hardware/hermes/tunnel.sh [--run|--record <hermes 导航参数...>]"
        echo "无参数：转发相机和底盘，供开发机运行导航。"
        echo "--run：随车端运行导航，默认开启 Rerun 并转发到本机浏览器。"
        echo "--record：随车端仅录制，导航退出后下载 RRD 和 JSONL，并用本机 Rerun 打开。"
        exit 0
        ;;
    *) echo "未知参数：$mode；使用 --help 查看用法。" >&2; exit 2 ;;
esac

onboard_host="${ROBOT_NAV_ONBOARD_HOST:-hri@10.113.45.27}"
hermes_host="${ROBOT_NAV_HERMES_HOST:-192.168.11.1}"
hermes_port="${ROBOT_NAV_HERMES_PORT:-1448}"
local_hermes_port="${ROBOT_NAV_LOCAL_HERMES_PORT:-11448}"
local_camera_socket="${ROBOT_NAV_LOCAL_CAMERA_SOCKET:-/tmp/robot-nav-camera.sock}"
remote_camera_socket="${ROBOT_NAV_REMOTE_CAMERA_SOCKET:-/tmp/rgbd_pose.ipc}"
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
ssh_options+=(
    -o ConnectTimeout=10
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
)

if [[ "$mode" == "--run" || "$mode" == "--record" ]]; then
    onboard_repo="${ROBOT_NAV_ONBOARD_REPO:-/home/hri/nav}"
    # SSH 的远端命令会经过 shell；逐项引用，保留目标描述中的空格和特殊字符。
    printf -v remote_command 'cd %q && ' "$onboard_repo"
    if [[ "$mode" == "--run" ]]; then
        remote_command+='exec '
    else
        # 留一个 shell 等待 Python 完整收尾，将 Ctrl+C 的信号退出转成明确退出码。
        remote_command="trap ':' INT; $remote_command"
    fi
    remote_command+='.venv/bin/python -m robot_nav hermes --rerun'
    if (($#)); then
        printf -v navigation_args ' %q' "$@"
        remote_command+="$navigation_args"
    fi
    if [[ "$mode" == "--record" ]]; then
        for arg in "$@"; do
            case "$arg" in
                --no-rerun|--preflight-only|--rerun-save|--rerun-save=*|--run-log|--run-log=*)
                    echo "--record 自动管理录制和日志路径，不能与 $arg 一起使用。" >&2
                    exit 2 ;;
            esac
        done
        rerun_bin="${ROBOT_NAV_RERUN_BIN:-}"
        if [[ -z "$rerun_bin" ]]; then
            rerun_bin="$(command -v rerun || true)"
            for candidate in "$project_root/.venv/bin/rerun" "$HOME/micromamba/envs/robot-nav/bin/rerun"; do
                if [[ -z "$rerun_bin" && -x "$candidate" ]]; then rerun_bin="$candidate"; fi
            done
        fi
        if [[ -z "$rerun_bin" ]] || ! command -v "$rerun_bin" >/dev/null; then
            echo "未找到本机 Rerun；激活 robot-nav 环境或设置 ROBOT_NAV_RERUN_BIN。导航尚未启动。" >&2
            exit 2
        fi
        mkdir -p "$project_root/data/run_logs/onboard"
        replay_dir="$(mktemp -d "$project_root/data/run_logs/onboard/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
        run_id="${replay_dir##*/}"
        remote_rrd="$onboard_repo/data/run_logs/rerun-$run_id.rrd"
        remote_jsonl="$onboard_repo/data/run_logs/hermes-$run_id.jsonl"
        printf -v recording_args ' --rerun --rerun-viewer record --rerun-save %q --run-log %q' "$remote_rrd" "$remote_jsonl"
        remote_command+="$recording_args"
        remote_command+='; navigation_status=$?; exit "$navigation_status"'
        echo "随车端仅录制；正常结束或 Ctrl+C 收尾后自动下载并打开本机回放。"
        echo "本次回放目录：$replay_dir"
        # SSH 的 PTY 将 Ctrl+C 发给远端导航；本地脚本继续完成下载。
        trap ':' INT
        navigation_status=0
        ssh "${ssh_options[@]}" -t "$onboard_host" "$remote_command" || navigation_status=$?
        trap - INT
        if [[ "$navigation_status" == 255 ]]; then
            echo "SSH 连接失败或中断，无法确认远端已结束；不下载可能仍在写入的录制。" >&2
            echo "确认随车程序结束后，可从 $remote_rrd 手动取回。" >&2
            exit "$navigation_status"
        fi
        echo "随车导航已退出（状态 $navigation_status），正在下载日志与录制。"
        # 通过 SSH 读取确定的本次文件，避免误取另一次运行；下载失败不留下伪完整文件。
        download_failed=0
        for remote_file in "$remote_jsonl" "$remote_rrd"; do
            local_file="$replay_dir/${remote_file##*/}"
            printf -v fetch_command 'cat -- %q' "$remote_file"
            if ssh "${ssh_options[@]}" -T "$onboard_host" "$fetch_command" > "$local_file.part"; then
                mv -- "$local_file.part" "$local_file"
            else
                echo "下载未完成：$remote_file；部分数据保存在 $local_file.part" >&2
                download_failed=1
            fi
        done
        if [[ "$download_failed" != 0 ]]; then exit 1; fi
        echo "本机回放：$replay_dir/${remote_rrd##*/}"
        "$rerun_bin" "$replay_dir/${remote_rrd##*/}"
        exit "$navigation_status"
    fi
    remote_command+=' --rerun-viewer web'
    echo "正在启动随车导航并转发 Rerun；服务启动后在本机浏览器打开："
    echo "http://127.0.0.1:9090/?url=ws://127.0.0.1:9877"
    # PTY 将 Ctrl+C 交给随车导航，保留原有取消底盘动作与资源收尾逻辑。
    exec ssh "${ssh_options[@]}" -t \
        -L 127.0.0.1:9090:127.0.0.1:9090 \
        -L 127.0.0.1:9877:127.0.0.1:9877 \
        "$onboard_host" "$remote_command"
fi

exec ssh "${ssh_options[@]}" -N -T \
    -L "127.0.0.1:${local_hermes_port}:${hermes_host}:${hermes_port}" \
    -L "${local_camera_socket}:${remote_camera_socket}" \
    "$onboard_host"
