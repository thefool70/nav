#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "用法: slam/s100_l515/run.sh python -m robot_nav s100-l515 --slam ..." >&2
    exit 2
fi

robot_nav_slam_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
robot_nav_slam_pids=()
robot_nav_command=("$@")

prepare_windows_usb() {
    local s100_bus_id
    local windows_script_path

    if [[ "${ROBOT_NAV_SKIP_USB_PREPARE:-0}" == "1" ]]; then
        echo "已跳过 Windows USB 自动转发。"
        return
    fi
    if ! command -v powershell.exe >/dev/null 2>&1; then
        echo "未找到 powershell.exe；请在 WSL 中运行，或设置 ROBOT_NAV_SKIP_USB_PREPARE=1 后手动连接设备。" >&2
        exit 1
    fi
    if ! command -v wslpath >/dev/null 2>&1; then
        echo "未找到 wslpath，无法定位 Windows USB 准备脚本。" >&2
        exit 1
    fi

    windows_script_path="$(wslpath -w "$robot_nav_slam_dir/prepare_usb.ps1")"
    s100_bus_id="${ROBOT_NAV_S100_BUSID:-}"
    if [[ -n "$s100_bus_id" ]]; then
        echo "选择 S100 UART4 的 Windows USB BUSID：$s100_bus_id"
    fi
    powershell.exe \
        -NoProfile \
        -ExecutionPolicy Bypass \
        -File "$windows_script_path" \
        -S100BusId "$s100_bus_id"
}

check_l515_usb() {
    local bus_number
    local candidate
    local device=""
    local device_number
    local failed=0
    local speed_mbps
    local usb_node

    for candidate in /sys/bus/usb/devices/*; do
        [[ -f "$candidate/idVendor" && -f "$candidate/idProduct" ]] || continue
        if [[ "$(<"$candidate/idVendor")" == "8086" && \
              "$(<"$candidate/idProduct")" == "0b64" ]]; then
            device="$candidate"
            break
        fi
    done
    if [[ -z "$device" ]]; then
        echo "USB 转发完成后仍未发现 L515（8086:0b64）。" >&2
        return 1
    fi

    bus_number="$(<"$device/busnum")"
    device_number="$(<"$device/devnum")"
    printf -v usb_node "/dev/bus/usb/%03d/%03d" \
        "$bus_number" "$device_number"
    if [[ ! -r "$usb_node" || ! -w "$usb_node" ]]; then
        echo "当前用户无 L515 USB 读写权限：$usb_node" >&2
        echo "请先运行：slam/s100_l515/setup_usb_permissions.sh" >&2
        failed=1
    fi

    speed_mbps="$(<"$device/speed")"
    if [[ "$speed_mbps" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        if ((10#${speed_mbps%%.*} < 5000)); then
            echo "L515 使用 USB 2（${speed_mbps} Mbit/s），相机流采用 320×240@30。" >&2
        fi
    fi

    return "$failed"
}

check_l515_video_access() {
    local found=0
    local properties
    local video_node

    shopt -s nullglob
    for video_node in /dev/video*; do
        properties="$(
            udevadm info --query=property --name="$video_node" 2>/dev/null
        )" || continue
        if [[ "$properties" != *"ID_VENDOR_ID=8086"* || \
              "$properties" != *"ID_MODEL_ID=0b64"* ]]; then
            continue
        fi
        found=1
        if [[ ! -r "$video_node" || ! -w "$video_node" ]]; then
            echo "当前用户无 L515 视频节点读写权限：$video_node" >&2
            echo "请重新运行：slam/s100_l515/setup_usb_permissions.sh" >&2
            return 1
        fi
    done

    if ((found == 0)); then
        echo "未发现 L515 的 Video4Linux 节点，请检查 WSL uvcvideo 驱动。" >&2
        return 1
    fi
}

find_s100_serial_port() {
    local path
    local path_name
    local -a stable_paths=()
    local -a fallback_paths=()

    shopt -s nullglob
    for path in /dev/serial/by-id/*; do
        path_name="${path##*/}"
        path_name="${path_name,,}"
        if [[ "$path_name" =~ ch910|usb[-_]?enhanced[-_]?serial|wch|1a86 ]]; then
            stable_paths+=("$path")
        fi
    done

    if ((${#stable_paths[@]} == 1)); then
        printf '%s\n' "${stable_paths[0]}"
        return 0
    fi
    if ((${#stable_paths[@]} > 1)); then
        echo "检测到多个可能的 S100 稳定串口，请用 --serial-port 明确指定：" >&2
        printf '  %s\n' "${stable_paths[@]}" >&2
        return 2
    fi

    fallback_paths=(/dev/ttyUSB* /dev/ttyACM*)
    if ((${#fallback_paths[@]} == 1)); then
        printf '%s\n' "${fallback_paths[0]}"
        return 0
    fi
    if ((${#fallback_paths[@]} > 1)); then
        echo "无法从多个串口中确定 S100，请用 --serial-port 明确指定：" >&2
        printf '  %s\n' "${fallback_paths[@]}" >&2
        return 2
    fi
    return 1
}

wait_for_s100_serial_port() {
    local attempt
    local serial_port
    local status

    for ((attempt = 1; attempt <= 40; attempt++)); do
        if serial_port="$(find_s100_serial_port)"; then
            printf '%s\n' "$serial_port"
            return 0
        else
            status=$?
        fi
        if ((status == 2)); then
            return 2
        fi
        sleep 0.25
    done

    echo "USB 转发完成后仍未发现 S100 串口。请检查 usbipd 输出和 /dev/serial/by-id。" >&2
    return 1
}

configure_s100_serial_port() {
    local argument
    local index
    local serial_argument_index=-1
    local serial_port=""

    for ((index = 0; index < ${#robot_nav_command[@]}; index++)); do
        argument="${robot_nav_command[index]}"
        if [[ "$argument" == "--serial-port" ]]; then
            if ((serial_argument_index >= 0)); then
                echo "--serial-port 只能指定一次。" >&2
                exit 2
            fi
            if ((index + 1 >= ${#robot_nav_command[@]})); then
                echo "--serial-port 缺少路径。" >&2
                exit 2
            fi
            serial_argument_index=$((index + 1))
            serial_port="${robot_nav_command[index + 1]}"
        elif [[ "$argument" == --serial-port=* ]]; then
            if ((serial_argument_index >= 0)); then
                echo "--serial-port 只能指定一次。" >&2
                exit 2
            fi
            serial_argument_index=$index
            serial_port="${argument#*=}"
        fi
    done

    if [[ -n "$serial_port" && "$serial_port" != "auto" ]]; then
        echo "使用指定的 S100 串口：$serial_port"
    else
        serial_port="$(wait_for_s100_serial_port)"
        if ((serial_argument_index < 0)); then
            robot_nav_command+=("--serial-port" "$serial_port")
        elif [[ "${robot_nav_command[serial_argument_index]}" == --serial-port=* ]]; then
            robot_nav_command[serial_argument_index]="--serial-port=$serial_port"
        else
            robot_nav_command[serial_argument_index]="$serial_port"
        fi
        echo "自动选择 S100 串口：$serial_port"
    fi

    if [[ ! -r "$serial_port" || ! -w "$serial_port" ]]; then
        echo "当前用户无 S100 串口读写权限：$serial_port" >&2
        echo "请先运行：slam/s100_l515/setup_usb_permissions.sh" >&2
        exit 1
    fi
}

is_calibration_command() {
    local argument
    for argument in "${robot_nav_command[@]}"; do
        if [[ "$argument" == "calibrate-s100-l515" ]]; then
            return 0
        fi
    done
    return 1
}

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

prepare_windows_usb
check_l515_usb
check_l515_video_access
configure_s100_serial_port

if is_calibration_command; then
    "${robot_nav_command[@]}"
    exit $?
fi

camera_arguments=(
    enable_color:=true
    enable_depth:=true
    enable_sync:=true
    align_depth.enable:=true
    publish_tf:=false
    rgb_camera.profile:=320x240x30
    depth_module.profile:=320x240x30
)
camera_serial="${ROBOT_NAV_L515_SERIAL:-}"
previous_argument=""
for argument in "${robot_nav_command[@]}"; do
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

ros2 run depthimage_to_laserscan depthimage_to_laserscan_node \
    --ros-args \
    --params-file "$robot_nav_slam_dir/depthimage_to_laserscan.yaml" \
    --remap depth:=/camera/camera/aligned_depth_to_color/image_raw \
    --remap depth_camera_info:=/camera/camera/color/camera_info \
    --remap scan:=/scan &
robot_nav_slam_pids+=("$!")

ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="$robot_nav_slam_dir/slam_toolbox.yaml" &
robot_nav_slam_pids+=("$!")

"${robot_nav_command[@]}"
