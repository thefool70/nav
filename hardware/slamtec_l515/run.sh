#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "用法: hardware/slamtec_l515/run.sh python -m robot_nav slamtec-l515 ..." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
realsense_dir="$(cd -- "$script_dir/../realsense" && pwd)"
rsusb_prefix="$realsense_dir/.rsusb"
l515_vendor_id="8086"
l515_product_id="0b64"
l515_enumeration_attempts=60
l515_permission_attempts=20
l515_enumeration_interval_s=0.25

# 项目中的模型和 Ultralytics 权重均使用相对项目根目录的固定位置。
cd "$project_root"
export YOLO_CONFIG_DIR="$project_root/data/ultralytics"

configure_rsusb() {
    if [[ ! -f "$rsusb_prefix/lib/librealsense2.so.2.54.1" ]] || \
       ! compgen -G "$rsusb_prefix/python/pyrealsense2*.so" >/dev/null; then
        echo "缺少 WSL 所需的 librealsense RSUSB 后端。请先运行：" >&2
        echo "  hardware/realsense/build_rsusb.sh" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$rsusb_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$rsusb_prefix/python${PYTHONPATH:+:$PYTHONPATH}"
}

invoke_windows_powershell() {
    local powershell_path="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if [[ ! -f "$powershell_path" ]]; then
        echo "未找到 Windows PowerShell，无法自动转发 L515。" >&2
        exit 1
    fi
    if [[ -e /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
        "$powershell_path" "$@"
    else
        /init "$powershell_path" "$@"
    fi
}

prepare_l515_usb() {
    local windows_script_path
    if [[ "${ROBOT_NAV_SKIP_USB_PREPARE:-0}" == "1" ]]; then
        echo "已跳过 Windows L515 USB 自动转发。"
        return
    fi
    windows_script_path="$(wslpath -w "$realsense_dir/prepare_l515.ps1")"
    invoke_windows_powershell \
        -NoProfile \
        -NonInteractive \
        -ExecutionPolicy Bypass \
        -File "$windows_script_path"
}

find_l515_sysfs_device() {
    local candidate
    for candidate in /sys/bus/usb/devices/*; do
        [[ -f "$candidate/idVendor" && -f "$candidate/idProduct" ]] || continue
        if [[ "$(<"$candidate/idVendor")" == "$l515_vendor_id" && \
              "$(<"$candidate/idProduct")" == "$l515_product_id" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

check_l515_access() {
    local attempt
    local device=""
    local usb_node

    for ((attempt = 1; attempt <= l515_enumeration_attempts; attempt++)); do
        if device="$(find_l515_sysfs_device)"; then
            break
        fi
        if ((attempt == 1)); then
            echo "正在等待 WSL 枚举 L515……"
        fi
        sleep "$l515_enumeration_interval_s"
    done
    if [[ -z "$device" ]]; then
        echo "USB 转发后等待 15 秒仍未发现 L515（8086:0b64）。" >&2
        exit 1
    fi
    printf -v usb_node "/dev/bus/usb/%03d/%03d" \
        "$(<"$device/busnum")" "$(<"$device/devnum")"
    if [[ ! -r "$usb_node" || ! -w "$usb_node" ]]; then
        echo "正在等待 L515 udev 权限规则生效……"
        for ((attempt = 1; attempt <= l515_permission_attempts; attempt++)); do
            sleep "$l515_enumeration_interval_s"
            if [[ -r "$usb_node" && -w "$usb_node" ]]; then
                break
            fi
        done
    fi
    if [[ ! -r "$usb_node" || ! -w "$usb_node" ]]; then
        echo "当前用户无 L515 USB 读写权限：$usb_node" >&2
        echo "请先运行：hardware/realsense/setup_usb_permissions.sh" >&2
        exit 1
    fi
    echo "L515 已接入：USB $("$realsense_dir/read_usb_speed.sh" "$device")。"
}

configure_rsusb
prepare_l515_usb
check_l515_access
exec "$@"
