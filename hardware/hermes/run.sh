#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "用法: hardware/hermes/run.sh python -m robot_nav hermes ..." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
realsense_dir="$(cd -- "$script_dir/../realsense" && pwd)"
rsusb_prefix="$realsense_dir/.rsusb"
d435i_vendor_id="8086"
d435i_product_id="0b3a"
d435i_enumeration_attempts=60
d435i_permission_attempts=20
d435i_enumeration_interval_s=0.25

# 项目中的模型和 Ultralytics 权重均使用相对项目根目录的固定位置。
cd "$project_root"
export YOLO_CONFIG_DIR="$project_root/data/ultralytics"

configure_rsusb() {
    if [[ ! -f "$rsusb_prefix/lib/librealsense2.so.2.56.5" ]] || \
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
        echo "未找到 Windows PowerShell，无法自动转发 D435i。" >&2
        exit 1
    fi
    if [[ -e /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
        "$powershell_path" "$@"
    else
        /init "$powershell_path" "$@"
    fi
}

prepare_d435i_usb() {
    local windows_script_path
    if [[ "${ROBOT_NAV_SKIP_USB_PREPARE:-0}" == "1" ]]; then
        echo "已跳过 Windows D435i USB 自动转发。"
        return
    fi
    windows_script_path="$(wslpath -w "$realsense_dir/prepare_d435i.ps1")"
    invoke_windows_powershell \
        -NoProfile \
        -NonInteractive \
        -ExecutionPolicy Bypass \
        -File "$windows_script_path"
}

find_d435i_sysfs_device() {
    local candidate
    for candidate in /sys/bus/usb/devices/*; do
        [[ -f "$candidate/idVendor" && -f "$candidate/idProduct" ]] || continue
        if [[ "$(<"$candidate/idVendor")" == "$d435i_vendor_id" && \
              "$(<"$candidate/idProduct")" == "$d435i_product_id" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

check_d435i_access() {
    local attempt
    local device=""
    local usb_node

    for ((attempt = 1; attempt <= d435i_enumeration_attempts; attempt++)); do
        if device="$(find_d435i_sysfs_device)"; then
            break
        fi
        if ((attempt == 1)); then
            echo "正在等待 WSL 枚举 D435i……"
        fi
        sleep "$d435i_enumeration_interval_s"
    done
    if [[ -z "$device" ]]; then
        echo "USB 转发后等待 15 秒仍未发现 D435i（8086:0b3a）。" >&2
        exit 1
    fi
    printf -v usb_node "/dev/bus/usb/%03d/%03d" \
        "$(<"$device/busnum")" "$(<"$device/devnum")"
    if [[ ! -r "$usb_node" || ! -w "$usb_node" ]]; then
        echo "正在等待 D435i udev 权限规则生效……"
        for ((attempt = 1; attempt <= d435i_permission_attempts; attempt++)); do
            sleep "$d435i_enumeration_interval_s"
            if [[ -r "$usb_node" && -w "$usb_node" ]]; then
                break
            fi
        done
    fi
    if [[ ! -r "$usb_node" || ! -w "$usb_node" ]]; then
        echo "当前用户无 D435i USB 读写权限：$usb_node" >&2
        echo "请先运行：hardware/realsense/setup_usb_permissions.sh" >&2
        exit 1
    fi
    echo "D435i 已接入：USB $("$realsense_dir/read_usb_speed.sh" "$device")。"
}

configure_rsusb
prepare_d435i_usb
check_d435i_access
exec "$@"
