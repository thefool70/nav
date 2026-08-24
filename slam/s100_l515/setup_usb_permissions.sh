#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rules_source="$script_dir/99-robot-nav-usb.rules"
rules_target="/etc/udev/rules.d/99-robot-nav-usb.rules"

if ! command -v sudo >/dev/null 2>&1; then
    echo "未找到 sudo，无法安装 USB 权限规则。" >&2
    exit 1
fi
if ! command -v udevadm >/dev/null 2>&1; then
    echo "未找到 udevadm，无法加载 USB 权限规则。" >&2
    exit 1
fi

echo "需要 WSL sudo 权限来安装 L515 和 S100 的 udev 规则。"
sudo install -Dm644 "$rules_source" "$rules_target"
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=usb
sudo udevadm trigger --action=change --subsystem-match=hidraw
sudo udevadm trigger --action=change --subsystem-match=tty
sudo udevadm settle

echo "USB 权限规则已安装；现在可以重新运行 S100 + L515 命令。"
