#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rules_source="$script_dir/99-robot-nav-realsense.rules"
rules_target="/etc/udev/rules.d/99-robot-nav-realsense.rules"

echo "需要 WSL sudo 权限来安装 L515 udev 规则。"
sudo install -Dm644 "$rules_source" "$rules_target"
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=usb
sudo udevadm trigger --action=change --subsystem-match=video4linux
sudo udevadm trigger --action=change --subsystem-match=hidraw
sudo udevadm settle
echo "L515 USB 权限规则已安装。"
