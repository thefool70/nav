#!/usr/bin/env bash
set -euo pipefail

device="${1:?缺少 USB sysfs 设备路径}"
speed="$(<"$device/speed")"
if [[ "$speed" =~ ^[0-9]+([.][0-9]+)?$ ]] && ((10#${speed%%.*} < 5000)); then
    printf '2 (%s Mbit/s)\n' "$speed"
else
    printf '3 (%s Mbit/s)\n' "$speed"
fi
