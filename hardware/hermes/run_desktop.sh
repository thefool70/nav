#!/usr/bin/env bash
set -euo pipefail

# 在随车笔记本执行：SSH 负责终端，Rerun 连接同一用户已登录的桌面。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"
# 使用绝对路径启动 Python 不会激活虚拟环境；Rerun 还需从 PATH 查找查看器。
export PATH="$project_root/.venv/bin:$PATH"

if ! desktop_environment="$(systemctl --user show-environment)"; then
    echo "无法读取当前用户的桌面环境；请先在随车笔记本登录同一用户的桌面。" >&2
    exit 1
fi
# 去掉 SSH X11 转发等会话遗留值；只导入桌面连接需要的变量。
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_SESSION_TYPE
while IFS= read -r assignment; do
    case "$assignment" in
        DISPLAY=*|WAYLAND_DISPLAY=*|XAUTHORITY=*|XDG_RUNTIME_DIR=*|DBUS_SESSION_BUS_ADDRESS=*|XDG_SESSION_TYPE=*)
            # systemctl 对值使用 Bash 可读取的转义；不执行用户输入或完整环境。
            eval "export $assignment"
            ;;
    esac
done <<< "$desktop_environment"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "没有可用的桌面显示环境；请先在随车笔记本登录同一用户的桌面。" >&2
    exit 1
fi
if [[ -n "${WAYLAND_DISPLAY:-}" && ! -S "${XDG_RUNTIME_DIR:-}/$WAYLAND_DISPLAY" ]]; then
    echo "桌面的 Wayland 连接已失效；请在随车笔记本重新登录桌面后再启动。" >&2
    exit 1
fi

echo "导航在当前终端运行；Rerun 窗口显示在随车笔记本桌面，Ctrl+C 停止导航。"
exec "$project_root/.venv/bin/python" -m robot_nav hermes "$@" --rerun --rerun-viewer native
