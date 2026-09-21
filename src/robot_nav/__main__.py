"""robot-nav 启动入口：读取参数并分派导航或标定。

本模块只做三件事：读取参数、按需读取凭据、把控制权交给 :mod:`~robot_nav.launch`。
参数定义在 ``cli.py``，运行装配在 ``launch.py``，导航循环在 ``app.py``，外参标定在 ``calibration_launch.py``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .calibration_launch import run_calibration_entries
from .cli import parse_arguments
from .launch import run_entries


def _resolve_vlm_api_key() -> str:
    """优先读取项目环境变量，否则复用 OpenCode Go 本地凭据。"""
    environment_key = os.environ.get("ROBOT_NAV_VLM_API_KEY", "").strip()
    if environment_key:
        return environment_key

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    data_home = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    auth_path = data_home / "opencode" / "auth.json"
    try:
        auth_payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""

    if not isinstance(auth_payload, dict):
        return ""
    credential = auth_payload.get("opencode-go")
    if not isinstance(credential, dict) or credential.get("type") != "api":
        return ""
    api_key = credential.get("key")
    return api_key.strip() if isinstance(api_key, str) else ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析运行环境并启动所选入口。"""
    parser, args = parse_arguments(argv)
    if args.adapter == "calibrate-hermes":
        return run_calibration_entries(args)

    api_key = ""
    if not args.preflight_only and not args.debug_random_score:
        api_key = _resolve_vlm_api_key()
        if not api_key:
            parser.error(
                "缺少视觉模型凭据：设置 ROBOT_NAV_VLM_API_KEY，"
                "或先用 opencode auth login 登录 OpenCode Go"
            )
    return run_entries(args, api_key)


if __name__ == "__main__":
    raise SystemExit(main())
