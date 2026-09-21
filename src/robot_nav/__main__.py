"""robot-nav 启动入口：读取参数并分派导航或标定。

本模块只做三件事：解析参数、校验参数组合、把控制权交给 :mod:`~robot_nav.launch`。
参数定义在 ``cli.py``，运行装配在 ``launch.py``，导航循环在 ``app.py``，外参标定在 ``calibration_launch.py``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .calibration_launch import run_calibration_entries
from .core.models import SearchMode
from .cli import parse_arguments
from .launch import MISSING_VLM_CREDENTIAL, run_entries


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
    api_key = _resolve_vlm_api_key()
    if (
        getattr(args, "search_mode", SearchMode.OBJECT.value)
        == SearchMode.SCENE.value
        and getattr(args, "debug_random_score", False)
    ):
        parser.error("场景搜索需要 VLM，不能与 --debug-random-score 同时使用")

    if args.adapter in ("habitat", "hermes"):
        if not getattr(args, "preflight_only", False) and not args.debug_random_score and not api_key:
            parser.error(MISSING_VLM_CREDENTIAL)
        return run_entries(args, api_key)

    if args.adapter == "calibrate-hermes":
        if not args.enable_motion:
            parser.error("外参标定会移动真机，必须显式提供 --enable-motion")
        return run_calibration_entries(args)

    parser.error(f"未知 Adapter：{args.adapter}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
