"""统一运行配置：文件默认值 → 类型校验 → CLI 覆盖，不被算法模块直接读取。"""

import argparse
import json
from pathlib import Path

# 这里只定义字段归属；运行默认值唯一保存在根目录 config.json。
FIELDS = {
    "navigation": "target search_mode max_cycles debug_random_score debug_frontier max_unknown_path_m",
    "habitat": "scene seed gpu_device_id",
    "hermes": (
        "base_url action_timeout_s action_stall_timeout_s min_localization_quality "
        "startup_forward_m request_timeout_s action_poll_interval_s "
        "action_progress_interval_s action_stall_translation_m action_stall_rotation_deg "
        "action_arrival_position_m action_arrival_hold_s motion_frame_interval_s "
        "position_tolerance_m yaw_tolerance_deg"
    ),
    "camera": (
        "camera_source camera_endpoint camera_topic camera_timeout_s camera_serial "
        "camera_calibration camera_height_m camera_forward_m camera_left_m camera_yaw_deg "
        "camera_pitch_down_deg camera_roll_deg"
    ),
    "perception": (
        "vlm_endpoint vlm_model vlm_api_format vlm_timeout_s vlm_max_output_tokens "
        "object_class object_python object_device object_yolo_model object_sam_checkpoint "
        "object_timeout_s"
    ),
    "logging": "no_rerun rerun_save run_log",
    "calibration": "output turn_angle_deg drive_distance_m",
}
NULLABLE = {
    "target", "scene", "camera_serial", "camera_height_m", "camera_forward_m",
    "camera_left_m", "camera_yaw_deg", "camera_pitch_down_deg", "camera_roll_deg",
    "object_class", "object_python", "rerun_save", "run_log",
}
PATHS = {
    "scene", "camera_calibration", "object_yolo_model", "object_sam_checkpoint",
    "rerun_save", "run_log", "output",
}
BOOLS = {"debug_random_score", "debug_frontier", "no_rerun"}
INTS = {"seed", "gpu_device_id", "max_cycles", "min_localization_quality", "vlm_max_output_tokens"}


def load_config(path):
    """完整配置文件必须包含已声明的分组和字段，拼写错误与危险开关直接拒绝。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError("配置顶层必须包含且仅包含：" + ", ".join(FIELDS))
    # JSON 分组用于人工阅读；内部按参数名展开，供 argparse 的默认值统一使用。
    flattened = {}
    for group, fields in FIELDS.items():
        values = data[group]
        expected = set(fields.split())
        if not isinstance(values, dict):
            raise ValueError(f"配置 {group} 必须是对象")
        unknown, missing = set(values) - expected, expected - set(values)
        if unknown or missing:
            raise ValueError(f"配置 {group} 字段错误：未知 {sorted(unknown)}，缺少 {sorted(missing)}")
        flattened.update(values)
    return flattened


def apply_config(subparsers, values, directory):
    """将文件值设为子命令默认值；后续 parse_args 自然应用显式 CLI 覆盖。"""
    actions = {}
    for parser in subparsers.values():
        for action in parser._actions:
            actions[action.dest] = action

    converted = {}
    for name, value in values.items():
        try:
            converted[name] = _convert(name, value, actions[name], directory)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise ValueError(f"配置字段 {name} 无效：{exc}") from exc

    # 只给子命令实际注册的参数赋默认值，随后显式命令行参数自然覆盖它们。
    for parser in subparsers.values():
        defaults = {action.dest: converted[action.dest]
                    for action in parser._actions if action.dest in converted}
        parser.set_defaults(**defaults)


def _unique_object(pairs):
    """在 JSON 对象转成字典之前检查重复键，防止同名配置被后值静默覆盖。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"配置存在重复字段：{key}")
        result[key] = value
    return result


def _validate_type(name, value, action):
    """JSON 原始类型在转换前检查，避免把 true 当作数字或把小数截成整数。"""
    if value is None:
        if name not in NULLABLE:
            raise ValueError("不能为 null")
    elif name in BOOLS:
        if type(value) is not bool:
            raise ValueError("必须为布尔值")
    elif name in INTS:
        if type(value) is not int:
            raise ValueError("必须为整数")
    elif action.type is not None and action.type is not Path:
        if type(value) not in (int, float):
            raise ValueError("必须为数字")
    elif not isinstance(value, str) or not value.strip():
        raise ValueError("必须为非空字符串")


def _resolve_path(name, value, directory):
    """文件内的相对路径以配置目录为基准；解释器名称如 python3 保持原样。"""
    interpreter_path = name == "object_python" and ("/" in value or "\\" in value)
    if name not in PATHS and not interpreter_path:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else directory / path)


def _convert(name, value, action, directory):
    """检查 JSON 类型、解析路径，再复用 CLI 的数值范围和枚举约束。"""
    _validate_type(name, value, action)
    if value is None:
        return None
    if isinstance(value, str):
        value = _resolve_path(name, value, directory)
    if action.type is not None:
        value = action.type(value)
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"必须是 {tuple(action.choices)} 之一")
    return value
