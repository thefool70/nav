"""统一运行配置：文件默认值 → 类型校验 → CLI 覆盖，不被算法模块直接读取。"""

import argparse
import json
import math
from pathlib import Path

# 这里只定义字段归属；运行默认值唯一保存在根目录 config.json。
FIELDS = {
    "navigation": "target search_mode max_cycles debug_random_score debug_frontier",
    "habitat": "scene seed gpu_device_id",
    "hermes": "base_url action_timeout_s action_stall_timeout_s max_unknown_path_m min_localization_quality startup_forward_m request_timeout_s action_poll_interval_s action_progress_interval_s action_stall_translation_m action_stall_rotation_deg action_arrival_position_m action_arrival_hold_s motion_frame_interval_s position_tolerance_m yaw_tolerance_deg",
    "camera": "camera_source camera_endpoint camera_topic camera_timeout_s camera_serial camera_calibration camera_height_m camera_forward_m camera_left_m camera_yaw_deg camera_pitch_down_deg camera_roll_deg",
    "perception": "vlm_endpoint vlm_model vlm_api_format vlm_timeout_s vlm_max_output_tokens object_class object_python object_device object_yolo_model object_sam_checkpoint object_timeout_s",
    "logging": "no_rerun rerun_save run_log",
    "calibration": "output turn_angle_deg drive_distance_m",
}
NULLABLE = set("target scene camera_serial camera_height_m camera_forward_m camera_left_m camera_yaw_deg camera_pitch_down_deg camera_roll_deg object_class object_python rerun_save run_log".split())
PATHS = set("scene camera_calibration object_yolo_model object_sam_checkpoint rerun_save run_log output".split())
BOOLS = {"debug_random_score", "debug_frontier", "no_rerun"}
INTS = {"seed", "gpu_device_id", "max_cycles", "min_localization_quality", "vlm_max_output_tokens"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"配置存在重复字段：{key}")
        result[key] = value
    return result


def load_config(path):
    """完整配置文件必须包含已声明的分组和字段，拼写错误与危险开关直接拒绝。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取配置 {path}：{exc}") from exc
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError("配置顶层必须包含且仅包含：" + ", ".join(FIELDS))
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


def _convert(name, value, action, directory):
    """复用 CLI 的范围及枚举校验，同时拒绝 JSON 中错误的原始类型。"""
    if value is None:
        if name not in NULLABLE:
            raise ValueError(f"{name} 不能为 null")
        return None
    if name in BOOLS:
        if type(value) is not bool:
            raise ValueError(f"{name} 必须为布尔值")
        return value
    if name in INTS:
        if type(value) is not int:
            raise ValueError(f"{name} 必须为整数")
    elif action.type is not None and action.type is not Path:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} 必须为有限数")
    elif not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空字符串或允许的 null")
    if action.type:
        value = action.type(value)
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"{name} 必须是 {tuple(action.choices)} 之一")
    if name in PATHS or (name == "object_python" and ("/" in value or "\\" in value)):
        path = Path(value).expanduser()
        value = str(path if path.is_absolute() else directory / path)
        if action.type is Path:
            value = Path(value)
    return value


def apply_config(parser, values, directory):
    """将合法文件值设为各子命令的默认值；显式命令行参数随后自然覆盖。"""
    subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    actions = {a.dest: a for sub in subcommands.choices.values() for a in sub._actions}
    converted = {}
    for name, value in values.items():
        try:
            converted[name] = _convert(name, value, actions[name], directory)
        except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
            raise ValueError(f"配置字段 {name} 无效：{exc}") from exc
    for sub in subcommands.choices.values():
        destinations = {a.dest for a in sub._actions}
        sub.set_defaults(**{k: v for k, v in converted.items() if k in destinations})
