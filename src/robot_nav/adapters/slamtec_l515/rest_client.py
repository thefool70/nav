"""Hermes Robot Agent REST 协议边界。"""

from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    OpenerDirector,
    ProxyHandler,
    Request,
    build_opener,
)

from ...core.models import Pose2D


class SlamtecActionError(RuntimeError):
    """Hermes 已接受 Action，但规划执行失败或未在时限内结束。"""


@dataclass(frozen=True)
class SlamtecExploreMap:
    """Robot Agent 栅格图原始数据；origin 是首格左下角而非格中心。"""

    origin_x_m: float
    origin_y_m: float
    width: int
    height: int
    resolution_m: float
    cells: bytes


@dataclass(frozen=True)
class SlamtecSlamState:
    """运动安全检查所需的 SLAM 工作状态。"""

    mapping_enabled: bool
    localization_enabled: bool
    localization_quality: int

    @property
    def mode(self) -> str:
        if self.mapping_enabled:
            return "mapping"
        if self.localization_enabled:
            return "localization"
        return "odometry"


@dataclass(frozen=True)
class SlamtecRobotHealth:
    """底盘健康摘要；warning 可继续，error 和 fatal 拒绝运动。"""

    has_warning: bool
    has_error: bool
    has_fatal: bool


class SlamtecRestClient:
    """通过 Robot Agent HTTP API 读取 SLAM 数据并管理运动 Action。"""

    def __init__(self, base_url: str, request_timeout_s: float) -> None:
        _validate_base_url(base_url)
        if not _is_positive_finite(request_timeout_s):
            raise ValueError("request_timeout_s 必须为正有限数")
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        # 底盘位于局域网，不能让系统的 HTTP_PROXY 接管 192.168.11.1。
        self._opener: OpenerDirector = build_opener(ProxyHandler({}))

    def get_robot_info(self) -> Mapping[str, Any]:
        """读取型号、设备 ID 和固件版本。"""
        return _require_mapping(
            self._request_json("GET", "/api/core/system/v1/robot/info"),
            "设备信息",
        )

    def get_pose(self) -> Pose2D:
        """读取地图坐标系下的二维位姿，单位为米和弧度。"""
        payload = _require_mapping(
            self._request_json(
                "GET", "/api/core/slam/v1/localization/pose"
            ),
            "机器人位姿",
        )
        values = tuple(
            _finite_number(payload, name) for name in ("x", "y", "yaw")
        )
        return Pose2D(x_m=values[0], y_m=values[1], yaw_rad=values[2])

    def get_localization_quality(self) -> int:
        """读取 0-100 的定位质量。"""
        value = self._request_json(
            "GET", "/api/core/slam/v1/localization/quality"
        )
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("Hermes 返回的定位质量不是整数")
        if not 0 <= value <= 100:
            raise RuntimeError(f"Hermes 返回了越界定位质量：{value}")
        return value

    def get_explore_map(self) -> SlamtecExploreMap:
        """读取并解析当前激光探索栅格图。"""
        payload = self._request_bytes(
            "GET", "/api/core/slam/v1/maps/explore"
        )
        return parse_explore_map(payload)

    def get_slam_state(self) -> SlamtecSlamState:
        """读取建图/定位开关和质量，明确区分两种工作模式。"""
        mapping_enabled = self._request_boolean(
            "/api/core/slam/v1/mapping/:enable",
            "建图开关",
        )
        localization_enabled = self._request_boolean(
            "/api/core/slam/v1/localization/:enable",
            "定位开关",
        )
        return SlamtecSlamState(
            mapping_enabled=mapping_enabled,
            localization_enabled=localization_enabled,
            localization_quality=self.get_localization_quality(),
        )

    def get_robot_health(self) -> SlamtecRobotHealth:
        """读取会影响运动安全的 warning/error/fatal 摘要。"""
        payload = _require_mapping(
            self._request_json(
                "GET", "/api/core/system/v1/robot/health"
            ),
            "机器人健康状态",
        )
        return SlamtecRobotHealth(
            has_warning=_boolean_field(payload, "hasWarning"),
            has_error=_boolean_field(payload, "hasError"),
            has_fatal=_boolean_field(payload, "hasFatal"),
        )

    def get_action_names(self) -> Tuple[str, ...]:
        """读取本机固件实际支持的运动 Action 名称。"""
        payload = self._request_json(
            "GET", "/api/core/motion/v1/action-factories"
        )
        if not isinstance(payload, list):
            raise RuntimeError("Hermes Action 列表不是数组")
        names = []
        for item in payload:
            mapping = _require_mapping(item, "Action 条目")
            name = mapping.get("action_name")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("Hermes Action 条目缺少 action_name")
            names.append(name.strip())
        return tuple(names)

    def create_action(self, action_name: str, options: Mapping[str, Any]) -> int:
        """创建一个运动 Action，并返回固件分配的 action_id。"""
        payload = _require_mapping(
            self._request_json(
                "POST",
                "/api/core/motion/v1/actions",
                {"action_name": action_name, "options": dict(options)},
            ),
            "Action 创建结果",
        )
        action_id = payload.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise RuntimeError("Hermes Action 创建结果缺少整数 action_id")
        return action_id

    def get_remaining_path(self) -> Tuple[Tuple[float, float], ...]:
        """读取当前 Action 的剩余地图系路径点。"""
        payload = _require_mapping(
            self._request_json("GET", "/api/core/motion/v1/path"),
            "剩余路径",
        )
        raw_points = payload.get("path_points")
        if not isinstance(raw_points, list):
            raise RuntimeError("Hermes 剩余路径缺少 path_points 数组")

        points = []
        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise RuntimeError(
                    f"Hermes 剩余路径点 {index} 不是 [x, y]"
                )
            x_m, y_m = raw_point
            if not _is_finite_number(x_m) or not _is_finite_number(y_m):
                raise RuntimeError(
                    f"Hermes 剩余路径点 {index} 包含非有限坐标"
                )
            points.append((float(x_m), float(y_m)))
        return tuple(points)

    def wait_for_action(
        self,
        action_id: int,
        timeout_s: float,
        poll_interval_s: float,
        on_poll: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """轮询 Action；把执行状态和阶段交给调用方监控。"""
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id 必须为整数")
        if not _is_positive_finite(timeout_s):
            raise ValueError("timeout_s 必须为正有限数")
        if not _is_positive_finite(poll_interval_s):
            raise ValueError("poll_interval_s 必须为正有限数")

        deadline = time.monotonic() + float(timeout_s)
        while True:
            payload = _require_mapping(
                self._request_json(
                    "GET", f"/api/core/motion/v1/actions/{action_id}"
                ),
                "Action 状态",
            )
            state = _require_mapping(payload.get("state"), "Action state")
            status = state.get("status")
            if isinstance(status, bool) or not isinstance(status, int):
                raise RuntimeError("Hermes Action state 缺少整数 status")
            stage = payload.get("stage")
            if not isinstance(stage, str):
                stage = ""
            if status == 4:
                result = state.get("result")
                if result == 0:
                    return
                reason = state.get("reason")
                detail = reason if isinstance(reason, str) and reason else result
                raise SlamtecActionError(
                    f"Hermes Action {action_id} 执行失败：{detail}"
                )

            now = time.monotonic()
            if now >= deadline:
                try:
                    self.abort_current_action()
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Hermes Action {action_id} 已超时，且终止请求失败：{exc}"
                    ) from exc
                raise SlamtecActionError(
                    f"Hermes Action {action_id} 超过 {timeout_s:.1f} 秒未结束"
                )
            if on_poll is not None:
                on_poll(status, stage)
            time.sleep(min(float(poll_interval_s), deadline - now))

    def abort_current_action(self) -> None:
        """终止当前 Action；用于超时、中断和采集失败后的安全收尾。"""
        self._request_bytes(
            "DELETE", "/api/core/motion/v1/actions/:current"
        )

    def _request_boolean(self, path: str, description: str) -> bool:
        value = self._request_json("GET", path)
        if not isinstance(value, bool):
            raise RuntimeError(f"Hermes 返回的{description}不是布尔值")
        return value

    def _request_json(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        raw = self._request_bytes(method, path, body)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Hermes {path} 返回的不是有效 JSON") from exc

    def _request_bytes(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
    ) -> bytes:
        data = None
        headers = {"Accept": "application/json, application/octet-stream"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=self.request_timeout_s
            ) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f"：{detail}" if detail else ""
            raise RuntimeError(
                f"Hermes {method} {path} 返回 HTTP {exc.code}{suffix}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"无法访问 Hermes {self.base_url}{path}：{exc}"
            ) from exc


def parse_explore_map(payload: bytes) -> SlamtecExploreMap:
    """按官方 36 字节小端头解析 maps/explore 响应。"""
    if len(payload) < 36:
        raise RuntimeError("Hermes 栅格图短于 36 字节协议头")
    origin_x, origin_y, width, height, resolution = struct.unpack_from(
        "<ffIIf", payload, 0
    )
    data_length = struct.unpack_from("<I", payload, 32)[0]
    expected_length = width * height
    if width == 0 or height == 0:
        raise RuntimeError("Hermes 返回了空栅格图")
    if data_length != expected_length or len(payload) != 36 + data_length:
        raise RuntimeError(
            "Hermes 栅格图尺寸不一致："
            f"{width}×{height}, 声明 {data_length} 字节, "
            f"实际 {len(payload) - 36} 字节"
        )
    if not all(
        math.isfinite(value) for value in (origin_x, origin_y, resolution)
    ):
        raise RuntimeError("Hermes 栅格图包含非有限元数据")
    if resolution <= 0.0:
        raise RuntimeError("Hermes 栅格图分辨率必须为正数")
    return SlamtecExploreMap(
        origin_x_m=float(origin_x),
        origin_y_m=float(origin_y),
        width=width,
        height=height,
        resolution_m=float(resolution),
        cells=payload[36:],
    )


def resolve_action_name(names: Sequence[str], short_name: str) -> str:
    """按末段匹配 Action，兼容 agent.actions 与 slamtec.agent.actions 前缀。"""
    matches = tuple(
        name for name in names if name.rsplit(".", 1)[-1] == short_name
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Hermes 应提供且仅提供一个 {short_name}，实际匹配 {matches}"
        )
    return matches[0]


def _validate_base_url(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url 必须为非空 URL")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url 必须是 http 或 https URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url 不能包含 query 或 fragment")


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Hermes 返回的{description}不是对象")
    return value


def _boolean_field(payload: Mapping[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise RuntimeError(f"Hermes 健康字段 {name} 不是布尔值")
    return value


def _finite_number(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"Hermes 位姿字段 {name} 不是有限数")
    return float(value)


def _is_positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0.0
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


__all__ = [
    "SlamtecActionError",
    "SlamtecExploreMap",
    "SlamtecRestClient",
    "SlamtecRobotHealth",
    "SlamtecSlamState",
    "parse_explore_map",
    "resolve_action_name",
]
