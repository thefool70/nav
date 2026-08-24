"""S100 反馈里程计与差速运动执行。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ...core.models import Pose2D
from .s100_serial import S100SerialConnection, S100Status


_PREFLIGHT_COMMANDS = 10
_PREFLIGHT_MIN_FRAMES = 3
_STATIONARY_X_MPS = 0.02
_STATIONARY_Z_RADPS = 0.03
_STOP_STABLE_FRAMES = 3
_ERROR_STOP_COMMANDS = 10


@dataclass(frozen=True)
class S100MotionConfig:
    """低速差速控制参数，距离单位为米，角度单位为弧度。"""

    drive_speed_mps: float = 0.10
    turn_rate_radps: float = 0.25
    command_period_s: float = 0.05
    feedback_timeout_s: float = 0.60
    position_tolerance_m: float = 0.03
    yaw_tolerance_rad: float = 0.03


class S100MotionController:
    """持续发送速度命令，根据反馈积分位姿，并同步等待动作完成。"""

    def __init__(
        self,
        connection: S100SerialConnection,
        config: S100MotionConfig,
    ) -> None:
        if not isinstance(connection, S100SerialConnection):
            raise ValueError("connection 必须为 S100SerialConnection")
        _validate_config(config)
        self._connection = connection
        self.config = config
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._x_mps = 0.0
        self._z_radps = 0.0
        self._last_integrated_at = time.monotonic()
        self._last_feedback_at = 0.0
        self._latest_status: Optional[S100Status] = None

    @property
    def pose(self) -> Pose2D:
        """返回最近一次反馈积分得到的 S100 里程计位姿。"""
        self._integrate_to(time.monotonic())
        return self._pose

    def preflight(self) -> None:
        """重复发送停止并确认底盘已使能、反馈充足且处于静止状态。"""
        statuses = []
        for _ in range(_PREFLIGHT_COMMANDS):
            cycle_started = time.monotonic()
            self._connection.send_stop()
            status = self._connection.wait_for_status(
                self.config.command_period_s
            )
            if status is not None:
                statuses.append(status)
                self._accept_status(status, time.monotonic())
            self._wait_until_cycle_end(cycle_started)

        if len(statuses) < _PREFLIGHT_MIN_FRAMES:
            raise RuntimeError(
                "S100 预检失败：有效反馈不足 "
                f"({len(statuses)}/{_PREFLIGHT_MIN_FRAMES})"
            )
        latest = statuses[-1]
        if latest.stop_flag:
            raise RuntimeError("S100 预检失败：底盘未使能")
        if not _is_stationary(latest):
            raise RuntimeError(
                "S100 预检失败：底盘未静止 "
                f"(X={latest.x_mps:.3f} m/s, Z={latest.z_radps:.3f} rad/s)"
            )

        now = time.monotonic()
        self._x_mps = 0.0
        self._z_radps = 0.0
        self._last_integrated_at = now
        self._last_feedback_at = now

    def read_pose(self) -> Pose2D:
        """等待一个控制周期的反馈，然后返回当前里程计位姿。"""
        status = self._connection.wait_for_status(
            self.config.command_period_s
        )
        now = time.monotonic()
        if status is not None:
            self._accept_status(status, now)
        else:
            self._integrate_to(now)
        self._require_healthy_feedback(now)
        return self._pose

    def turn_to_world_yaw(self, target_yaw_rad: float) -> None:
        """原地旋转到指定世界系 yaw，完成并确认静止后返回。"""
        target = _wrap_angle(_finite(target_yaw_rad, "target_yaw_rad"))
        initial_error = abs(_angle_difference(target, self.read_pose().yaw_rad))
        if initial_error <= self.config.yaw_tolerance_rad:
            return

        timeout_s = initial_error / self.config.turn_rate_radps * 2.0 + 2.0

        def is_complete(pose: Pose2D) -> bool:
            return (
                abs(_angle_difference(target, pose.yaw_rad))
                <= self.config.yaw_tolerance_rad
            )

        def command_for(pose: Pose2D) -> tuple[float, float]:
            error = _angle_difference(target, pose.yaw_rad)
            turn_rate = math.copysign(self.config.turn_rate_radps, error)
            return 0.0, turn_rate

        self._execute_control(is_complete, command_for, timeout_s, "转向")

    def drive_to_world_xy(self, target_world_xy: tuple[float, float]) -> None:
        """先朝向目标，再低速直行到世界系二维目标点。"""
        target_x = _finite(target_world_xy[0], "target_world_xy[0]")
        target_y = _finite(target_world_xy[1], "target_world_xy[1]")
        start = self.read_pose()
        distance = math.hypot(target_x - start.x_m, target_y - start.y_m)
        if distance <= self.config.position_tolerance_m:
            return

        heading = math.atan2(target_y - start.y_m, target_x - start.x_m)
        self.turn_to_world_yaw(heading)
        timeout_s = distance / self.config.drive_speed_mps * 3.0 + 2.0

        def is_complete(pose: Pose2D) -> bool:
            return (
                math.hypot(target_x - pose.x_m, target_y - pose.y_m)
                <= self.config.position_tolerance_m
            )

        def command_for(pose: Pose2D) -> tuple[float, float]:
            desired = math.atan2(target_y - pose.y_m, target_x - pose.x_m)
            error = _angle_difference(desired, pose.yaw_rad)
            if abs(error) > 0.35:
                return 0.0, math.copysign(self.config.turn_rate_radps, error)
            correction = max(
                -self.config.turn_rate_radps,
                min(self.config.turn_rate_radps, 1.5 * error),
            )
            return self.config.drive_speed_mps, correction

        self._execute_control(is_complete, command_for, timeout_s, "直行")

    def stop(self) -> None:
        """发送停止并等待连续静止反馈；失败时抛出明确异常。"""
        deadline = time.monotonic() + max(
            1.0, self.config.feedback_timeout_s * 2.0
        )
        stable_frames = 0
        while time.monotonic() < deadline:
            cycle_started = time.monotonic()
            self._connection.send_stop()
            status = self._connection.wait_for_status(
                self.config.command_period_s
            )
            now = time.monotonic()
            if status is not None:
                self._accept_status(status, now)
                if status.stop_flag:
                    raise RuntimeError("S100 停止时底盘变为未使能状态")
                stable_frames = (
                    stable_frames + 1 if _is_stationary(status) else 0
                )
                if stable_frames >= _STOP_STABLE_FRAMES:
                    self._x_mps = 0.0
                    self._z_radps = 0.0
                    return
            self._wait_until_cycle_end(cycle_started)
        raise RuntimeError("S100 停止后未收到连续静止反馈")

    def _execute_control(
        self,
        is_complete: Callable[[Pose2D], bool],
        command_for: Callable[[Pose2D], tuple[float, float]],
        timeout_s: float,
        action_name: str,
    ) -> None:
        """以固定周期闭环发送速度，任何退出路径都会先尝试停止底盘。"""
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                cycle_started = time.monotonic()
                self._ingest_available_statuses()
                now = time.monotonic()
                self._integrate_to(now)
                self._require_healthy_feedback(now)
                if is_complete(self._pose):
                    break
                if now >= deadline:
                    raise RuntimeError(f"S100 {action_name}超时")

                x_mps, z_radps = command_for(self._pose)
                self._connection.send_velocity(x_mps, z_radps)
                status = self._connection.wait_for_status(
                    self.config.command_period_s
                )
                if status is not None:
                    self._accept_status(status, time.monotonic())
                self._wait_until_cycle_end(cycle_started)
        except Exception:
            self._stop_best_effort()
            raise
        self.stop()

    def _ingest_available_statuses(self) -> None:
        statuses = self._connection.read_available_statuses()
        if statuses:
            self._accept_status(statuses[-1], time.monotonic())

    def _accept_status(self, status: S100Status, received_at: float) -> None:
        self._integrate_to(received_at)
        self._latest_status = status
        self._x_mps = status.x_mps
        self._z_radps = status.z_radps
        self._last_feedback_at = received_at

    def _integrate_to(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_integrated_at)
        if elapsed <= 0.0:
            return
        yaw_change = self._z_radps * elapsed
        middle_yaw = self._pose.yaw_rad + yaw_change * 0.5
        distance = self._x_mps * elapsed
        self._pose = Pose2D(
            x_m=self._pose.x_m + distance * math.cos(middle_yaw),
            y_m=self._pose.y_m + distance * math.sin(middle_yaw),
            yaw_rad=_wrap_angle(self._pose.yaw_rad + yaw_change),
        )
        self._last_integrated_at = now

    def _require_healthy_feedback(self, now: float) -> None:
        if (
            self._latest_status is None
            or now - self._last_feedback_at > self.config.feedback_timeout_s
        ):
            raise RuntimeError("S100 状态反馈超时")
        if self._latest_status.stop_flag:
            raise RuntimeError("S100 状态反馈显示底盘未使能")

    def _wait_until_cycle_end(self, cycle_started: float) -> None:
        remaining = self.config.command_period_s - (
            time.monotonic() - cycle_started
        )
        if remaining > 0.0:
            time.sleep(remaining)

    def _stop_best_effort(self) -> None:
        for _ in range(_ERROR_STOP_COMMANDS):
            try:
                self._connection.send_stop()
            except Exception:
                return
            time.sleep(self.config.command_period_s)


def _is_stationary(status: S100Status) -> bool:
    return (
        abs(status.x_mps) <= _STATIONARY_X_MPS
        and abs(status.z_radps) <= _STATIONARY_Z_RADPS
    )


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为有限数")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须为有限数") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须为有限数")
    return converted


def _validate_config(config: S100MotionConfig) -> None:
    if not isinstance(config, S100MotionConfig):
        raise ValueError("config 必须为 S100MotionConfig")
    positive_names = (
        "drive_speed_mps",
        "turn_rate_radps",
        "command_period_s",
        "feedback_timeout_s",
        "position_tolerance_m",
        "yaw_tolerance_rad",
    )
    values = {
        name: _finite(getattr(config, name), name) for name in positive_names
    }
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("S100 运动参数必须为正有限数")
    if values["command_period_s"] >= values["feedback_timeout_s"]:
        raise ValueError("command_period_s 必须小于 feedback_timeout_s")


__all__ = ["S100MotionConfig", "S100MotionController"]
