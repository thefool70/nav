"""把导航决策帧和 Adapter 运动帧记录到 Rerun。"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from ..core.models import (
    DepthImage,
    Grid,
    NavigationFrame,
    NavigationResult,
    Pose2D,
    RelativePoseCommand,
    RgbImage,
    SearchDirectionState,
    TargetObservation,
)


UNKNOWN_RGB = (90, 90, 90)
FREE_RGB = (235, 235, 235)
OCCUPIED_RGB = (25, 25, 25)
ROBOT_RGB = (0, 170, 255)
HEADING_RGB = (0, 220, 140)
TRAJECTORY_RGB = (0, 120, 255)
COMMAND_RGB = (255, 140, 0)
BBOX_RGB = (0, 255, 80)

DIRECTION_COLORS = {
    SearchDirectionState.PENDING: (255, 210, 0),
    SearchDirectionState.COMMITTED: (0, 170, 255),
    SearchDirectionState.EXPLORED: (130, 130, 130),
    SearchDirectionState.INVALIDATED: (255, 70, 70),
}

HEADING_LENGTH_M = 0.6
OCCUPANCY_THRESHOLD = 0.5

STATUS_IMAGE_WIDTH = 560
STATUS_FONT_SIZE = 16
STATUS_LINE_SPACING_PX = 6
STATUS_PADDING_PX = 12
STATUS_BG_RGB = (24, 24, 24)
STATUS_TEXT_RGB = (235, 235, 235)

STATUS_FONT_CANDIDATES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)

_FONT_NOTICE_PRINTED = False


class RerunVisualizer:
    """显示传感器、地图、轨迹和算法状态的实时调试界面。"""

    def __init__(self, target_text: str) -> None:
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError(
                "未安装 rerun-sdk，请通过 pip install -e '.[visualization]' 安装"
            ) from exc

        self._rr = rr
        self._target_text = target_text
        self._sample_index = 0
        self._trajectory_xy: List[Tuple[float, float]] = []
        self._status_font = _load_status_font()
        if self._status_font is None:
            _print_font_notice_once()
        rr.init("robot-nav")
        rr.serve_web(open_browser=True, web_port=9090, ws_port=9877)
        print(
            "Rerun Web Viewer 地址："
            "http://127.0.0.1:9090/?url=ws://127.0.0.1:9877"
        )

    def log_cycle(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
        result: NavigationResult,
    ) -> None:
        """记录算法决策帧及其观测、命令和状态。"""
        self._begin_sample()
        self._log_rgb(frame, observation)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)
        self._log_command(frame, result)
        self._log_candidates(result)
        self._log_status(frame, observation, result)

    def log_motion_frame(self, frame: NavigationFrame) -> None:
        """记录 Adapter 执行动作后的传感器帧，不推进算法状态。"""
        self._begin_sample()
        self._log_rgb(frame, None)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)

    def _begin_sample(self) -> None:
        """让决策帧与运动帧共享同一条递增时间轴。"""
        self._sample_index += 1
        self._rr.set_time_sequence("frame", self._sample_index)

    def _log_rgb(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
    ) -> None:
        """记录 RGB，并把归一化目标框换算为像素框。"""
        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            self._rr.log("camera/rgb", self._rr.Clear(recursive=True))
            return

        image = _rgb_to_numpy(frame.rgb)
        self._rr.log("camera/rgb", self._rr.Image(image))
        if observation is None or observation.bbox_norm is None:
            self._clear("camera/rgb/target")
            return

        box_min, box_size = _bbox_norm_to_pixel_box(
            observation.bbox_norm,
            height=image.shape[0],
            width=image.shape[1],
        )
        self._rr.log(
            "camera/rgb/target",
            self._rr.Boxes2D(
                mins=[box_min],
                sizes=[box_size],
                colors=[BBOX_RGB],
                labels=["target"],
            ),
        )

    def _log_depth(self, frame: NavigationFrame) -> None:
        """记录米制深度图；None 像素转换为 NaN。"""
        if (
            frame.depth is None
            or len(frame.depth) == 0
            or len(frame.depth[0]) == 0
        ):
            self._clear("camera/depth")
            return
        self._rr.log(
            "camera/depth",
            self._rr.DepthImage(_depth_to_numpy(frame.depth), meter=1.0),
        )

    def _log_occupancy_map(self, frame: NavigationFrame) -> None:
        """用灰/白/黑显示未知、自由和障碍；图像上方对应世界 +y。"""
        occupancy = frame.obstacle_map.occupancy
        if len(occupancy) == 0 or len(occupancy[0]) == 0:
            self._clear("map/occupancy")
            return
        image = _occupancy_to_rgb_numpy(occupancy)
        self._rr.log("map/occupancy", self._rr.Image(np.flipud(image).copy()))

    def _log_robot_pose(self, frame: NavigationFrame) -> None:
        """记录世界系机器人位置、朝向和累计轨迹。"""
        position = (frame.pose.x_m, frame.pose.y_m)
        self._rr.log(
            "world/robot",
            self._rr.Points2D([position], colors=[ROBOT_RGB], radii=0.16),
        )
        self._rr.log(
            "world/robot/heading",
            self._rr.Arrows2D(
                origins=[position],
                vectors=[_yaw_vector(frame.pose.yaw_rad, HEADING_LENGTH_M)],
                colors=[HEADING_RGB],
                radii=0.035,
            ),
        )

        if not self._trajectory_xy or position != self._trajectory_xy[-1]:
            self._trajectory_xy.append(position)
        if len(self._trajectory_xy) >= 2:
            self._rr.log(
                "world/trajectory",
                self._rr.LineStrips2D(
                    [self._trajectory_xy],
                    colors=[TRAJECTORY_RGB],
                    radii=0.025,
                ),
            )

    def _log_command(self, frame: NavigationFrame, result: NavigationResult) -> None:
        """把相对位姿命令转换为世界系平移和目标朝向箭头。"""
        command = result.command
        if command is None:
            self._rr.log("world/command", self._rr.Clear(recursive=True))
            return

        origin = (frame.pose.x_m, frame.pose.y_m)
        self._rr.log(
            "world/command/translation",
            self._rr.Arrows2D(
                origins=[origin],
                vectors=[_command_world_vector(command, frame.pose)],
                colors=[COMMAND_RGB],
                radii=0.04,
                labels=["translation"],
            ),
        )
        self._rr.log(
            "world/command/heading",
            self._rr.Arrows2D(
                origins=[origin],
                vectors=[
                    _yaw_vector(
                        frame.pose.yaw_rad + command.yaw_rad,
                        HEADING_LENGTH_M,
                    )
                ],
                colors=[COMMAND_RGB],
                radii=0.025,
                labels=["target heading"],
            ),
        )

    def _log_candidates(self, result: NavigationResult) -> None:
        """按方向状态着色显示历史观测节点中的候选点。"""
        positions = []
        colors = []
        labels = []
        for node in result.state.observation_history:
            for direction in node.directions:
                if direction.candidate_world_xy is None:
                    continue
                positions.append(direction.candidate_world_xy)
                colors.append(DIRECTION_COLORS[direction.state])
                labels.append(direction.direction_id)

        if not positions:
            self._clear("world/history_candidates")
            return
        self._rr.log(
            "world/history_candidates",
            self._rr.Points2D(
                positions,
                colors=colors,
                radii=0.11,
                labels=labels,
                show_labels=False,
            ),
        )

    def _log_status(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
        result: NavigationResult,
    ) -> None:
        """记录状态面板；有 CJK 字体时渲染为图像，否则回退为 ASCII 文本。"""
        lines = _status_lines(self._target_text, frame, observation, result)
        if self._status_font is None:
            self._rr.log(
                "navigation/status",
                self._rr.TextDocument(
                    "\n".join(f"- {_ascii_only(line)}" for line in lines),
                    media_type="text/markdown",
                ),
            )
            return
        image = _render_status_image(self._status_font, lines)
        self._rr.log("navigation/status", self._rr.Image(image))

    def _clear(self, path: str) -> None:
        self._rr.log(path, self._rr.Clear(recursive=False))


def _rgb_to_numpy(rgb: RgbImage) -> np.ndarray:
    return np.asarray(rgb, dtype=np.uint8)


def _depth_to_numpy(depth: DepthImage) -> np.ndarray:
    return np.asarray(
        [[np.nan if value is None else value for value in row] for row in depth],
        dtype=np.float32,
    )


def _occupancy_to_rgb_numpy(occupancy: Grid) -> np.ndarray:
    """把占用栅格转换为未知/自由/障碍三色图像。"""
    image = np.empty((len(occupancy), len(occupancy[0]), 3), dtype=np.uint8)
    for row_index, row in enumerate(occupancy):
        for col_index, value in enumerate(row):
            if value is None:
                color = UNKNOWN_RGB
            elif float(value) > OCCUPANCY_THRESHOLD:
                color = OCCUPIED_RGB
            else:
                color = FREE_RGB
            image[row_index, col_index] = color
    return image


def _bbox_norm_to_pixel_box(
    bbox_norm: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min, y_min, x_max, y_max = bbox_norm
    return (
        (x_min * width, y_min * height),
        ((x_max - x_min) * width, (y_max - y_min) * height),
    )


def _yaw_vector(yaw_rad: float, length_m: float) -> Tuple[float, float]:
    return (math.cos(yaw_rad) * length_m, math.sin(yaw_rad) * length_m)


def _command_world_vector(
    command: RelativePoseCommand,
    pose: Pose2D,
) -> Tuple[float, float]:
    """把机器人系前/左平移转换为世界系向量。"""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        command.forward_m * cosine - command.left_m * sine,
        command.forward_m * sine + command.left_m * cosine,
    )


def _load_status_font() -> Optional[object]:
    """加载状态面板 CJK 字体；Pillow 缺失或所有候选加载失败时返回 None。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in STATUS_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size=STATUS_FONT_SIZE)
        except OSError:
            continue
    return None


def _print_font_notice_once() -> None:
    """Pillow/CJK 字体缺失时最多打印一次提示，避免每个周期刷屏。"""
    global _FONT_NOTICE_PRINTED
    if _FONT_NOTICE_PRINTED:
        return
    _FONT_NOTICE_PRINTED = True
    print("提示：Pillow 或 CJK 字体不可用，状态面板回退为 ASCII 文本")


def _wrap_text(font: object, text: str, max_pixels: float) -> List[str]:
    """按像素宽度把文本折成多行，不打断单个字符。"""
    lines: List[str] = []
    current = ""
    for char in text:
        if font.getlength(current + char) <= max_pixels:
            current += char
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def _render_status_image(font: object, lines: List[str]) -> np.ndarray:
    """把状态行渲染为深色背景 RGB 图像，供 rr.Image 记录。"""
    from PIL import Image, ImageDraw

    max_pixels = STATUS_IMAGE_WIDTH - 2 * STATUS_PADDING_PX
    wrapped: List[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(font, line, max_pixels))

    line_height = STATUS_FONT_SIZE + STATUS_LINE_SPACING_PX
    height = 2 * STATUS_PADDING_PX + len(wrapped) * line_height
    image = Image.new("RGB", (STATUS_IMAGE_WIDTH, height), STATUS_BG_RGB)
    draw = ImageDraw.Draw(image)
    y = STATUS_PADDING_PX
    for line in wrapped:
        draw.text((STATUS_PADDING_PX, y), line, font=font, fill=STATUS_TEXT_RGB)
        y += line_height
    return np.asarray(image)


def _ascii_only(text: str) -> str:
    """把非 ASCII 字符替换为 '?'，供无字体时生成纯 ASCII 回退文本。"""
    return "".join(char if ord(char) < 128 else "?" for char in text)


def _status_lines(
    target_text: str,
    frame: NavigationFrame,
    observation: Optional[TargetObservation],
    result: NavigationResult,
) -> List[str]:
    """构造状态面板文本行，覆盖目标、状态、阶段、步骤、位姿与观测细节。"""
    lines = [
        f"target: {target_text}",
        f"status: {result.status.value}",
        f"phase: {result.state.phase.value}",
        f"stage: {result.debug.stage}",
        f"message: {result.debug.message}",
        (
            "pose: "
            f"x={frame.pose.x_m:.2f} m, y={frame.pose.y_m:.2f} m, "
            f"yaw={frame.pose.yaw_rad:.2f} rad"
        ),
    ]
    if observation is not None:
        lines.append(f"visibility: {observation.visibility.value}")
        if observation.direction_score is not None:
            lines.append(f"direction score: {observation.direction_score:.3f}")
        if observation.reason:
            lines.append(f"observation: {observation.reason}")
    if result.debug.details:
        details = dict(result.debug.details)
        details.pop("frontier_candidates", None)
        lines.append(f"details: {details}")
    return lines


__all__ = ["RerunVisualizer"]
