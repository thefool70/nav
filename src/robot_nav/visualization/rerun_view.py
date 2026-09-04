"""把导航决策帧和 Adapter 运动帧记录到 Rerun。"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..adapters.perception import LocalPerceptionEvent, VlmInteraction
from ..core.geometry import world_to_nearest_grid_cell
from ..core.models import (
    DepthImage,
    Grid,
    MaskImage,
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
TRAJECTORY_RGB = (0, 120, 255)
COMMAND_RGB = (255, 140, 0)
MOTION_TARGET_RGB = (255, 70, 70)
MOTION_PATH_RGB = (190, 90, 255)
BBOX_RGB = (0, 255, 80)
SAM2_MASK_RGB = (255, 60, 180)
SAM2_MASK_ALPHA = 0.38
MAP_FRONTIER_RGB = (0, 220, 100)
SELECTED_FRONTIER_RGB = (255, 210, 0)
OBSERVATION_NODE_RGB = (245, 245, 245)
SCAN_HEADING_RGB = (100, 170, 255)
CURRENT_SCAN_HEADING_RGB = (255, 100, 180)
WORLD_HUD_RGB = (210, 225, 235)
SELECTED_FRONTIER_RADIUS_CELLS = 1

FrontierMarker = Tuple[int, int, Tuple[int, int, int], int]

DIRECTION_COLORS = {
    SearchDirectionState.PENDING: (255, 210, 0),
    SearchDirectionState.COMMITTED: (0, 170, 255),
    SearchDirectionState.EXPLORED: (130, 130, 130),
    SearchDirectionState.INVALIDATED: (255, 70, 70),
}

COMMAND_HEADING_LENGTH_M = 0.6
SCAN_HEADING_LENGTH_M = 1.2
HISTORY_DASH_LENGTH_M = 0.12
HISTORY_DASH_GAP_M = 0.08
ROBOT_FRONT_M = 0.30
ROBOT_REAR_M = 0.20
ROBOT_HALF_WIDTH_M = 0.22
ROBOT_LINE_RADIUS_M = 0.035
WORLD_HUD_MARGIN_RATIO = 0.04
OCCUPANCY_THRESHOLD = 0.5

STATUS_IMAGE_WIDTH = 560
STATUS_FONT_SIZE = 16
STATUS_LINE_SPACING_PX = 6
STATUS_PADDING_PX = 12
STATUS_BG_RGB = (24, 24, 24)
STATUS_TEXT_RGB = (235, 235, 235)

VLM_CARD_MIN_WIDTH = 960
VLM_CARD_MAX_WIDTH = 1400
VLM_CARD_PADDING_PX = 24
VLM_CARD_GAP_PX = 14
VLM_CARD_BORDER_RGB = (75, 90, 105)
VLM_CARD_TITLE_RGB = (100, 210, 255)
VLM_CARD_SECTION_RGB = (255, 190, 90)
VLM_CARD_IMAGE_BORDER_RGB = (120, 135, 150)

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
        self._log_lock = threading.RLock()
        self._sample_index = 0
        self._trajectory_xy: List[Tuple[float, float]] = []
        self._frontier_markers: Tuple[FrontierMarker, ...] = ()
        self._current_frontiers: Tuple[Mapping[str, Any], ...] = ()
        self._selected_frontier_id: Optional[str] = None
        self._last_result: Optional[NavigationResult] = None
        self._last_frame: Optional[NavigationFrame] = None
        self._vlm_samples: Dict[int, int] = {}
        self._panel_font = _load_panel_font()
        if self._panel_font is None:
            _print_font_notice_once()
        rr.init("robot-nav")
        _send_default_blueprint(
            rr,
            text_panels_as_images=self._panel_font is not None,
        )
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
        with self._log_lock:
            self._log_cycle(frame, observation, result)

    def _log_cycle(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
        result: NavigationResult,
    ) -> None:
        self._begin_sample()
        self._last_frame = frame
        self._last_result = result
        self._update_frontier_markers(result)
        self._log_rgb(frame, observation)
        self._log_sam2_result(frame, observation)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)
        self._log_command(frame, result)
        self._log_scan_plan(frame, result)
        self._log_current_frontiers()
        self._log_history(result)
        self._log_world_hud(frame, result)
        self._log_status(frame, observation, result)

    def log_motion_frame(self, frame: NavigationFrame) -> None:
        """记录 Adapter 执行动作后的传感器帧，不推进算法状态。"""
        with self._log_lock:
            self._log_motion_frame(frame)

    def _log_motion_frame(self, frame: NavigationFrame) -> None:
        self._begin_sample()
        self._last_frame = frame
        self._log_rgb(frame, None)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)
        if self._last_result is not None:
            self._log_world_hud(frame, self._last_result)

    def log_motion_plan(
        self,
        target_world_xy: Optional[Tuple[float, float]],
        remaining_path_world_xy: Tuple[Tuple[float, float], ...],
    ) -> None:
        """显示 Adapter 实际采用的目标与规划路径。"""
        with self._log_lock:
            if self._sample_index == 0:
                self._begin_sample()
            else:
                self._rr.set_time_sequence("frame", self._sample_index)
            self._log_motion_plan(
                target_world_xy,
                remaining_path_world_xy,
            )

    def _log_motion_plan(
        self,
        target_world_xy: Optional[Tuple[float, float]],
        remaining_path_world_xy: Tuple[Tuple[float, float], ...],
    ) -> None:
        if target_world_xy is None:
            self._rr.log("world/motion_plan", self._rr.Clear(recursive=True))
            self._rr.log(
                "map/occupancy/motion_plan",
                self._rr.Clear(recursive=True),
            )
            return

        target_view = _world_to_view_point(target_world_xy)
        self._rr.log(
            "world/motion_plan/target",
            self._rr.Points2D(
                [target_view],
                colors=[MOTION_TARGET_RGB],
                radii=0.13,
            ),
        )
        if len(remaining_path_world_xy) >= 2:
            self._rr.log(
                "world/motion_plan/path",
                self._rr.LineStrips2D(
                    [[
                        _world_to_view_point(point)
                        for point in remaining_path_world_xy
                    ]],
                    colors=[MOTION_PATH_RGB],
                    radii=0.035,
                ),
            )
        else:
            self._clear("world/motion_plan/path")

        frame = self._last_frame
        if frame is None:
            return
        target_pixel = _world_to_map_pixel(target_world_xy, frame)
        if target_pixel is None:
            self._clear("map/occupancy/motion_plan/target")
        else:
            self._rr.log(
                "map/occupancy/motion_plan/target",
                self._rr.Points2D(
                    [target_pixel],
                    colors=[MOTION_TARGET_RGB],
                    radii=4.0,
                    draw_order=35.0,
                ),
            )

        path_pixels = tuple(
            pixel
            for point in remaining_path_world_xy
            for pixel in [_world_to_map_pixel(point, frame)]
            if pixel is not None
        )
        if (
            len(path_pixels) >= 2
            and len(path_pixels) == len(remaining_path_world_xy)
        ):
            self._rr.log(
                "map/occupancy/motion_plan/path",
                self._rr.LineStrips2D(
                    [path_pixels],
                    colors=[MOTION_PATH_RGB],
                    radii=1.5,
                    draw_order=34.0,
                ),
            )
        else:
            self._clear("map/occupancy/motion_plan/path")

    def log_local_perception(
        self,
        frame: NavigationFrame,
        event: LocalPerceptionEvent,
    ) -> None:
        """更新后台检测面板，不重复写入导航帧、地图或轨迹。"""
        with self._log_lock:
            observation = event.observation
            if self._sample_index == 0:
                self._begin_sample()
            else:
                self._rr.set_time_sequence("frame", self._sample_index)
            self._log_local_detection_image(frame, observation)
            self._log_sam2_result(frame, observation)
            confidence = (
                "-"
                if observation.confidence is None
                else f"{observation.confidence:.3f}"
            )
            status_lines = (
                "YOLO-World + SAM2 live detection",
                f"sequence: {event.sequence_index}",
                f"visibility: {observation.visibility.value}",
                f"confidence: {confidence}",
                f"candidates: {event.candidate_count}",
                f"inference: {event.inference_s:.3f} s",
                f"reason: {observation.reason or '-'}",
            )
            if self._panel_font is None:
                self._rr.log(
                    "model/yolo_world/status",
                    self._rr.TextDocument(
                        "\n".join(_ascii_only(line) for line in status_lines),
                        media_type="text/plain",
                    ),
                )
            else:
                self._rr.log(
                    "model/yolo_world/status",
                    self._rr.Image(
                        _render_status_image(self._panel_font, status_lines)
                    ),
                )

    def _log_local_detection_image(
        self,
        frame: NavigationFrame,
        observation: TargetObservation,
    ) -> None:
        """显示后台推理实际处理的 RGB，不覆盖主相机视图。"""
        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            self._clear("model/yolo_world/latest")
            return

        image = _rgb_to_numpy(frame.rgb)
        if observation.target_mask is not None:
            image = _overlay_target_mask(image, observation.target_mask)
        else:
            image = image.copy()
        if observation.bbox_norm is not None:
            _draw_bbox_outline(image, observation.bbox_norm, BBOX_RGB)
        self._rr.log("model/yolo_world/latest", self._rr.Image(image))

    def log_vlm_interaction(
        self,
        interaction: VlmInteraction,
    ) -> None:
        """把一次 VLM 请求和回应收纳到同一张交互卡片。"""
        with self._log_lock:
            self._log_vlm_interaction(interaction)

    def _log_vlm_interaction(
        self,
        interaction: VlmInteraction,
    ) -> None:
        sample_index = self._vlm_samples.get(interaction.interaction_id)
        if sample_index is None:
            if self._sample_index == 0:
                self._begin_sample()
            sample_index = self._sample_index
            self._vlm_samples[interaction.interaction_id] = sample_index
        self._rr.set_time_sequence("frame", sample_index)
        self._log_vlm_card(interaction)

    def _log_vlm_card(self, interaction: VlmInteraction) -> None:
        """用本机 CJK 字体渲染单一卡片，避免 Rerun 中文字体问题。"""
        if self._panel_font is None:
            self._rr.log(
                "model/interaction",
                self._rr.TextDocument(
                    _ascii_only(_vlm_card_text(interaction)),
                    media_type="text/plain",
                ),
            )
            return
        self._rr.log(
            "model/interaction",
            self._rr.Image(
                _render_vlm_interaction_card(
                    self._panel_font,
                    interaction,
                )
            ),
        )

    def _begin_sample(self) -> None:
        """让决策帧与运动帧共享同一条递增时间轴。"""
        self._sample_index += 1
        self._rr.set_time_sequence("frame", self._sample_index)

    def _log_rgb(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
    ) -> None:
        """记录 RGB，叠加 SAM2 掩码，并把检测框换算为像素框。"""
        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            self._rr.log("camera/rgb", self._rr.Clear(recursive=True))
            return

        image = _rgb_to_numpy(frame.rgb)
        if observation is not None and observation.target_mask is not None:
            image = _overlay_target_mask(image, observation.target_mask)
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
                labels=[observation.source or "target"],
            ),
        )

    def _log_sam2_result(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
    ) -> None:
        """单独保留最近一次 SAM2 成功结果，避免随后运动帧将其冲走。"""
        if observation is None:
            return

        status_lines = [
            "SAM2 current observation",
            f"visibility: {observation.visibility.value}",
        ]
        if observation.target_mask is None:
            status_lines.append("mask: not available in this observation")
            self._rr.log(
                "model/sam2/status",
                self._rr.TextDocument(
                    "\n".join(status_lines),
                    media_type="text/plain",
                ),
            )
            return

        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            status_lines.append("mask: RGB image is unavailable")
            self._rr.log(
                "model/sam2/status",
                self._rr.TextDocument(
                    "\n".join(status_lines),
                    media_type="text/plain",
                ),
            )
            return

        image = _rgb_to_numpy(frame.rgb)
        mask = _target_mask_to_numpy(observation.target_mask)
        if mask is None:
            status_lines.append("mask: invalid array")
        elif mask.shape != image.shape[:2]:
            status_lines.append(
                "mask: size mismatch "
                f"({mask.shape[1]}x{mask.shape[0]} vs "
                f"RGB {image.shape[1]}x{image.shape[0]})"
            )
        elif not np.any(mask):
            status_lines.append("mask: empty")
        else:
            result_image = _overlay_target_mask(image, observation.target_mask)
            if observation.bbox_norm is not None:
                _draw_bbox_outline(
                    result_image,
                    observation.bbox_norm,
                    BBOX_RGB,
                )
            foreground = int(np.count_nonzero(mask))
            status_lines.extend(
                (
                    "mask: success",
                    f"size: {mask.shape[1]}x{mask.shape[0]}",
                    f"foreground: {foreground} px",
                    "display: magenta SAM2 mask, green detector box",
                    "image: latest successful result is retained",
                )
            )
            self._rr.log(
                "model/sam2/latest_success",
                self._rr.Image(result_image),
            )

        self._rr.log(
            "model/sam2/status",
            self._rr.TextDocument(
                "\n".join(status_lines),
                media_type="text/plain",
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
        """显示占据栅格，并在对应格子直接叠加当前 Frontier。"""
        occupancy = frame.obstacle_map.occupancy
        if len(occupancy) == 0 or len(occupancy[0]) == 0:
            self._rr.log("map/occupancy", self._rr.Clear(recursive=True))
            return
        image = _occupancy_to_rgb_numpy(occupancy)
        _draw_frontier_markers(image, self._frontier_markers)
        self._rr.log("map/occupancy", self._rr.Image(np.flipud(image).copy()))

    def _update_frontier_markers(self, result: NavigationResult) -> None:
        """保存当前决策的 Frontier，供随后运动帧继续显示。"""
        if "frontier_candidates" not in result.debug.details:
            # 一轮多视角扫描的后续周期不重复携带格子，继续显示首次规划结果。
            if not result.debug.stage.startswith("scan."):
                self._frontier_markers = ()
                self._current_frontiers = ()
                self._selected_frontier_id = None
            return

        candidates = tuple(result.debug.details["frontier_candidates"])
        selected_id = result.debug.details.get("candidate_id")
        markers = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            markers.extend(
                (int(row), int(col), MAP_FRONTIER_RGB, 0)
                for row, col in candidate["frontier_cells"]
            )
            if candidate_id == selected_id:
                markers.append(
                    (
                        int(candidate["row"]),
                        int(candidate["col"]),
                        SELECTED_FRONTIER_RGB,
                        SELECTED_FRONTIER_RADIUS_CELLS,
                    )
                )
        self._frontier_markers = tuple(markers)
        self._current_frontiers = candidates
        self._selected_frontier_id = (
            None if selected_id is None else str(selected_id)
        )

    def _log_robot_pose(self, frame: NavigationFrame) -> None:
        """用朝向三角形在 world 和 map 中记录机器人实时位姿。"""
        world_position = (frame.pose.x_m, frame.pose.y_m)
        triangle_world = _robot_triangle_world(frame.pose)
        self._rr.log(
            "world/robot",
            self._rr.LineStrips2D(
                [[_world_to_view_point(point) for point in triangle_world]],
                colors=[ROBOT_RGB],
                radii=ROBOT_LINE_RADIUS_M,
            ),
        )

        if not self._trajectory_xy or world_position != self._trajectory_xy[-1]:
            self._trajectory_xy.append(world_position)
        if len(self._trajectory_xy) >= 2:
            self._rr.log(
                "world/trajectory",
                self._rr.LineStrips2D(
                    [[_world_to_view_point(point) for point in self._trajectory_xy]],
                    colors=[TRAJECTORY_RGB],
                    radii=0.025,
                ),
            )
        self._log_robot_pose_on_map(frame)

    def _log_robot_pose_on_map(self, frame: NavigationFrame) -> None:
        """把机器人朝向三角形换算到翻转后的地图图像坐标。"""
        triangle = tuple(
            pixel
            for point in _robot_triangle_world(frame.pose)
            for pixel in [_world_to_map_pixel(point, frame)]
            if pixel is not None
        )
        if len(triangle) != 4:
            self._rr.log(
                "map/occupancy/robot", self._rr.Clear(recursive=True)
            )
            return

        self._rr.log(
            "map/occupancy/robot",
            self._rr.LineStrips2D(
                [triangle],
                colors=[ROBOT_RGB],
                radii=1.8,
                draw_order=30.0,
            ),
        )

        trajectory = tuple(
            pixel
            for point in self._trajectory_xy
            for pixel in [_world_to_map_pixel(point, frame)]
            if pixel is not None
        )
        if len(trajectory) < 2:
            self._clear("map/occupancy/trajectory")
            return
        self._rr.log(
            "map/occupancy/trajectory",
            self._rr.LineStrips2D(
                [trajectory],
                colors=[TRAJECTORY_RGB],
                radii=1.0,
                draw_order=20.0,
            ),
        )

    def _log_command(self, frame: NavigationFrame, result: NavigationResult) -> None:
        """在 world 和 map 中显示本周期平移目标与目标朝向。"""
        command = result.command
        if command is None:
            self._rr.log("world/command", self._rr.Clear(recursive=True))
            self._rr.log(
                "map/occupancy/command", self._rr.Clear(recursive=True)
            )
            return

        origin = (frame.pose.x_m, frame.pose.y_m)
        world_vector = _command_world_vector(command, frame.pose)
        target = (origin[0] + world_vector[0], origin[1] + world_vector[1])
        view_origin = _world_to_view_point(origin)
        self._rr.log(
            "world/command/translation",
            self._rr.Arrows2D(
                origins=[view_origin],
                vectors=[_world_to_view_vector(world_vector)],
                colors=[COMMAND_RGB],
                radii=0.04,
            ),
        )
        self._rr.log(
            "world/command/heading",
            self._rr.Arrows2D(
                origins=[view_origin],
                vectors=[
                    _world_yaw_to_view_vector(
                        frame.pose.yaw_rad + command.yaw_rad,
                        COMMAND_HEADING_LENGTH_M,
                    )
                ],
                colors=[COMMAND_RGB],
                radii=0.025,
            ),
        )

        origin_pixel = _world_to_map_pixel(origin, frame)
        target_pixel = _world_to_map_pixel(target, frame)
        if origin_pixel is None or target_pixel is None:
            self._rr.log(
                "map/occupancy/command", self._rr.Clear(recursive=True)
            )
            return
        pixel_vector = (
            target_pixel[0] - origin_pixel[0],
            target_pixel[1] - origin_pixel[1],
        )
        self._rr.log(
            "map/occupancy/command/translation",
            self._rr.Arrows2D(
                origins=[origin_pixel],
                vectors=[pixel_vector],
                colors=[COMMAND_RGB],
                radii=1.5,
                draw_order=32.0,
            ),
        )
        self._rr.log(
            "map/occupancy/command/heading",
            self._rr.Arrows2D(
                origins=[target_pixel],
                vectors=[
                    _world_yaw_to_map_vector(
                        frame.pose.yaw_rad + command.yaw_rad,
                        frame,
                        COMMAND_HEADING_LENGTH_M,
                    )
                ],
                colors=[COMMAND_RGB],
                radii=1.2,
                draw_order=32.0,
            ),
        )

    def _log_scan_plan(
        self, frame: NavigationFrame, result: NavigationResult
    ) -> None:
        """显示本轮全部机器人扫描朝向，并突出下一个待执行朝向。"""
        headings = result.state.scan_headings_world_rad
        if not headings:
            self._rr.log("world/scan", self._rr.Clear(recursive=True))
            return

        origin = (frame.pose.x_m, frame.pose.y_m)
        view_origin = _world_to_view_point(origin)
        strips = [
            [
                view_origin,
                _world_to_view_point(
                    _point_along_heading(
                        origin, heading, SCAN_HEADING_LENGTH_M
                    )
                ),
            ]
            for heading in headings
        ]
        self._rr.log(
            "world/scan/planned",
            self._rr.LineStrips2D(
                strips,
                colors=[SCAN_HEADING_RGB] * len(strips),
                radii=0.012,
            ),
        )
        current_index = result.state.next_scan_index
        current_heading = headings[current_index]
        self._rr.log(
            "world/scan/current",
            self._rr.Arrows2D(
                origins=[view_origin],
                vectors=[
                    _world_yaw_to_view_vector(
                        current_heading, SCAN_HEADING_LENGTH_M
                    )
                ],
                colors=[CURRENT_SCAN_HEADING_RGB],
                radii=0.025,
            ),
        )

    def _log_current_frontiers(self) -> None:
        """在 world 中显示当前有效 Frontier 的代表点和评分摘要。"""
        if not self._current_frontiers:
            self._clear("world/current_frontiers")
            return

        positions = []
        colors = []
        for candidate in self._current_frontiers:
            candidate_id = str(candidate["candidate_id"])
            positions.append(
                _world_to_view_point(
                    (
                        float(candidate["world_x_m"]),
                        float(candidate["world_y_m"]),
                    )
                )
            )
            colors.append(
                SELECTED_FRONTIER_RGB
                if candidate_id == self._selected_frontier_id
                else MAP_FRONTIER_RGB
            )
        self._rr.log(
            "world/current_frontiers",
            self._rr.Points2D(
                positions,
                colors=colors,
                radii=0.14,
            ),
        )

    def _log_history(self, result: NavigationResult) -> None:
        """显示历史观测节点、候选点，以及两者之间的状态着色虚线。"""
        node_positions = []
        candidate_positions = []
        candidate_colors = []
        link_segments = []
        link_colors = []
        for node in result.state.observation_history:
            node_view_position = _world_to_view_point(node.position_world_xy)
            node_positions.append(node_view_position)
            for direction in node.directions:
                if direction.candidate_world_xy is None:
                    continue
                color = DIRECTION_COLORS[direction.state]
                candidate_view_position = _world_to_view_point(
                    direction.candidate_world_xy
                )
                candidate_positions.append(candidate_view_position)
                candidate_colors.append(color)
                segments = _dashed_line_segments(
                    node_view_position,
                    candidate_view_position,
                )
                link_segments.extend(segments)
                link_colors.extend([color] * len(segments))

        if node_positions:
            self._rr.log(
                "world/history/nodes",
                self._rr.Points2D(
                    node_positions,
                    colors=[OBSERVATION_NODE_RGB] * len(node_positions),
                    radii=0.16,
                ),
            )
        else:
            self._clear("world/history/nodes")

        if candidate_positions:
            self._rr.log(
                "world/history/candidates",
                self._rr.Points2D(
                    candidate_positions,
                    colors=candidate_colors,
                    radii=0.11,
                ),
            )
        else:
            self._clear("world/history/candidates")

        if not link_segments:
            self._clear("world/history/links")
            return
        self._rr.log(
            "world/history/links",
            self._rr.LineStrips2D(
                link_segments,
                colors=link_colors,
                radii=0.012,
            ),
        )

    def _log_world_hud(
        self,
        frame: NavigationFrame,
        result: NavigationResult,
    ) -> None:
        """把简要决策信息合并为 world 左上角的一张标签。"""
        view_points = [
            _world_to_view_point(point)
            for point in _world_hud_reference_points(
                frame,
                result,
                self._trajectory_xy,
                self._current_frontiers,
            )
        ]
        x_values = [point[0] for point in view_points]
        y_values = [point[1] for point in view_points]
        span = max(
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
            1.0,
        )
        margin = WORLD_HUD_MARGIN_RATIO * span
        anchor = (min(x_values) + margin, min(y_values) + margin)
        self._rr.log(
            "world/hud",
            self._rr.Points2D(
                [anchor],
                colors=[WORLD_HUD_RGB],
                radii=0.01,
                labels=[_world_hud_text(frame, result)],
                show_labels=True,
                draw_order=50.0,
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
        if self._panel_font is None:
            self._rr.log(
                "navigation/status",
                self._rr.TextDocument(
                    "\n".join(f"- {_ascii_only(line)}" for line in lines),
                    media_type="text/markdown",
                ),
            )
            return
        image = _render_status_image(self._panel_font, lines)
        self._rr.log("navigation/status", self._rr.Image(image))

    def _clear(self, path: str) -> None:
        self._rr.log(path, self._rr.Clear(recursive=False))


def _send_default_blueprint(
    rr: Any,
    *,
    text_panels_as_images: bool,
) -> None:
    """固定调试布局，避免 Rerun 为每个实体自动创建散乱视图。"""
    import rerun.blueprint as rrb

    panel_view = (
        rrb.Spatial2DView if text_panels_as_images else rrb.TextDocumentView
    )
    main_views = rrb.Vertical(
        rrb.Spatial2DView(origin="/camera/rgb", name="RGB + target"),
        rrb.Horizontal(
            rrb.Spatial2DView(origin="/map/occupancy", name="Map"),
            rrb.Spatial2DView(origin="/world", name="World"),
            column_shares=[1, 1],
        ),
        row_shares=[1, 1],
        name="Navigation",
    )
    debug_views = rrb.Tabs(
        panel_view(origin="/navigation/status", name="Status"),
        rrb.Spatial2DView(
            origin="/model/yolo_world/latest",
            name="YOLO + SAM2",
        ),
        rrb.Spatial2DView(
            origin="/model/sam2/latest_success",
            name="SAM2 mask",
        ),
        panel_view(origin="/model/interaction", name="VLM"),
        rrb.Spatial2DView(origin="/camera/depth", name="Depth"),
        active_tab=0,
        name="Debug",
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                main_views,
                debug_views,
                column_shares=[3, 2],
            ),
            auto_views=False,
            auto_layout=False,
            collapse_panels=True,
        )
    )


def _rgb_to_numpy(rgb: RgbImage) -> np.ndarray:
    return np.asarray(rgb, dtype=np.uint8)


def _overlay_target_mask(image: np.ndarray, mask: MaskImage) -> np.ndarray:
    """用半透明洋红色显示 SAM2 掩码；非法尺寸时保留原始 RGB。"""
    mask_array = _target_mask_to_numpy(mask)
    if mask_array is None or mask_array.shape != image.shape[:2]:
        return image
    if not np.any(mask_array):
        return image

    overlay = image.copy()
    foreground = overlay[mask_array].astype(np.float32)
    mask_color = np.asarray(SAM2_MASK_RGB, dtype=np.float32)
    overlay[mask_array] = np.rint(
        foreground * (1.0 - SAM2_MASK_ALPHA)
        + mask_color * SAM2_MASK_ALPHA
    ).astype(np.uint8)
    return overlay


def _target_mask_to_numpy(mask: MaskImage) -> Optional[np.ndarray]:
    """把矩形二维掩码转换为 bool 数组；非法输入返回 None。"""
    try:
        mask_array = np.asarray(mask, dtype=bool)
    except (TypeError, ValueError):
        return None
    if mask_array.ndim != 2 or mask_array.size == 0:
        return None
    return mask_array


def _draw_bbox_outline(
    image: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
) -> None:
    """在独立 SAM2 结果图上直接画 VLM 框，避免依赖视图叠加设置。"""
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox_norm
    left = max(0, min(width - 1, int(math.floor(x_min * width))))
    top = max(0, min(height - 1, int(math.floor(y_min * height))))
    right = max(0, min(width - 1, int(math.ceil(x_max * width)) - 1))
    bottom = max(0, min(height - 1, int(math.ceil(y_max * height)) - 1))
    if left > right or top > bottom:
        return

    thickness = max(2, min(5, round(min(height, width) / 160)))
    image[top : min(bottom + 1, top + thickness), left : right + 1] = color
    image[max(top, bottom - thickness + 1) : bottom + 1, left : right + 1] = color
    image[top : bottom + 1, left : min(right + 1, left + thickness)] = color
    image[top : bottom + 1, max(left, right - thickness + 1) : right + 1] = color


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


def _draw_frontier_markers(
    image: np.ndarray,
    markers: Tuple[FrontierMarker, ...],
) -> None:
    """逐格绘制 Frontier，并把选中代表点画得更醒目。"""
    height, width = image.shape[:2]
    for row, col, color, radius in markers:
        if not 0 <= row < height or not 0 <= col < width:
            continue
        row_min = max(0, row - radius)
        row_max = min(height, row + radius + 1)
        col_min = max(0, col - radius)
        col_max = min(width, col + radius + 1)
        image[row_min:row_max, col_min:col_max] = color


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


def _vlm_request_metadata(interaction: VlmInteraction) -> str:
    """构造不含凭据、但覆盖全部模型请求参数的可读文本。"""
    reasoning = interaction.reasoning_effort or "none"
    return "\n".join(
        (
            f"interaction_id: {interaction.interaction_id}",
            f"task: {interaction.task}",
            f"endpoint: {interaction.endpoint_url}",
            f"model: {interaction.model}",
            f"api_format: {interaction.api_format}",
            f"reasoning_effort: {reasoning}",
            f"max_output_tokens: {interaction.max_output_tokens}",
            (
                "image: "
                f"{interaction.image.width_px}x"
                f"{interaction.image.height_px} RGB"
            ),
        )
    )


def _vlm_card_sections(
    interaction: VlmInteraction,
) -> Tuple[Tuple[str, str], ...]:
    """按请求顺序组织卡片文本，不丢弃模型原始回应。"""
    sections = [
        ("请求参数", _vlm_request_metadata(interaction)),
        ("完整输入提示词", interaction.prompt or "<empty>"),
    ]
    if interaction.phase == "request":
        sections.append(("模型输出", "等待模型返回……"))
        return tuple(sections)

    sections.extend(
        (
            ("Assistant 输出", interaction.assistant_text or "<empty>"),
            (
                "完整 HTTP JSON",
                interaction.response_json or "<no response>",
            ),
            ("解析结果", interaction.parsed_result or "<not parsed>"),
            ("错误", interaction.error or "<none>"),
        )
    )
    return tuple(sections)


def _vlm_card_status(interaction: VlmInteraction) -> str:
    if interaction.phase == "request":
        return "等待模型返回"
    if interaction.error:
        return "请求或解析失败"
    return "已完成"


def _vlm_card_text(interaction: VlmInteraction) -> str:
    """构造单卡片的文本回退内容。"""
    parts = [
        (
            f"VLM #{interaction.interaction_id} | "
            f"{interaction.task} | {_vlm_card_status(interaction)}"
        )
    ]
    for title, body in _vlm_card_sections(interaction):
        parts.append(f"[{title}]\n{body}")
        if title == "完整输入提示词":
            parts.append(
                "[实际输入 RGB]\n"
                f"{interaction.image.width_px}x"
                f"{interaction.image.height_px} RGB"
            )
    return "\n\n".join(parts)


def _render_vlm_interaction_card(
    font: object,
    interaction: VlmInteraction,
) -> np.ndarray:
    """把完整 VLM 交互和输入图渲染为一张 CJK 卡片。"""
    from PIL import Image, ImageDraw

    input_image = _vlm_input_pil_image(font, interaction)
    max_inner_width = VLM_CARD_MAX_WIDTH - 2 * VLM_CARD_PADDING_PX
    if input_image.width > max_inner_width:
        scale = max_inner_width / input_image.width
        input_image = input_image.resize(
            (
                max_inner_width,
                max(1, round(input_image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    card_width = max(
        VLM_CARD_MIN_WIDTH,
        min(
            VLM_CARD_MAX_WIDTH,
            input_image.width + 2 * VLM_CARD_PADDING_PX,
        ),
    )
    max_text_width = card_width - 2 * VLM_CARD_PADDING_PX
    wrapped_sections = [
        (
            title,
            _wrap_multiline_text(font, body, max_text_width),
        )
        for title, body in _vlm_card_sections(interaction)
    ]
    line_height = STATUS_FONT_SIZE + STATUS_LINE_SPACING_PX
    card_height = 2 * VLM_CARD_PADDING_PX + line_height
    for title, lines in wrapped_sections:
        card_height += VLM_CARD_GAP_PX + line_height
        card_height += len(lines) * line_height
        if title == "完整输入提示词":
            card_height += VLM_CARD_GAP_PX + line_height + input_image.height

    card = Image.new("RGB", (card_width, card_height), STATUS_BG_RGB)
    draw = ImageDraw.Draw(card)
    draw.rectangle(
        (0, 0, card_width - 1, card_height - 1),
        outline=VLM_CARD_BORDER_RGB,
        width=2,
    )
    y = VLM_CARD_PADDING_PX
    draw.text(
        (VLM_CARD_PADDING_PX, y),
        (
            f"VLM #{interaction.interaction_id} | "
            f"{interaction.task} | {_vlm_card_status(interaction)}"
        ),
        font=font,
        fill=VLM_CARD_TITLE_RGB,
    )
    y += line_height

    for title, lines in wrapped_sections:
        y += VLM_CARD_GAP_PX
        draw.line(
            (
                VLM_CARD_PADDING_PX,
                y + line_height - 2,
                card_width - VLM_CARD_PADDING_PX,
                y + line_height - 2,
            ),
            fill=VLM_CARD_BORDER_RGB,
            width=1,
        )
        draw.text(
            (VLM_CARD_PADDING_PX, y),
            title,
            font=font,
            fill=VLM_CARD_SECTION_RGB,
        )
        y += line_height
        for line in lines:
            draw.text(
                (VLM_CARD_PADDING_PX, y),
                line,
                font=font,
                fill=STATUS_TEXT_RGB,
            )
            y += line_height

        if title != "完整输入提示词":
            continue
        y += VLM_CARD_GAP_PX
        draw.text(
            (VLM_CARD_PADDING_PX, y),
            (
                "实际输入 RGB "
                f"({interaction.image.width_px}x"
                f"{interaction.image.height_px})"
            ),
            font=font,
            fill=VLM_CARD_SECTION_RGB,
        )
        y += line_height
        image_x = (card_width - input_image.width) // 2
        card.paste(input_image, (image_x, y))
        draw.rectangle(
            (
                image_x,
                y,
                image_x + input_image.width - 1,
                y + input_image.height - 1,
            ),
            outline=VLM_CARD_IMAGE_BORDER_RGB,
            width=2,
        )
        y += input_image.height
    return np.asarray(card)


def _vlm_input_pil_image(font: object, interaction: VlmInteraction):
    """还原模型实际输入 RGB，并可选叠加解析成功的目标框。"""
    from PIL import Image, ImageDraw

    packed = interaction.image
    rgb = np.frombuffer(packed.rgb_bytes, dtype=np.uint8).reshape(
        packed.height_px,
        packed.width_px,
        3,
    )
    image = Image.fromarray(rgb.copy(), mode="RGB")
    if interaction.bbox_norm is None:
        return image

    x_min, y_min, x_max, y_max = interaction.bbox_norm
    left = max(0, min(image.width - 1, round(x_min * image.width)))
    top = max(0, min(image.height - 1, round(y_min * image.height)))
    right = max(0, min(image.width - 1, round(x_max * image.width)))
    bottom = max(0, min(image.height - 1, round(y_max * image.height)))
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.width, image.height) / 120))
    draw.rectangle(
        (left, top, right, bottom),
        outline=BBOX_RGB,
        width=line_width,
    )
    label = (
        "YOLO-World 候选"
        if interaction.task == "target_confirmation"
        else "VLM 目标定位"
    )
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 8
    label_height = label_box[3] - label_box[1] + 6
    label_left = min(left, max(0, image.width - label_width))
    label_top = max(0, top - label_height)
    draw.rectangle(
        (
            label_left,
            label_top,
            label_left + label_width,
            label_top + label_height,
        ),
        fill=(15, 45, 30),
    )
    draw.text(
        (label_left + 4, label_top + 2),
        label,
        font=font,
        fill=BBOX_RGB,
    )
    return image


def _wrap_multiline_text(
    font: object,
    text: str,
    max_pixels: float,
) -> List[str]:
    """保留原始换行，再按卡片宽度折行。"""
    wrapped: List[str] = []
    for source_line in text.expandtabs(4).split("\n"):
        line_parts = _wrap_text(font, source_line, max_pixels)
        wrapped.extend(line_parts if line_parts else [""])
    return wrapped


def _robot_triangle_world(
    pose: Pose2D,
) -> Tuple[Tuple[float, float], ...]:
    """返回指向机器人前方的闭合三角形世界坐标。"""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    local_points = (
        (ROBOT_FRONT_M, 0.0),
        (-ROBOT_REAR_M, ROBOT_HALF_WIDTH_M),
        (-ROBOT_REAR_M, -ROBOT_HALF_WIDTH_M),
        (ROBOT_FRONT_M, 0.0),
    )
    return tuple(
        (
            pose.x_m + forward_m * cosine - left_m * sine,
            pose.y_m + forward_m * sine + left_m * cosine,
        )
        for forward_m, left_m in local_points
    )


def _world_hud_reference_points(
    frame: NavigationFrame,
    result: NavigationResult,
    trajectory: List[Tuple[float, float]],
    current_frontiers: Tuple[Mapping[str, Any], ...],
) -> Tuple[Tuple[float, float], ...]:
    """收集 world 当前图形范围，用于把汇总标签放到左上角。"""
    points = list(trajectory)
    origin = (frame.pose.x_m, frame.pose.y_m)
    points.extend(_robot_triangle_world(frame.pose))
    if result.command is not None:
        command_vector = _command_world_vector(result.command, frame.pose)
        points.append(
            (origin[0] + command_vector[0], origin[1] + command_vector[1])
        )
    points.extend(
        (float(candidate["world_x_m"]), float(candidate["world_y_m"]))
        for candidate in current_frontiers
    )
    for node in result.state.observation_history:
        points.append(node.position_world_xy)
        points.extend(
            direction.candidate_world_xy
            for direction in node.directions
            if direction.candidate_world_xy is not None
        )
    points.extend(
        _point_along_heading(origin, heading, SCAN_HEADING_LENGTH_M)
        for heading in result.state.scan_headings_world_rad
    )
    return tuple(points)


def _world_hud_text(
    frame: NavigationFrame,
    result: NavigationResult,
) -> str:
    """构造 world 角落使用的紧凑英文摘要，避免空间标签相互遮挡。"""
    lines = [
        f"{result.debug.stage} | {result.state.phase.value}",
        (
            f"pose x={frame.pose.x_m:.2f} y={frame.pose.y_m:.2f} "
            f"yaw={math.degrees(frame.pose.yaw_rad):.1f}deg"
        ),
    ]
    headings = result.state.scan_headings_world_rad
    if headings:
        scan_index = min(result.state.next_scan_index, len(headings) - 1)
        lines.append(
            f"scan {scan_index + 1}/{len(headings)} "
            f"yaw={math.degrees(headings[scan_index]):.1f}deg"
        )
    if result.command is not None:
        command = result.command
        lines.append(
            f"command move={math.hypot(command.forward_m, command.left_m):.2f}m "
            f"turn={math.degrees(command.yaw_rad):.1f}deg"
        )
    return "\n".join(lines)


def _world_to_view_point(
    world_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """翻转世界 Y 轴，使 Rerun 的二维画布按数学坐标显示 Y 向上。"""
    return (world_xy[0], -world_xy[1])


def _world_to_view_vector(
    world_vector: Tuple[float, float],
) -> Tuple[float, float]:
    """把世界系向量转换到 Y 向上的 Rerun 二维显示坐标。"""
    return (world_vector[0], -world_vector[1])


def _world_yaw_to_view_vector(
    yaw_rad: float,
    length_m: float,
) -> Tuple[float, float]:
    return _world_to_view_vector(
        (math.cos(yaw_rad) * length_m, math.sin(yaw_rad) * length_m)
    )


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


def _world_to_map_pixel(
    world_xy: Tuple[float, float],
    frame: NavigationFrame,
) -> Optional[Tuple[float, float]]:
    """把世界点转换到 ``np.flipud`` 后的占据图像素坐标。"""
    obstacle_map = frame.obstacle_map
    row, col = world_to_nearest_grid_cell(world_xy, obstacle_map)
    height = len(obstacle_map.occupancy)
    width = len(obstacle_map.occupancy[0]) if height else 0
    if not 0 <= row < height or not 0 <= col < width:
        return None
    return (float(col), float(height - 1 - row))


def _world_yaw_to_map_vector(
    yaw_rad: float,
    frame: NavigationFrame,
    length_m: float,
) -> Tuple[float, float]:
    """把世界朝向转换为翻转后地图图像中的像素向量。"""
    obstacle_map = frame.obstacle_map
    relative_yaw = yaw_rad - obstacle_map.origin.yaw_rad
    length_cells = length_m / obstacle_map.resolution_m
    return (
        math.cos(relative_yaw) * length_cells,
        -math.sin(relative_yaw) * length_cells,
    )


def _point_along_heading(
    origin: Tuple[float, float],
    heading_rad: float,
    distance_m: float,
) -> Tuple[float, float]:
    return (
        origin[0] + math.cos(heading_rad) * distance_m,
        origin[1] + math.sin(heading_rad) * distance_m,
    )


def _dashed_line_segments(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]:
    """把一条线拆成短线段，模拟 Rerun 0.22 尚不支持的虚线。"""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        return ()
    direction_x = delta_x / length
    direction_y = delta_y / length
    segments = []
    distance = 0.0
    while distance < length:
        segment_end = min(distance + HISTORY_DASH_LENGTH_M, length)
        segments.append(
            (
                (
                    start[0] + direction_x * distance,
                    start[1] + direction_y * distance,
                ),
                (
                    start[0] + direction_x * segment_end,
                    start[1] + direction_y * segment_end,
                ),
            )
        )
        distance += HISTORY_DASH_LENGTH_M + HISTORY_DASH_GAP_M
    return tuple(segments)


def _load_panel_font() -> Optional[object]:
    """加载面板 CJK 字体；Pillow 缺失或所有候选加载失败时返回 None。"""
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
    print("提示：Pillow 或 CJK 字体不可用，面板回退为 ASCII 文本")


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
    """构造按决策顺序组织的状态面板，避免直接打印难读的 details 字典。"""
    lines = [
        f"target: {target_text}",
        (
            f"decision: status={result.status.value}, "
            f"phase={result.state.phase.value}, stage={result.debug.stage}"
        ),
        f"message: {result.debug.message}",
        (
            "pose: "
            f"x={frame.pose.x_m:.2f} m, y={frame.pose.y_m:.2f} m, "
            f"yaw={math.degrees(frame.pose.yaw_rad):.1f} deg"
        ),
    ]
    if result.command is not None:
        command = result.command
        world_vector = _command_world_vector(command, frame.pose)
        target_world = (
            frame.pose.x_m + world_vector[0],
            frame.pose.y_m + world_vector[1],
        )
        lines.append(
            "command: "
            f"move={math.hypot(command.forward_m, command.left_m):.2f} m, "
            f"forward={command.forward_m:.2f} m, left={command.left_m:.2f} m, "
            f"turn={math.degrees(command.yaw_rad):.1f} deg, "
            f"target=({target_world[0]:.2f}, {target_world[1]:.2f})"
        )

    details = result.debug.details
    if "scan_heading_count" in details:
        target_heading_deg = math.degrees(
            float(details.get("target_heading_world_rad", 0.0))
        )
        lines.append(
            "scan plan: "
            f"mode={details.get('scan_mode')}, "
            f"view={int(details.get('scan_index', 0)) + 1}/"
            f"{details.get('scan_heading_count')}, "
            f"target yaw={target_heading_deg:.1f} deg"
        )
    if "frontier_scan_candidate_count" in details:
        lines.append(
            "scan source: "
            f"clusters={details['frontier_scan_candidate_count']}, "
            f"raw_cells={details['frontier_scan_cell_count']}"
        )

    candidates = details.get("frontier_candidates", ())
    selected_id = details.get("candidate_id")
    if selected_id is not None:
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["candidate_id"] == selected_id
            ),
            None,
        )
        if selected is not None:
            semantic_score = selected["semantic_score"]
            semantic_text = (
                "none"
                if semantic_score is None
                else f"{float(semantic_score):.2f}"
            )
            lines.append(
                "selected frontier: "
                f"{selected_id}, rank=1/{details.get('candidate_count')}, "
                f"path={float(selected['path_distance_m']):.2f} m, "
                f"span={float(selected['frontier_span_m']):.2f} m, "
                f"vlm={semantic_text}, "
                f"semantic bonus={float(selected['semantic_bonus']):.2f}, "
                f"score={float(selected['score']):.2f}"
            )

    context = [
        f"{key}={details[key]}"
        for key in ("node_id", "direction_id", "reason")
        if key in details
    ]
    if context:
        lines.append("context: " + ", ".join(context))

    if observation is not None:
        lines.append(f"visibility: {observation.visibility.value}")
        if observation.source:
            confidence = (
                "none"
                if observation.confidence is None
                else f"{observation.confidence:.3f}"
            )
            lines.append(
                f"detector: source={observation.source}, confidence={confidence}"
            )
        if observation.target_mask is not None:
            mask = _target_mask_to_numpy(observation.target_mask)
            if mask is None:
                lines.append("sam2 mask: invalid")
            elif (
                frame.rgb is not None
                and len(frame.rgb) > 0
                and len(frame.rgb[0]) > 0
                and mask.shape != (len(frame.rgb), len(frame.rgb[0]))
            ):
                lines.append(
                    "sam2 mask: size mismatch, "
                    f"mask={mask.shape[1]}x{mask.shape[0]}, "
                    f"rgb={len(frame.rgb[0])}x{len(frame.rgb)}"
                )
            else:
                lines.append(
                    "sam2 mask: magenta overlay, "
                    f"size={mask.shape[1]}x{mask.shape[0]}, "
                    f"foreground={int(np.count_nonzero(mask))} px"
                )
        if observation.reason:
            lines.append(f"observation: {observation.reason}")

    direction_counts = {state.value: 0 for state in SearchDirectionState}
    for node in result.state.observation_history:
        for direction in node.directions:
            direction_counts[direction.state.value] += 1
    lines.append(
        "history: "
        f"nodes={len(result.state.observation_history)}, "
        f"pending={direction_counts['pending']}, "
        f"committed={direction_counts['committed']}, "
        f"explored={direction_counts['explored']}, "
        f"invalidated={direction_counts['invalidated']}"
    )
    lines.extend(
        (
            "map legend: robot=blue, frontier=green, selected=yellow, "
            "command=orange, adapter target=red, adapter path=purple",
            "history links: pending=yellow, committed=blue, explored=gray, "
            "invalidated=red",
        )
    )
    return lines


__all__ = ["RerunVisualizer"]
