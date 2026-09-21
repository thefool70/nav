"""把导航决策帧和 Adapter 运动帧记录到 Rerun。"""

from __future__ import annotations

from ..core.actions import action_command

import math
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np

from ..adapters.perception import VlmInteraction
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
from .vlm_trace import VlmTraceHistory, context_text, job_path, request_path
from .semantic_world import SemanticWorldNodes


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
VLM_CAPTURE_RGB = (0, 220, 220)
VLM_SNAPSHOT_RGB = (255, 100, 190)
OBSERVATION_NODE_RGB = (245, 245, 245)
SCAN_HEADING_RGB = (100, 170, 255)
CURRENT_SCAN_HEADING_RGB = (255, 100, 180)
SELECTED_FRONTIER_RADIUS_CELLS = 1

FrontierMarker = Tuple[int, int, Tuple[int, int, int], int]

DIRECTION_COLORS = {
    SearchDirectionState.PENDING: (255, 210, 0),
    SearchDirectionState.COMMITTED: (0, 170, 255),
    SearchDirectionState.EXPLORED: (130, 130, 130),
    SearchDirectionState.INVALIDATED: (255, 70, 70),
    SearchDirectionState.STALLED: (255, 150, 40),
}

COMMAND_HEADING_LENGTH_M = 0.6
SCAN_HEADING_LENGTH_M = 1.2
HISTORY_DASH_LENGTH_M = 0.12
HISTORY_DASH_GAP_M = 0.08
ROBOT_FRONT_M = 0.30
ROBOT_REAR_M = 0.20
ROBOT_HALF_WIDTH_M = 0.22
ROBOT_LINE_RADIUS_M = 0.035
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
RERUN_SERVER_MEMORY_LIMIT = "25%"


class RerunVisualizer:
    """把传感器、地图、轨迹和算法状态同时送往实时界面与 RRD 文件。"""

    def __init__(
        self,
        target_text: str,
        *,
        recording_path: Optional[Path] = None,
    ) -> None:
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
        self._vlm_trace = VlmTraceHistory()
        self._world_nodes = SemanticWorldNodes(rr, self._log)
        self._object_progress = None
        self._panel_font = _load_panel_font()
        if self._panel_font is None:
            _print_font_notice_once()
        rr.init("robot-nav")
        live_recording = rr.get_data_recording()
        if live_recording is None:
            raise RuntimeError("Rerun 未创建实时记录流")

        if recording_path is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
            recording_path = Path("data/run_logs") / f"rerun-{timestamp}-{os.getpid()}.rrd"
        recording_path = recording_path.expanduser().resolve()
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        # save 会覆盖文件；先独占创建，拒绝覆盖用户已有的录制。
        with recording_path.open("xb"):
            pass
        # 0.22.1 的 save / serve_web 会替换各自记录流的 sink。
        # 用不同 ID 建立独立文件流，避免保存文件时关闭实时服务。
        disk_recording = rr.new_recording("robot-nav", recording_id=uuid4())
        rr.save(recording_path, recording=disk_recording)
        self._recordings = (disk_recording, live_recording)
        print(f"Rerun 自动录制文件：{recording_path}", flush=True)
        # Rerun 0.22.1 的 serve_web 持有 GIL 等待 sink 切换完成；必须在
        # 发送 blueprint / Arrow 数据前启动，避免后台释放数据时争用 GIL。
        # 浏览器由用户手动打开，Rerun 启动不依赖 WSL 的浏览器调用。
        print(
            "正在启动 Rerun Web 服务；启动后请手动打开显示的地址。",
            flush=True,
        )
        rr.serve_web(
            open_browser=False,
            web_port=9090,
            ws_port=9877,
            recording=live_recording,
            server_memory_limit=RERUN_SERVER_MEMORY_LIMIT,
        )
        print("Rerun Web 服务已启动；正在发送界面布局。", flush=True)
        _send_default_blueprint(
            rr,
            recordings=self._recordings,
            text_panels_as_images=self._panel_font is not None,
        )
        # 0.22.1 的 flush 位于记录流对象上；等待时会释放 GIL。
        # 文件由 SDK 后台持续写入；SDK 的退出钩子负责刷新并关闭两个流。
        for recording in self._recordings:
            recording.flush(blocking=True)
        print(
            "Rerun Web Viewer 地址："
            "http://127.0.0.1:9090/?url=ws://127.0.0.1:9877",
            flush=True,
        )

    def _log(
        self,
        entity_path: str,
        entity: Any,
        *extra: Any,
        static: bool = False,
    ) -> None:
        """同一份可视化数据写入文件和实时流，不依赖 Viewer 是否保留旧帧。"""
        for recording in self._recordings:
            self._rr.log(entity_path, entity, *extra, static=static, recording=recording)

    def _set_frame_time(self, sample_index: int) -> None:
        """每个回调线程都为两个记录流设置相同的 frame 时间。"""
        for recording in self._recordings:
            self._rr.set_time_sequence("frame", sample_index, recording=recording)

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
        self._object_progress = None
        self._world_nodes.set_map(frame.obstacle_map.frame_id)
        self._world_nodes.record_cycle(result)
        self._vlm_trace.record_cycle(result, self._sample_index)
        updated_jobs = {
            item["job_id"] for name in ("semantic_received_jobs", "semantic_score_sources")
            for item in result.debug.details.get(name, ())
        }
        for job_id in sorted(updated_jobs):
            self._log_vlm_job(job_id)
        self._update_frontier_markers(result)
        self._log_rgb(frame, observation)
        self._log_sam2_result(frame, observation)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)
        self._log_command(frame, result)
        self._log_scan_plan(frame, result)
        self._log_current_frontiers(result)
        self._log_history(result)
        self._log_motion_status(frame, result)
        self._log_status(frame, observation, result)
        self._log_vlm_overview()

    def log_motion_frame(self, frame: NavigationFrame) -> None:
        """记录 Adapter 执行动作后的传感器帧，不推进算法状态。"""
        with self._log_lock:
            self._log_motion_frame(frame)

    def _log_motion_frame(self, frame: NavigationFrame) -> None:
        self._begin_sample()
        self._last_frame = frame
        self._world_nodes.set_map(frame.obstacle_map.frame_id)
        self._log_rgb(frame, None)
        self._log_depth(frame)
        self._log_occupancy_map(frame)
        self._log_robot_pose(frame)
        if self._last_result is not None:
            self._log_motion_status(frame, self._last_result)

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
                self._set_frame_time(self._sample_index)
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
            self._log("world/motion_plan", self._rr.Clear(recursive=True))
            self._log(
                "map/occupancy/motion_plan",
                self._rr.Clear(recursive=True),
            )
            return

        target_view = _world_to_view_point(target_world_xy)
        self._log(
            "world/motion_plan/target",
            self._rr.Points2D(
                [target_view],
                colors=[MOTION_TARGET_RGB],
                radii=0.13,
            ),
        )
        if len(remaining_path_world_xy) >= 2:
            self._log(
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
            self._log(
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
            self._log(
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



    def log_vlm_interaction(
        self,
        interaction: VlmInteraction,
    ) -> None:
        """完整卡片按真实到达时刻记录，简表和归档通过请求编号关联。"""
        with self._log_lock:
            self._log_vlm_interaction(interaction)

    def _log_vlm_interaction(
        self,
        interaction: VlmInteraction,
    ) -> None:
        self._begin_sample()
        self._vlm_trace.record_interaction(interaction, self._sample_index)
        self._world_nodes.record_interaction(interaction)
        self._log_vlm_card(interaction, "model/interaction", request_path(interaction.interaction_id))
        self._log(request_path(interaction.interaction_id) + "/text", self._rr.TextDocument(
            _vlm_card_text(interaction), media_type="text/plain",
        ))
        self._log_vlm_overview()
        if interaction.phase == "response":
            if self._last_frame is not None and self._last_result is not None:
                self._log_motion_status(self._last_frame, self._last_result)

    def _log_vlm_card(self, interaction: VlmInteraction, *paths: str) -> None:
        """用本机 CJK 字体渲染单一卡片，避免 Rerun 中文字体问题。"""
        if self._panel_font is None:
            entity = self._rr.TextDocument(
                _ascii_only(_vlm_card_text(interaction)), media_type="text/plain",
            )
        else:
            entity = self._rr.Image(_render_vlm_interaction_card(self._panel_font, interaction))
        for path in paths:
            self._log(path, entity)

    def log_semantic_queue_event(self, event: Mapping[str, Any]) -> None:
        """接收 FIFO 生命周期事件；任务编号与模型请求编号分别显示。"""
        with self._log_lock:
            self._begin_sample()
            self._vlm_trace.record_queue_event(event, self._sample_index)
            self._world_nodes.record_queue_event(event)
            if event["event"].startswith("object_"):
                previous = self._object_progress if event["event"] == "object_localized" else None
                self._object_progress = {**(previous or {}), **event}
                if event["event"] == "object_localized":
                    self._object_progress["stage"] = "finished"
                self._log_object_progress()
            job_id = event.get("job_id")
            job_ids = () if event["event"].startswith("object_") else (
                (job_id,) if job_id is not None else event.get("job_ids", ())
            )
            for key in job_ids:
                self._log_vlm_job(key)
            self._log_vlm_overview()

    def _log_vlm_job(self, job_id: int) -> None:
        self._log(job_path(job_id), self._rr.TextDocument(
            json.dumps(self._vlm_trace.jobs[job_id], ensure_ascii=False, indent=2), media_type="text/plain",
        ))

    def _log_vlm_overview(self) -> None:
        self._world_nodes.refresh_panels(self._panel_font)
        self._log("model/vlm/summary", self._rr.TextDocument(
            self._vlm_trace.overview(), media_type="text/markdown",
        ))

    def _log_object_progress(self) -> None:
        """在阻塞推理开始前及期间更新状态，不沿用上一条运动命令。"""
        event = self._object_progress
        phase = event.get("phase", "localizing_object")
        model = event.get("model", event.get("target_source", "YOLO / VLM"))
        stage = event.get("stage", "finished" if event["event"] == "object_localized" else "starting")
        lines = [
            f"target: {self._target_text}",
            f"phase: {phase} | robot paused for perception",
            f"clue: {event.get('clue_id', '-')}",
            f"input: {event.get('source', '-')}",
            f"model: {model} / {stage}",
            f"elapsed: {event.get('elapsed_s', 0.0):.1f}s",
        ]
        pose = event.get("robot_pose")
        if pose is not None:
            lines.append(f"robot pose: ({pose['x_m']:.2f}, {pose['y_m']:.2f}) m")
        if event.get("localization_map") is not None:
            lines.append(f"localization map: {event['localization_map']}")
        if event.get("localization_method") == "obstacle_assumption":
            lines.append("position basis: first obstacle on image ray (assumption)")
        text = "\n".join(lines)
        self._log("navigation/motion", self._rr.TextDocument(text))
        self._log("navigation/live", self._rr.TextDocument(
            f"**{phase}** | {event.get('clue_id', '-')}\n\n"
            f"{model}: {stage} / {event.get('elapsed_s', 0.0):.1f}s | robot paused",
            media_type="text/markdown",
        ))
        self._log("navigation/status_text", self._rr.TextDocument(text))
        if self._panel_font is None:
            self._log("navigation/status", self._rr.TextDocument(text))
        else:
            self._log("navigation/status", self._rr.Image(_render_status_image(self._panel_font, lines)))

    def _begin_sample(self) -> None:
        """决策、运动与模型事件共享递增时间轴，迟到结果不能写回旧帧。"""
        self._sample_index += 1
        self._set_frame_time(self._sample_index)

    def _log_rgb(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
    ) -> None:
        """记录 RGB，叠加 SAM2 掩码，并把检测框换算为像素框。"""
        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            self._log("camera/rgb", self._rr.Clear(recursive=True))
            return

        image = _rgb_to_numpy(frame.rgb)
        if observation is not None and observation.target_mask is not None:
            image = _overlay_target_mask(image, observation.target_mask)
        self._log("camera/rgb", self._rr.Image(image))
        if observation is None or observation.bbox_norm is None:
            self._clear("camera/rgb/target")
            return

        box_min, box_size = _bbox_norm_to_pixel_box(
            observation.bbox_norm,
            height=image.shape[0],
            width=image.shape[1],
        )
        self._log(
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
            self._log(
                "model/sam2/status",
                self._rr.TextDocument(
                    "\n".join(status_lines),
                    media_type="text/plain",
                ),
            )
            return

        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0:
            status_lines.append("mask: RGB image is unavailable")
            self._log(
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
            self._log(
                "model/sam2/latest_success",
                self._rr.Image(result_image),
            )

        self._log(
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
        self._log(
            "camera/depth",
            self._rr.DepthImage(_depth_to_numpy(frame.depth), meter=1.0),
        )

    def _log_occupancy_map(self, frame: NavigationFrame) -> None:
        """显示占据栅格，并在对应格子直接叠加当前 Frontier。"""
        occupancy = frame.obstacle_map.occupancy
        if len(occupancy) == 0 or len(occupancy[0]) == 0:
            self._log("map/occupancy", self._rr.Clear(recursive=True))
            self._log("world/occupancy", self._rr.Clear(recursive=True))
            return
        image = _occupancy_to_rgb_numpy(occupancy)
        self._log_world_map(frame, np.flipud(image).copy())
        _draw_frontier_markers(image, self._frontier_markers)
        self._log("map/occupancy", self._rr.Image(np.flipud(image).copy()))

    def _log_world_map(self, frame: NavigationFrame, image: np.ndarray) -> None:
        """将栅格边缘对齐世界米制坐标，World 直接叠加机器人与任务标记。"""
        grid = frame.obstacle_map
        origin, scale = grid.origin, grid.resolution_m
        c, s = math.cos(origin.yaw_rad), math.sin(origin.yaw_rad)
        # origin 是左下格中心；图片从翻转后的左上角边缘开始，World 的 y 取反。
        local_x, local_y = -0.5 * scale, (image.shape[0] - 0.5) * scale
        self._log("world/occupancy", self._rr.Transform3D(
            translation=[origin.x_m + c * local_x - s * local_y,
                         -origin.y_m - s * local_x - c * local_y, 0],
            mat3x3=[[scale * c, scale * s, 0], [-scale * s, scale * c, 0], [0, 0, 1]],
        ), self._rr.Image(image, draw_order=-20))

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
        self._log(
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
            self._log(
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
            self._log(
                "map/occupancy/robot", self._rr.Clear(recursive=True)
            )
            return

        self._log(
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
        self._log(
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
        command = action_command(result.action, frame.pose)
        if command is None:
            self._log("world/command", self._rr.Clear(recursive=True))
            self._log(
                "map/occupancy/command", self._rr.Clear(recursive=True)
            )
            return

        origin = (frame.pose.x_m, frame.pose.y_m)
        world_vector = _command_world_vector(command, frame.pose)
        target = (origin[0] + world_vector[0], origin[1] + world_vector[1])
        view_origin = _world_to_view_point(origin)
        self._log(
            "world/command/translation",
            self._rr.Arrows2D(
                origins=[view_origin],
                vectors=[_world_to_view_vector(world_vector)],
                colors=[COMMAND_RGB],
                radii=0.04,
            ),
        )
        self._log(
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
            self._log(
                "map/occupancy/command", self._rr.Clear(recursive=True)
            )
            return
        pixel_vector = (
            target_pixel[0] - origin_pixel[0],
            target_pixel[1] - origin_pixel[1],
        )
        self._log(
            "map/occupancy/command/translation",
            self._rr.Arrows2D(
                origins=[origin_pixel],
                vectors=[pixel_vector],
                colors=[COMMAND_RGB],
                radii=1.5,
                draw_order=32.0,
            ),
        )
        self._log(
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
            self._log("world/scan", self._rr.Clear(recursive=True))
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
        self._log(
            "world/scan/planned",
            self._rr.LineStrips2D(
                strips,
                colors=[SCAN_HEADING_RGB] * len(strips),
                radii=0.012,
            ),
        )
        current_index = result.state.next_scan_index
        current_heading = headings[current_index]
        self._log(
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

    def _log_current_frontiers(self, result: NavigationResult) -> None:
        """World 只画候选点；编号、暂存顺序和分数在独立面板中显示。"""
        self._log(
            "navigation/frontiers",
            self._rr.TextDocument(
                _frontier_table_text(self._current_frontiers, self._selected_frontier_id, result),
                media_type="text/markdown",
            ),
        )
        if not self._current_frontiers:
            self._clear("world/current_frontiers")
            return

        positions = []
        colors = []
        labels = []
        for candidate in self._current_frontiers:
            candidate_id = str(candidate["candidate_id"])
            labels.append(candidate_id)
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
        self._log(
            "world/current_frontiers",
            self._rr.Points2D(
                positions,
                colors=colors,
                radii=0.14,
                labels=labels,
                show_labels=False,
            ),
        )

    def _log_history(self, result: NavigationResult) -> None:
        """显示实际尝试过的停靠点；未选区域只在当前 Frontier 图层显示。"""
        node_positions = []
        candidate_positions = []
        candidate_colors = []
        link_segments = []
        link_colors = []
        for node in result.state.observation_history:
            node_view_position = _world_to_view_point(node.position_world_xy)
            node_positions.append(node_view_position)
            for direction in node.directions:
                if direction.command_world_xy is None:
                    continue
                color = DIRECTION_COLORS[direction.state]
                candidate_view_position = _world_to_view_point(
                    direction.command_world_xy
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
            self._log(
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
            self._log(
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
        self._log(
            "world/history/links",
            self._rr.LineStrips2D(
                link_segments,
                colors=link_colors,
                radii=0.012,
            ),
        )

    def _log_motion_status(
        self,
        frame: NavigationFrame,
        result: NavigationResult,
    ) -> None:
        """在侧栏更新实时位姿与最近决策，不在 World 图形上叠加文字。"""
        if self._object_progress is not None:
            self._log_object_progress()
            return
        text = _motion_status_text(frame, result)
        self._log(
            "navigation/motion",
            self._rr.TextDocument(text),
        )
        self._log("navigation/live", self._rr.TextDocument(
            f"**{result.debug.stage}** | target: {self._target_text}\n\n"
            f"robot ({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}) m / "
            f"{math.degrees(frame.pose.yaw_rad):.0f} deg | {result.state.phase.value}",
            media_type="text/markdown",
        ))

    def _log_status(
        self,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
        result: NavigationResult,
    ) -> None:
        """记录状态面板；有 CJK 字体时渲染为图像，否则回退为 ASCII 文本。"""
        lines = _status_lines(self._target_text, frame, observation, result)
        # 始终保留可查询文本，离线分析 RRD 不必从中文状态图片做 OCR。
        self._log("navigation/status_text", self._rr.TextDocument("\n".join(lines)))
        if self._panel_font is None:
            self._log(
                "navigation/status",
                self._rr.TextDocument(
                    "\n".join(f"- {_ascii_only(line)}" for line in lines),
                    media_type="text/markdown",
                ),
            )
            return
        image = _render_status_image(self._panel_font, lines)
        self._log("navigation/status", self._rr.Image(image))

    def _clear(self, path: str) -> None:
        self._log(path, self._rr.Clear(recursive=False))


def _send_default_blueprint(
    rr: Any,
    *,
    recordings: Tuple[Any, ...],
    text_panels_as_images: bool,
) -> None:
    """固定调试布局，避免 Rerun 为每个实体自动创建散乱视图。"""
    import rerun.blueprint as rrb

    panel_view = (
        rrb.Spatial2DView if text_panels_as_images else rrb.TextDocumentView
    )
    world_views = rrb.Tabs(
        rrb.Spatial2DView(
            origin="/world", name="World",
            contents=["/world/**", "- /world/observations/**", "- /world/history/**", "- /world/scan/planned/**"],
        ),
        rrb.Spatial2DView(
            origin="/world", name="World history",
            contents=["/world/**", "- /world/jobs/**"],
        ),
        rrb.Spatial2DView(origin="/map/occupancy", name="Map details"),
        active_tab=0,
    )
    debug_views = rrb.Tabs(
        rrb.TextDocumentView(origin="/observations/index", name="Observations"),
        rrb.TextDocumentView(origin="/navigation/frontiers", name="Frontiers"),
        rrb.TextDocumentView(origin="/navigation/motion", name="Motion details"),
        panel_view(origin="/navigation/status", name="Status"),
        rrb.Spatial2DView(
            origin="/model/yolo_world/latest",
            name="YOLO + SAM2",
        ),
        rrb.Spatial2DView(
            origin="/model/sam2/latest_success",
            name="SAM2 mask",
        ),
        rrb.TextDocumentView(origin="/model/vlm/summary", name="VLM summary"),
        panel_view(origin="/model/interaction", name="VLM full"),
        rrb.Spatial2DView(origin="/camera/depth", name="Depth"),
        active_tab=0,
        name="Debug",
    )
    sidebar = rrb.Vertical(
        rrb.Tabs(
            rrb.Spatial2DView(origin="/observations/focus", name="Inference RGB + scores"),
            rrb.Spatial2DView(origin="/camera/rgb", name="Live camera"),
            active_tab=0,
        ),
        debug_views,
        row_shares=[3, 2],
        name="Details",
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                world_views,
                rrb.TextDocumentView(origin="/navigation/live", name="Live"),
                row_shares=[8, 1],
            ),
            sidebar,
            column_shares=[2, 1],
        ),
        rrb.SelectionPanel(state="expanded"),
        auto_views=False,
        auto_layout=False,
        collapse_panels=True,
    )
    for recording in recordings:
        rr.send_blueprint(blueprint, recording=recording)


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
            f"request_elapsed_s: {interaction.elapsed_s if interaction.elapsed_s is not None else 'in flight'}",
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
        ("来源与编号对应", context_text(interaction.context)),
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
        return "已返回（含请求或解析错误，查看有效结果）"
    return "已完成"


def _vlm_card_text(interaction: VlmInteraction) -> str:
    """构造单卡片的文本回退内容。"""
    parts = [
        (
            f"VLM R{interaction.interaction_id} | "
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
            f"VLM R{interaction.interaction_id} | "
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


def _motion_status_text(
    frame: NavigationFrame,
    result: NavigationResult,
) -> str:
    """构造侧栏实时摘要；位姿随运动帧更新，阶段来自最近一次决策。"""
    lines = [
        f"decision: {result.debug.stage} | next: {result.state.phase.value}",
        (
            f"pose x={frame.pose.x_m:.2f} y={frame.pose.y_m:.2f} "
            f"yaw={math.degrees(frame.pose.yaw_rad):.1f}deg"
        ),
    ]
    headings = result.state.scan_headings_world_rad
    if result.state.asynchronous_perception:
        lines.append(
            f"semantic pending={result.state.pending_semantic_jobs} "
            f"failed={result.state.failed_semantic_jobs} "
            f"queued views={len(result.state.pending_observation_views)}"
        )
    if result.state.active_target_clue is not None:
        lines.append(f"target clue: {result.state.active_target_clue.clue_id}")
    approach = result.state.object_approach
    if approach.target is not None:
        lines.append(
            f"object source={approach.target.source} target={approach.target.target_world_xy} "
            f"moves={len(approach.tried_positions)} depth points={approach.target.sample_count}"
        )
    planning = result.debug.details
    if "standoff_map" in planning:
        lines.append(
            f"standoff map={planning['standoff_map']} "
            f"clearance={planning['standoff_clearance_m']:.2f}m "
            f"candidates={planning['standoff_candidate_count']}"
        )
        if "standoff_distance_m" in planning:
            lines.append(f"planned distance to target={planning['standoff_distance_m']:.2f}m")
    if headings and "scan_mode" in result.debug.details:
        scan_index = min(result.state.next_scan_index, len(headings) - 1)
        lines.append(
            f"scan {scan_index + 1}/{len(headings)} "
            f"yaw={math.degrees(headings[scan_index]):.1f}deg"
        )
        lines.append(_scan_basis_text(result.debug.details["scan_mode"]))
    if action_command(result.action, frame.pose) is not None:
        command = action_command(result.action, frame.pose)
        lines.append(
            f"command move={math.hypot(command.forward_m, command.left_m):.2f}m "
            f"turn={math.degrees(command.yaw_rad):.1f}deg"
        )
    selected_id = result.debug.details.get("candidate_id")
    parent_id = result.debug.details.get("parent_node_id") or result.state.backtrack_node_id
    if parent_id is not None:
        lines.append(f"parent: {parent_id}")
    if "branch_depth" in result.debug.details:
        lines.append(
            f"return depth: {result.debug.details['branch_depth']} | "
            f"pending at parent: {result.debug.details['pending_direction_count']}"
        )
    if selected_id is not None:
        lines.append(
            f"frontier: {selected_id} ({result.debug.details.get('frontier_selection_source')})"
        )
    return "\n".join(lines)


def _scan_basis_text(mode: str) -> str:
    """说明本次观察由 Frontier、首次环扫还是当前位置场景确认触发。"""
    basis = {
        "initial": "initial 360 deg sweep",
        "frontier": "unchecked local Frontier directions",
        "current_view": "target check at current pose (no extra turn)",
    }
    return f"scan basis: {basis.get(mode, mode)}"


def _frontier_table_text(
    candidates: Tuple[Mapping[str, Any], ...],
    selected_id: Optional[str],
    result: NavigationResult,
) -> str:
    """表格行与 Points2D 实例顺序一致；编号链接到对应点，暂存状态取当前状态。"""
    blocked_ids = ", ".join(
        region.region_id for region in result.state.blocked_frontier_regions
    )
    blocked_text = (
        f"Blocked regions (no retry during this run): {blocked_ids}."
        if blocked_ids else "Blocked regions: none."
    )
    if not candidates:
        return f"No current frontier candidates.\n\n{blocked_text}"
    orders = {region.region_id: region.deferred_order for region in result.state.frontier_regions}
    lines = [
        "Click an ID to select its World point. Yellow = selected; green = candidate.",
        "",
        "| Frontier | State | Saved order | Path (m) | Score |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        order = orders.get(candidate_id, candidate.get("deferred_order"))
        state = "selected" if candidate_id == selected_id else "deferred" if order is not None else "new"
        order_text = "-" if order is None else f"{order[0]}:{order[1]}"
        label = f"[{candidate_id}](recording://world/current_frontiers[#{index}])"
        lines.append(
            f"| {label} | {state} | {order_text} | "
            f"{float(candidate['path_distance_m']):.2f} | {float(candidate['score']):.2f} |"
        )
    lines.extend([
        "",
        blocked_text,
        "",
        "New directions rank by score. When they run out, return through the branch "
        "one node at a time until a reached node has a remaining direction to explore.",
        "Positions and distances are from the latest frontier update.",
    ])
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
    if action_command(result.action, frame.pose) is not None:
        command = action_command(result.action, frame.pose)
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
    if result.state.asynchronous_perception:
        lines.append(
            "semantic queue: "
            f"pending={result.state.pending_semantic_jobs}, "
            f"finished={details.get('semantic_finished', 0)}, "
            f"failed={result.state.failed_semantic_jobs}, "
            f"active={details.get('semantic_active_job')}, "
            f"pending views={len(result.state.pending_observation_views)}"
        )
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
        lines.append(_scan_basis_text(details.get("scan_mode", "unknown")))
    if "frontier_scan_candidate_count" in details:
        lines.append(
            "frontier map: "
            f"clusters={details['frontier_scan_candidate_count']}, "
            f"candidate_cells={details['frontier_scan_cell_count']}"
        )
    if "local_observation_point_count" in details:
        lines.append(
            "frontier coverage: "
            f"local_frontier_points={details.get('local_observation_point_count', 0)}, "
            f"need_check={details.get('observation_point_count', 0)}, "
            f"reused={details.get('reused_observation_point_count', 0)}, "
            f"scan_skipped={details.get('scan_skipped', False)}"
        )

    candidates = details.get("frontier_candidates", ())
    if "frontier_selection_source" in details:
        lines.append(
            f"frontier choice: source={details['frontier_selection_source']}, "
            f"new={details['new_frontier_count']}, "
            f"deferred={details['deferred_frontier_count']}"
        )
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
                f"deferred order={selected.get('deferred_order')}, "
                f"score={float(selected['score']):.2f}"
            )

    context = [
        f"{key}={details[key]}"
        for key in ("node_id", "parent_node_id", "direction_id", "reason")
        if details.get(key) is not None
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
        f"invalidated={direction_counts['invalidated']}, "
        f"stalled={direction_counts['stalled']}"
    )
    lines.append(
        f"regions={len(result.state.frontier_regions)}, "
        f"blocked={len(result.state.blocked_frontier_regions)}, "
        f"deferred={sum(region.deferred_order is not None for region in result.state.frontier_regions)}, "
        f"active={result.state.active_frontier_id}, "
        f"checked_views={len(result.state.observed_views)}, "
        f"planned_frontier_points={len(result.state.scan_observation_points)}"
    )
    if result.state.observed_views:
        view = result.state.observed_views[-1]
        lines.append(
            f"last checked view: visible_points={len(view.visible_world_xy)}, "
            f"depth_coverage={view.depth_coverage_available}"
        )
    latest_issue = next(
        (
            (node.node_id, direction)
            for node in reversed(result.state.observation_history)
            for direction in node.directions
            if direction.execution_reason
        ),
        None,
    )
    if latest_issue is not None:
        node_id, direction = latest_issue
        lines.append(
            f"last exploration issue: {node_id}, {direction.state.value}, "
            f"{direction.execution_reason}"
        )
    if "destination_world_xy" in details:
        lines.append(f"exploration destination={details['destination_world_xy']}")
    lines.extend(
        (
            "map legend: robot=blue, frontier=green, selected=yellow, "
            "command=orange, adapter target=red, adapter path=purple",
            "history links: pending=yellow, committed=blue, explored=gray, "
            "invalidated=red, stalled=orange",
        )
    )
    return lines


__all__ = ["RerunVisualizer"]
