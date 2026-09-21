"""把一轮扫描 RGB 与 Frontier 候选组成可供 VLM 评分的拼图。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..core.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    FrontierCandidate,
    NavigationFrame,
    Pose2D,
    RgbImage,
)
from ..core.frontier_projection import FrontierImageProjection, project_frontier_ground_points
from .perception import VlmInputImage


TILE_MAX_WIDTH_PX = 320
SHEET_MAX_COLUMNS = 4
TILE_GAP_PX = 2
SHEET_BACKGROUND_RGB = (241, 245, 249)
VIEW_BADGE_RGB = (29, 78, 216)
MARKER_RGB = (255, 80, 180)
MARKER_TEXT_RGB = (15, 15, 15)
MARKER_SCALE = 3


@dataclass(frozen=True)
class BufferedScanImage:
    """不携带地图的紧凑扫描帧，避免把大图放入感知缓冲。"""

    width_px: int
    height_px: int
    rgb_bytes: bytes
    pose: Pose2D
    intrinsics: CameraIntrinsics
    camera_yaw_rad: float
    camera_extrinsics_in_robot: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    frontier_projections: Tuple[FrontierImageProjection, ...] = ()


@dataclass(frozen=True)
class FrontierImageMarker:
    """拼图中数字标记与真实 Frontier ID 的对应。"""

    label: str
    candidate_id: str
    view_id: Optional[int] = None
    source_pixel_xy: Optional[Tuple[float, float]] = None


def buffer_scan_image(
    frame: NavigationFrame, candidates: Sequence[FrontierCandidate] = (),
) -> BufferedScanImage:
    """冻结 RGB、标定和通过深度核对的锚点，后台绘图不重读传感器。"""
    if frame.rgb is None or frame.camera_intrinsics is None:
        raise ValueError("Frontier 评分需要 RGB 和相机内参")
    packed = pack_rgb_image(frame.rgb)
    return BufferedScanImage(
        width_px=packed.width_px,
        height_px=packed.height_px,
        rgb_bytes=packed.rgb_bytes,
        pose=frame.pose,
        intrinsics=frame.camera_intrinsics,
        camera_yaw_rad=frame.camera_extrinsics_in_robot.yaw_rad,
        camera_extrinsics_in_robot=frame.camera_extrinsics_in_robot,
        frontier_projections=project_frontier_ground_points(frame, candidates),
    )


def pack_rgb_image(image: RgbImage) -> VlmInputImage:
    """把内部 RGB 序列复制为可记录、可编码的连续字节。"""
    rows = image
    height = len(rows)
    width = len(rows[0]) if height else 0
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("RGB 必须是非空矩形图像")

    packed = bytearray(width * height * 3)
    offset = 0
    for row in rows:
        for pixel in row:
            if len(pixel) != 3:
                raise ValueError("RGB 像素必须包含三个通道")
            for channel in pixel:
                packed[offset] = _rgb_channel(channel)
                offset += 1
    return VlmInputImage(
        width_px=width,
        height_px=height,
        rgb_bytes=bytes(packed),
    )








def visible_frontier_candidates(
    images: Mapping[int, BufferedScanImage], candidates: Sequence[FrontierCandidate],
) -> Tuple[FrontierCandidate, ...]:
    """只保留至少一张固定图像中有地面投影及深度支持的候选。"""
    ordered = tuple(sorted(images.items()))
    return tuple(candidate for candidate in candidates if _ground_projection_choices(candidate, ordered))


def has_frontier_direction_in_view(
    image: BufferedScanImage, candidates: Sequence[FrontierCandidate],
) -> bool:
    """运动采样沿用可见水平方位触发；地面锚点失败不能取消该帧的目标检测。"""
    return any(_best_scan_frame(candidate, ((1, image),)) is not None for candidate in candidates)


def build_semantic_analysis_sheet(
    images: Mapping[int, BufferedScanImage], candidates: Sequence[FrontierCandidate],
) -> Tuple[VlmInputImage, Tuple[FrontierImageMarker, ...]]:
    """保留全部 RGB；V 放在紧凑页眉，F 页脚仅按实际标签行数分配。"""
    if not images:
        raise ValueError("联合分析至少需要一张画面")
    ordered = tuple(sorted(images.items()))
    first = ordered[0][1]
    width = min(TILE_MAX_WIDTH_PX, first.width_px)
    height = max(1, round(width * first.height_px / first.width_px))
    header, label_row_height = 28, 30
    label_columns = _semantic_label_columns(width, len(candidates))
    assignments = {view_id: [] for view_id, _ in ordered}
    markers = []
    for candidate in candidates:
        available = [(view_id, projection) for view_id, projection in _ground_projection_choices(candidate, ordered)
                     if len(assignments[view_id]) < 3 * label_columns]
        if not available:
            continue
        view_id, projection = available[0]
        label = f"F{len(markers) + 1}"
        assignments[view_id].append((label, projection.pixel_xy))
        markers.append(FrontierImageMarker(label, candidate.candidate_id, view_id, projection.pixel_xy))
    tiles, tile_heights = [], []
    for view_id, source in ordered:
        labels = sorted(assignments[view_id], key=lambda item: item[1][0])
        rows = math.ceil(len(labels) / label_columns) if labels else 0
        tile_height = header + height + rows * label_row_height
        tile = bytearray(bytes(SHEET_BACKGROUND_RGB) * (width * tile_height))
        _copy_tile(tile, width, _resize_rgb(source, width, height), width, height, 0, header)
        _draw_view_badge(tile, width, tile_height, view_id)
        used_columns = math.ceil(len(labels) / rows) if rows else 0
        placements = []
        for index, (label, (raw_x, raw_y)) in enumerate(labels):
            x = min(width - 1, int(round(raw_x * width / source.width_px)))
            y = header + min(height - 1, int(round(raw_y * height / source.height_px)))
            label_x = round((index // rows + 0.5) * width / used_columns)
            label_y = header + height + label_row_height // 2 + label_row_height * (index % rows)
            # 小锚点与单像素引线定位候选，大块编号留在原图之外。
            _draw_thin_line(tile, width, tile_height, (x, y), (label_x, label_y), MARKER_RGB)
            placements.append((label, x, y, label_x, label_y))
        # 全部引线先画，再画锚点和编号，避免后画的线覆盖文字。
        for label, x, y, label_x, label_y in placements:
            _fill_rectangle(tile, width, tile_height, x - 1, y - 1, 3, 3, MARKER_RGB)
            _draw_number_label(tile, width, tile_height, label_x, label_y, label)
        tiles.append(tile)
        tile_heights.append(tile_height)
    return _compose_sheet(tiles, width, tile_heights), tuple(markers)


def _ground_projection_choices(candidate, frames):
    """选择已有锚点更靠近画面中心的视角，不将地面点重新投影到未核对的图像。"""
    choices = []
    for view_id, frame in frames:
        for projection in frame.frontier_projections:
            if projection.world_xy != candidate.world_xy:
                continue
            u, v = projection.pixel_xy
            center_distance = (u / frame.width_px - 0.5) ** 2 + (v / frame.height_px - 0.5) ** 2
            choices.append((center_distance, view_id, projection))
    return tuple((view_id, projection) for _, view_id, projection in sorted(choices, key=lambda item: (item[0], item[1])))


def _semantic_label_columns(width: int, candidate_count: int) -> int:
    """计算每行容量，最多三行；放不下的候选改用其他视角或几何分。"""
    characters = 1 + len(str(max(1, candidate_count)))
    label_width = (4 * characters + 3) * MARKER_SCALE
    return max(0, width // (label_width + 6))


def _draw_thin_line(image, width, height, start, end, color) -> None:
    """整数栅格单像素引线；保持输入图像尺寸。"""
    x, y = start
    end_x, end_y = end
    dx, dy = abs(end_x - x), -abs(end_y - y)
    sx, sy = (1 if x < end_x else -1), (1 if y < end_y else -1)
    error = dx + dy
    while True:
        _set_pixel(image, width, height, x, y, color)
        if x == end_x and y == end_y:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x += sx
        if doubled <= dx:
            error += dx
            y += sy


def _compose_sheet(
    tiles: Sequence[bytearray],
    tile_width: int,
    tile_heights: Sequence[int],
) -> VlmInputImage:
    """等宽画面按行对齐；每行高度只取该行实际需要的最大高度。"""
    if not tiles:
        raise ValueError("扫描拼图至少需要一张图片")
    columns = min(SHEET_MAX_COLUMNS, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    sheet_width = columns * tile_width + (columns - 1) * TILE_GAP_PX
    row_heights = [max(tile_heights[start:start + columns]) for start in range(0, len(tiles), columns)]
    sheet_height = sum(row_heights) + (rows - 1) * TILE_GAP_PX
    sheet = bytearray(bytes(SHEET_BACKGROUND_RGB) * (sheet_width * sheet_height))
    for tile_index, tile in enumerate(tiles):
        tile_row = tile_index // columns
        tile_col = tile_index % columns
        _copy_tile(
            sheet,
            sheet_width,
            tile,
            tile_width,
            tile_heights[tile_index],
            tile_col * (tile_width + TILE_GAP_PX),
            sum(row_heights[:tile_row]) + tile_row * TILE_GAP_PX,
        )
    return VlmInputImage(sheet_width, sheet_height, bytes(sheet))


def _best_scan_frame(
    candidate: FrontierCandidate,
    frames: Sequence[Tuple[int, BufferedScanImage]],
) -> Optional[Tuple[int, float]]:
    """选择候选水平方位最靠近画面中心的扫描帧。"""
    choices = []
    for frame_index, frame in frames:
        delta_x = candidate.world_xy[0] - frame.pose.x_m
        delta_y = candidate.world_xy[1] - frame.pose.y_m
        world_heading = math.atan2(delta_y, delta_x)
        camera_heading = frame.pose.yaw_rad + frame.camera_yaw_rad
        relative_heading = _wrap_angle(world_heading - camera_heading)
        if abs(relative_heading) < math.pi / 2.0:
            source_x = (
                frame.intrinsics.cx
                - frame.intrinsics.fx * math.tan(relative_heading)
            )
            if 0.0 <= source_x < frame.width_px:
                choices.append(
                    (abs(relative_heading), frame_index, source_x)
                )
    if choices:
        _, frame_index, source_x = min(choices)
        return frame_index, source_x

    return None


def _resize_rgb(
    frame: BufferedScanImage,
    target_width: int,
    target_height: int,
) -> bytearray:
    """用最近邻缩放打包 RGB，返回按行连续的三通道字节缓冲区。"""
    result = bytearray(target_width * target_height * 3)
    source = frame.rgb_bytes
    for target_y in range(target_height):
        source_y = min(
            frame.height_px - 1,
            target_y * frame.height_px // target_height,
        )
        for target_x in range(target_width):
            source_x = min(
                frame.width_px - 1,
                target_x * frame.width_px // target_width,
            )
            source_offset = (source_y * frame.width_px + source_x) * 3
            target_offset = (target_y * target_width + target_x) * 3
            result[target_offset : target_offset + 3] = source[
                source_offset : source_offset + 3
            ]
    return result




def _draw_number_label(
    image: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    text: str,
) -> None:
    """在锚点附近绘制编号底色与字形，并把标签框限制在图像内。"""
    glyph_width = 3 * MARKER_SCALE
    glyph_gap = MARKER_SCALE
    text_width = len(text) * glyph_width + max(0, len(text) - 1) * glyph_gap
    box_width = text_width + 4 * MARKER_SCALE
    box_height = 5 * MARKER_SCALE + 4 * MARKER_SCALE
    left = max(0, min(width - box_width, center_x - box_width // 2))
    top = max(0, min(height - box_height, center_y - box_height // 2))
    _fill_rectangle(
        image,
        width,
        height,
        left,
        top,
        box_width,
        box_height,
        MARKER_RGB,
    )
    cursor_x = left + 2 * MARKER_SCALE
    glyph_top = top + 2 * MARKER_SCALE
    for digit in text:
        _draw_digit(
            image,
            width,
            height,
            cursor_x,
            glyph_top,
            digit,
            MARKER_TEXT_RGB,
        )
        cursor_x += glyph_width + glyph_gap


def _draw_view_badge(image: bytearray, width: int, height: int, view_id: int) -> None:
    """蓝底白字区分 V 与粉色 F；用平滑笔画绘制编号，不依赖外部字体包。"""
    label = f"V{view_id}"
    _fill_rectangle(image, width, height, 3, 2, 16 * len(label) + 12, 24, VIEW_BADGE_RGB)
    for index, character in enumerate(label):
        left, top = 10 + index * 16, 6
        segments = [
            ((left + x1 * 1.6, top + y1 * 1.6), (left + x2 * 1.6, top + y2 * 1.6))
            for stroke in _VIEW_STROKES[character]
            for (x1, y1), (x2, y2) in zip(stroke, stroke[1:])
        ]
        for y in range(top - 2, min(height, top + 19)):
            for x in range(left - 2, min(width, left + 12)):
                distance = min(_point_segment_distance(x + 0.5, y + 0.5, start, end)
                               for start, end in segments)
                coverage = max(0.0, min(1.0, 1.7 - distance))
                if coverage:
                    color = tuple(round(base + (255 - base) * coverage) for base in VIEW_BADGE_RGB)
                    _set_pixel(image, width, height, x, y, color)


def _point_segment_distance(x, y, start, end) -> float:
    """像素中心到字形笔画的距离，用于约一像素宽的抗锯齿边缘。"""
    dx, dy = end[0] - start[0], end[1] - start[1]
    ratio = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - start[0] - ratio * dx, y - start[1] - ratio * dy)


# 6×10 字形坐标上的连续笔画，最终按亚像素距离绘制，避免 3×5 方块字。
_VIEW_STROKES = {
    "V": (((0, 0), (3, 10), (6, 0)),),
    "0": (((3, 0), (1, 1), (0, 3), (0, 7), (1, 9), (3, 10), (5, 9), (6, 7), (6, 3), (5, 1), (3, 0)),),
    "1": (((1, 2), (3, 0), (3, 10)), ((0, 10), (6, 10))),
    "2": (((0, 2), (1, 0), (5, 0), (6, 2), (6, 4), (0, 10), (6, 10)),),
    "3": (((0, 1), (2, 0), (5, 0), (6, 2), (5, 4), (3, 5), (5, 5), (6, 7), (6, 8), (5, 10), (1, 10), (0, 9)),),
    "4": (((5, 10), (5, 0), (0, 7), (6, 7)),),
    "5": (((6, 0), (0, 0), (0, 5), (4, 5), (6, 7), (6, 8), (5, 10), (1, 10), (0, 9)),),
    "6": (((6, 1), (4, 0), (2, 1), (0, 4), (0, 8), (1, 10), (5, 10), (6, 8), (6, 6), (5, 5), (1, 5), (0, 6)),),
    "7": (((0, 0), (6, 0), (2, 10)),),
    "8": (((3, 0), (1, 0), (0, 2), (1, 4), (5, 6), (6, 8), (5, 10), (1, 10), (0, 8), (1, 6), (5, 4), (6, 2), (5, 0), (3, 0)),),
    "9": (((6, 4), (5, 5), (1, 5), (0, 4), (0, 2), (1, 0), (5, 0), (6, 2), (6, 6), (4, 9), (2, 10), (0, 9)),),
}


def _draw_digit(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    digit: str,
    color: Tuple[int, int, int],
) -> None:
    """按内置点阵放大绘制单个字符，直接修改 RGB 缓冲区。"""
    glyph = _DIGITS[digit]
    for row_index, row in enumerate(glyph):
        for col_index, enabled in enumerate(row):
            if enabled == "0":
                continue
            _fill_rectangle(
                image,
                width,
                height,
                left + col_index * MARKER_SCALE,
                top + row_index * MARKER_SCALE,
                MARKER_SCALE,
                MARKER_SCALE,
                color,
            )




def _fill_rectangle(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    rectangle_width: int,
    rectangle_height: int,
    color: Tuple[int, int, int],
) -> None:
    for y in range(max(0, top), min(height, top + rectangle_height)):
        for x in range(max(0, left), min(width, left + rectangle_width)):
            _set_pixel(image, width, height, x, y, color)


def _set_pixel(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    if not 0 <= x < width or not 0 <= y < height:
        return
    offset = (y * width + x) * 3
    image[offset : offset + 3] = bytes(color)


def _copy_tile(
    sheet: bytearray,
    sheet_width: int,
    tile: bytearray,
    tile_width: int,
    tile_height: int,
    target_x: int,
    target_y: int,
) -> None:
    """将一块连续 RGB 图按行复制到拼图指定位置；目标区域由布局调用方预留。"""
    for row in range(tile_height):
        source_start = row * tile_width * 3
        target_start = ((target_y + row) * sheet_width + target_x) * 3
        sheet[target_start : target_start + tile_width * 3] = tile[
            source_start : source_start + tile_width * 3
        ]


def _rgb_channel(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("RGB 通道必须是 0 到 255 的整数")
    try:
        channel = int(value)
    except (TypeError, ValueError):
        raise ValueError("RGB 通道必须是 0 到 255 的整数") from None
    if channel != value or not 0 <= channel <= 255:
        raise ValueError("RGB 通道必须是 0 到 255 的整数")
    return channel


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


_DIGITS = {
    "V": ("101", "101", "101", "101", "010"),
    "F": ("111", "100", "110", "100", "100"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


__all__ = [
    "BufferedScanImage",
    "FrontierImageMarker",
    "buffer_scan_image",
    "build_semantic_analysis_sheet",
    "has_frontier_direction_in_view",
    "pack_rgb_image",
    "visible_frontier_candidates",
]
