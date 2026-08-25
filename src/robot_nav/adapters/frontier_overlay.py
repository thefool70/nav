"""把一轮扫描 RGB 与 Frontier 候选组成可供 VLM 评分的拼图。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from ..core.models import (
    CameraIntrinsics,
    FrontierCandidate,
    NavigationFrame,
    Pose2D,
    RgbImage,
)
from .perception import VlmInputImage


TILE_MAX_WIDTH_PX = 320
SHEET_MAX_COLUMNS = 4
TILE_GAP_PX = 6
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


@dataclass(frozen=True)
class FrontierImageMarker:
    """拼图中数字标记与真实 Frontier ID 的对应。"""

    label: str
    candidate_id: str


def buffer_scan_image(frame: NavigationFrame) -> BufferedScanImage:
    """只保留批量评分所需的 RGB、位姿和水平投影参数。"""
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


def build_frontier_score_sheet(
    scan_images: Mapping[int, BufferedScanImage],
    candidates: Sequence[FrontierCandidate],
) -> Tuple[VlmInputImage, Tuple[FrontierImageMarker, ...]]:
    """每个候选只标在最接近其方位的扫描帧上，再拼成一张图。"""
    if not scan_images:
        raise ValueError("没有可用的扫描 RGB")
    if not candidates:
        raise ValueError("没有待评分的 Frontier")

    ordered_frames = tuple(sorted(scan_images.items()))
    assignments: Dict[int, list] = {index: [] for index, _ in ordered_frames}
    markers = []
    for marker_index, candidate in enumerate(candidates, start=1):
        frame_index, source_x = _best_scan_frame(candidate, ordered_frames)
        label = str(marker_index)
        assignments[frame_index].append((label, source_x))
        markers.append(FrontierImageMarker(label, candidate.candidate_id))

    used_frames = tuple(
        (index, frame)
        for index, frame in ordered_frames
        if assignments[index]
    )
    first_frame = used_frames[0][1]
    tile_width = min(TILE_MAX_WIDTH_PX, first_frame.width_px)
    tile_height = max(
        1,
        round(tile_width * first_frame.height_px / first_frame.width_px),
    )
    tiles = []
    for frame_index, frame in used_frames:
        tile = _resize_rgb(frame, tile_width, tile_height)
        scaled_markers = tuple(
            (
                label,
                source_x * tile_width / frame.width_px,
            )
            for label, source_x in assignments[frame_index]
        )
        _draw_markers(tile, tile_width, tile_height, scaled_markers)
        tiles.append(tile)

    columns = min(SHEET_MAX_COLUMNS, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    sheet_width = columns * tile_width + (columns - 1) * TILE_GAP_PX
    sheet_height = rows * tile_height + (rows - 1) * TILE_GAP_PX
    sheet = bytearray(sheet_width * sheet_height * 3)
    for tile_index, tile in enumerate(tiles):
        tile_row = tile_index // columns
        tile_col = tile_index % columns
        _copy_tile(
            sheet,
            sheet_width,
            tile,
            tile_width,
            tile_height,
            tile_col * (tile_width + TILE_GAP_PX),
            tile_row * (tile_height + TILE_GAP_PX),
        )
    return (
        VlmInputImage(sheet_width, sheet_height, bytes(sheet)),
        tuple(markers),
    )


def _best_scan_frame(
    candidate: FrontierCandidate,
    frames: Sequence[Tuple[int, BufferedScanImage]],
) -> Tuple[int, float]:
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

    fallback = []
    for frame_index, frame in frames:
        delta_x = candidate.world_xy[0] - frame.pose.x_m
        delta_y = candidate.world_xy[1] - frame.pose.y_m
        world_heading = math.atan2(delta_y, delta_x)
        camera_heading = frame.pose.yaw_rad + frame.camera_yaw_rad
        relative_heading = _wrap_angle(world_heading - camera_heading)
        edge_x = 12.0 if relative_heading > 0.0 else frame.width_px - 13.0
        fallback.append((abs(relative_heading), frame_index, edge_x))
    _, frame_index, source_x = min(fallback)
    return frame_index, source_x


def _resize_rgb(
    frame: BufferedScanImage,
    target_width: int,
    target_height: int,
) -> bytearray:
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


def _draw_markers(
    image: bytearray,
    width: int,
    height: int,
    markers: Sequence[Tuple[str, float]],
) -> None:
    """数字标记放在画面下部，x 坐标仍对应 Frontier 的水平方位。"""
    placed = []
    for label, raw_x in sorted(markers, key=lambda item: item[1]):
        x = int(round(max(8.0, min(width - 9.0, raw_x))))
        occupied_layers = {
            layer
            for previous_x, layer in placed
            if abs(previous_x - x) < 34
        }
        layer = next(
            (candidate for candidate in range(3) if candidate not in occupied_layers),
            len(placed) % 3,
        )
        y = int(round(height * (0.62 + 0.13 * layer)))
        _draw_vertical_line(image, width, height, x, y, height - 1, MARKER_RGB)
        _draw_number_label(image, width, height, x, y, label)
        placed.append((x, layer))


def _draw_number_label(
    image: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    text: str,
) -> None:
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


def _draw_digit(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    digit: str,
    color: Tuple[int, int, int],
) -> None:
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


def _draw_vertical_line(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    start_y: int,
    end_y: int,
    color: Tuple[int, int, int],
) -> None:
    for y in range(max(0, start_y), min(height, end_y + 1)):
        _set_pixel(image, width, height, x, y, color)
        _set_pixel(image, width, height, x + 1, y, color)


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
    "build_frontier_score_sheet",
    "pack_rgb_image",
]
