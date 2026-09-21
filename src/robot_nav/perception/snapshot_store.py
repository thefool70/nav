"""快照存储：把一帧或多帧写入独立任务目录，并在需要时恢复历史 RGB-D。

本模块只负责磁盘格式与编解码，不关心队列顺序、暂停策略或模型调用。
图片与元数据写入独立运行目录；深度以二进制保存，旧快照仍可读取。
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..core.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    FrontierCandidate,
    NavigationFrame,
    ObservationView,
    Pose2D,
    SearchMode,
    TargetClue,
    TargetSearchGoal,
)
from ..core.frontier_projection import FrontierImageProjection
from ..adapters.frontier_overlay import BufferedScanImage
from ..adapters.snapshot_depth import (
    decode_depth,
)

__all__ = [
    "CapturedView",
    "read_clue_frame",
    "read_snapshot",
    "retain_clue_depth",
    "view_trace",
    "write_snapshot",
]


class CapturedView:
    """一张已采集画面及其覆盖；在队列模块中作为不可变数据传递。"""

    __slots__ = ("image", "coverage", "map_frame_id", "depth_gzip")

    def __init__(self, image, coverage, map_frame_id: str, depth_gzip: Optional[bytes]) -> None:
        self.image = image
        self.coverage = coverage
        self.map_frame_id = map_frame_id
        self.depth_gzip = depth_gzip


def write_snapshot(folder: Path, views, candidates, goal, source: str, job_id: int) -> None:
    """把整轮画面、候选与目标写入任务目录，供后台分析线程读取。"""
    folder.mkdir()
    metadata = {"source": source, "goal": asdict(goal), "views": [], "candidates": []}
    for index, view in enumerate(views, 1):
        image = view.image
        with gzip.open(folder / f"view-{index}.rgb.gz", "wb", compresslevel=1) as stream:
            stream.write(image.rgb_bytes)
        if view.depth_gzip is not None:
            (folder / f"view-{index}.depth.pending.f64.gz").write_bytes(view.depth_gzip)
        metadata["views"].append({
            "map_frame_id": view.map_frame_id, "coverage": asdict(view.coverage),
            "width_px": image.width_px, "height_px": image.height_px,
            "intrinsics": asdict(image.intrinsics), "camera_yaw_rad": image.camera_yaw_rad,
            "camera_extrinsics_in_robot": asdict(image.camera_extrinsics_in_robot),
            "frontier_projections": [asdict(projection) for projection in image.frontier_projections],
            "depth": {"captured": view.depth_gzip is not None, "encoding": "gzip-float64-le-v1", "unit": "m"},
        })
    for index, candidate in enumerate(candidates, 1):
        item = asdict(replace(candidate, candidate_id=f"snapshot:{job_id}:{index}", frontier_cells=()))
        item["source_region_id"] = candidate.candidate_id
        metadata["candidates"].append(item)
    (folder / "snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def read_snapshot(folder: Path):
    """读回任务目录中的图像、覆盖、候选与目标。"""
    metadata = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    goal = TargetSearchGoal(metadata["goal"]["target_text"], SearchMode(metadata["goal"]["search_mode"]))
    images, views, candidates, region_ids = {}, [], [], []
    for index, item in enumerate(metadata["views"], 1):
        raw = item["coverage"]
        raw["pose"] = Pose2D(**raw["pose"])
        for name in ("visible_world_xy", "map_visible_world_xy"):
            raw[name] = tuple(tuple(point) for point in raw[name])
        if raw["camera_world_xy"] is not None:
            raw["camera_world_xy"] = tuple(raw["camera_world_xy"])
        coverage = ObservationView(**raw)
        with gzip.open(folder / f"view-{index}.rgb.gz", "rb") as stream:
            rgb = stream.read()
        images[index] = BufferedScanImage(
            item["width_px"], item["height_px"], rgb, coverage.pose,
            CameraIntrinsics(**item["intrinsics"]), item["camera_yaw_rad"],
            CameraExtrinsics(**item["camera_extrinsics_in_robot"]),
            tuple(FrontierImageProjection(
                tuple(projection["world_xy"]), tuple(projection["pixel_xy"]),
                projection["camera_depth_m"], projection["observed_depth_m"],
            ) for projection in item["frontier_projections"]),
        )
        views.append((item["map_frame_id"], coverage))
    for item in metadata["candidates"]:
        region_ids.append(item.pop("source_region_id"))
        item["world_xy"] = tuple(item["world_xy"])
        item["frontier_cells"] = tuple(tuple(cell) for cell in item["frontier_cells"])
        if item["deferred_order"] is not None:
            item["deferred_order"] = tuple(item["deferred_order"])
        candidates.append(FrontierCandidate(**item))
    return images, tuple(views), tuple(candidates), goal, tuple(region_ids), metadata["source"]


def retain_clue_depth(
    folder: Path, view_count: int, target_view_ids: Optional[Tuple[int, ...]],
) -> Mapping[str, Any]:
    """匹配画面转为正式深度文件；明确未匹配的删除临时文件，检测失败仍保留待判定数据。"""
    if target_view_ids is None:
        return {"status": "pending_detection"}
    retained, missing, errors = [], [], []
    for index in range(1, view_count + 1):
        pending = folder / f"view-{index}.depth.pending.f64.gz"
        retained_path = folder / f"view-{index}.depth.f64.gz"
        try:
            if index in target_view_ids:
                if pending.is_file():
                    pending.replace(retained_path)
                if retained_path.is_file():
                    retained.append(index)
                else:
                    missing.append(index)
            else:
                pending.unlink(missing_ok=True)
        except OSError as exc:
            # 磁盘失败不推翻有效目标检测；临时数据保留，供调用方识别深度是否可用。
            errors.append({"view_id": index, "reason": str(exc)})
    return {
        "status": "selection_failed" if errors else "selected",
        "retained_view_ids": retained, "missing_view_ids": missing, "errors": errors,
    }


def read_clue_frame(directory: Path, clue: TargetClue, current: NavigationFrame) -> NavigationFrame:
    """恢复拍摄时的 RGB-D、位姿和标定，保留当前同坐标系地图供障碍射线查询。"""
    if clue.job_id is None or clue.view_id is None:
        raise ValueError("线索缺少快照编号")
    folder = directory / f"job-{clue.job_id:06d}"
    metadata = json.loads((folder / "snapshot.json").read_text())
    item = metadata["views"][clue.view_id - 1]
    if item["map_frame_id"] != current.obstacle_map.frame_id:
        raise ValueError("历史图像与当前地图坐标系不同")
    width, height = item["width_px"], item["height_px"]
    with gzip.open(folder / f"view-{clue.view_id}.rgb.gz", "rb") as stream:
        raw_rgb = stream.read()
    if len(raw_rgb) != width * height * 3:
        raise ValueError("历史 RGB 数据长度错误")
    depth = None
    depth_path = folder / f"view-{clue.view_id}.depth.f64.gz"
    legacy_depth_path = folder / f"view-{clue.view_id}.depth.json.gz"
    if depth_path.exists():
        depth = decode_depth(depth_path.read_bytes(), width, height)
    elif legacy_depth_path.exists():
        # 已保存的旧快照仍可读取；新任务仅写二进制深度。
        depth_path = legacy_depth_path
        with gzip.open(depth_path, "rt", encoding="utf-8") as stream:
            raw_depth = json.load(stream)
        if raw_depth["unit"] != "m" or raw_depth["width_px"] != width or raw_depth["height_px"] != height:
            raise ValueError("历史深度尺寸或单位不匹配")
        values = raw_depth["values"]
        if len(values) != height or any(len(row) != width for row in values):
            raise ValueError("历史深度数据长度错误")
        depth = tuple(tuple(float(value) if value is not None else None for value in row) for row in values)
    rgb = tuple(tuple(tuple(raw_rgb[(row * width + col) * 3:(row * width + col + 1) * 3]) for col in range(width)) for row in range(height))
    return replace(
        current, rgb=rgb, depth=depth, pose=Pose2D(**item["coverage"]["pose"]),
        timestamp_s=item["coverage"]["timestamp_s"], camera_intrinsics=CameraIntrinsics(**item["intrinsics"]),
        camera_extrinsics_in_robot=CameraExtrinsics(**item["camera_extrinsics_in_robot"]),
    )


def view_trace(views):
    """画面编号、拍摄时间与机器人位姿用于关联 VLM 请求，不携带覆盖点大数组。"""
    return tuple({"view_id": index, "map_frame_id": map_id, "timestamp_s": view.timestamp_s,
                  "pose": asdict(view.pose), "heading_world_rad": view.camera_heading_world_rad}
                 for index, (map_id, view) in enumerate(views, 1))


def write_localization_input(folder, image, frame, context):
    """保存历史物体定位的输入画面、地图和坐标信息，供事后复盘。"""
    with gzip.open(folder / "input.rgb.gz", "wb") as stream:
        stream.write(image.rgb_bytes)
    if frame.depth is not None:
        depth = [[float(value) if value is not None and math.isfinite(float(value)) else None for value in row]
                 for row in frame.depth]
        with gzip.open(folder / "input.depth.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"unit": "m", "width_px": image.width_px, "height_px": image.height_px, "values": depth}, stream)
    grid = frame.navigation_map if frame.navigation_map is not None else frame.obstacle_map
    with gzip.open(folder / "input.obstacle_map.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"frame_id": grid.frame_id, "origin": asdict(grid.origin),
                   "resolution_m": grid.resolution_m, "occupancy": grid.occupancy}, stream)
    metadata = {
        "width_px": image.width_px, "height_px": image.height_px,
        "timestamp_s": frame.timestamp_s, "pose": asdict(frame.pose),
        "intrinsics": asdict(frame.camera_intrinsics) if frame.camera_intrinsics is not None else None,
        "extrinsics": asdict(frame.camera_extrinsics_in_robot), "depth_available": frame.depth is not None,
        "map_frame_id": grid.frame_id, **context,
        "localization_map": "full_navigation" if frame.navigation_map is not None else "exploration",
    }
    (folder / "frame.json").write_text(json.dumps(metadata, ensure_ascii=False))


def write_localization_result(folder, result, details):
    """写入定位结果并返回诊断记录，调用者负责发送事件。"""
    record = {**asdict(result), **details}
    record["visibility"] = result.visibility.value
    record["vlm_confirmation"] = result.vlm_confirmation.value
    (folder / "localization.json").write_text(json.dumps(record, ensure_ascii=False))
    return record
