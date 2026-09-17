"""在物体周围的可达自由格中选择停靠点；探索图与完整导航图分别使用。"""

from __future__ import annotations

import math

from .frontier import reachable_free_distances
from .geometry import grid_cell_center_to_world, world_to_nearest_grid_cell
from .models import Pose2D


PREFERRED_STANDOFF_M = 0.75
MIN_STANDOFF_M = 0.60
MAX_STANDOFF_M = 2.0
REUSE_CURRENT_DISTANCE_M = 0.90
FAILED_STANDOFF_EXCLUSION_M = 0.20


def plan_object_standoff(frame, target_xy, tried_positions):
    """搜索目标周围完整圆域，返回停靠位姿和可直接写入日志的选点诊断。"""
    grid = frame.navigation_map if frame.navigation_map is not None else frame.obstacle_map
    clearance = frame.navigation_clearance_m if frame.navigation_map is not None else 0.0
    reachable = reachable_free_distances(grid, frame.pose, clearance_m=clearance)
    details = {
        "standoff_map": "full_navigation" if frame.navigation_map is not None else "exploration",
        "standoff_clearance_m": clearance,
        "standoff_search_radius_m": MAX_STANDOFF_M,
        "standoff_reachable_cells": len(reachable),
        "standoff_unknown_cells": 0,
        "standoff_occupied_cells": 0,
        "standoff_unreachable_cells": 0,
        "standoff_excluded_cells": 0,
        "standoff_candidate_count": 0,
    }
    robot_xy = (frame.pose.x_m, frame.pose.y_m)
    current_distance = math.dist(robot_xy, target_xy)
    current_cell = world_to_nearest_grid_cell(robot_xy, grid)
    if (
        MIN_STANDOFF_M <= current_distance <= REUSE_CURRENT_DISTANCE_M
        and current_cell in reachable
        and not _near_tried_position(robot_xy, tried_positions)
    ):
        details.update(standoff_candidate_count=1, standoff_distance_m=current_distance,
                       standoff_path_distance_m=0.0)
        return _facing_target(robot_xy, target_xy, frame.camera_extrinsics_in_robot.yaw_rad), details

    heading = math.atan2(robot_xy[1] - target_xy[1], robot_xy[0] - target_xy[0])
    preferred = (target_xy[0] + PREFERRED_STANDOFF_M * math.cos(heading),
                 target_xy[1] + PREFERRED_STANDOFF_M * math.sin(heading))
    center_row, center_col = world_to_nearest_grid_cell(target_xy, grid)
    steps = int(math.ceil(MAX_STANDOFF_M / grid.resolution_m)) + 1
    height = len(grid.occupancy)
    width = len(grid.occupancy[0]) if height else 0
    best = None
    for row in range(max(0, center_row - steps), min(height, center_row + steps + 1)):
        for col in range(max(0, center_col - steps), min(width, center_col + steps + 1)):
            xy = grid_cell_center_to_world(row, col, grid)
            distance = math.dist(xy, target_xy)
            if not MIN_STANDOFF_M <= distance <= MAX_STANDOFF_M:
                continue
            value = grid.occupancy[row][col]
            if value is None:
                details["standoff_unknown_cells"] += 1
            elif value > 0.5:
                details["standoff_occupied_cells"] += 1
            elif (row, col) not in reachable:
                # 包括被净空膨胀排除，以及与机器人所在自由区不连通的格子。
                details["standoff_unreachable_cells"] += 1
            elif _near_tried_position(xy, tried_positions):
                details["standoff_excluded_cells"] += 1
            else:
                details["standoff_candidate_count"] += 1
                path_distance = reachable[(row, col)] * grid.resolution_m
                rank = (math.dist(xy, preferred), path_distance, row, col)
                if best is None or rank < best[0]:
                    best = (rank, xy, distance, path_distance)
    if best is None:
        return None, details
    _, position, distance, path_distance = best
    details.update(standoff_distance_m=distance, standoff_path_distance_m=path_distance)
    return _facing_target(position, target_xy, frame.camera_extrinsics_in_robot.yaw_rad), details


def _near_tried_position(position, tried_positions):
    return any(math.dist(position, old) < FAILED_STANDOFF_EXCLUSION_M for old in tried_positions)


def _facing_target(position, target_xy, camera_yaw):
    yaw = math.atan2(target_xy[1] - position[1], target_xy[0] - position[0]) - camera_yaw
    return Pose2D(position[0], position[1], (yaw + math.pi) % (2.0 * math.pi) - math.pi)
