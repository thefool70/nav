"""动作位姿转换，执行与诊断共用同一计算。"""

from typing import Optional

from .geometry import world_point_to_robot
from .models import ActionKind, NavigationAction, Pose2D, RelativePoseCommand
from .scan import shortest_turn_to_heading


def action_command(
    action: Optional[NavigationAction], pose: Pose2D,
) -> Optional[RelativePoseCommand]:
    """将世界系目标位姿转换为以输入帧为基准的相对位姿（米、弧度）。"""
    if action is None:
        return None
    if action.action is not ActionKind.MOVE_TO_POSE:
        return action.command
    destination = action.destination
    if destination is None:
        raise ValueError("世界系移动缺少目标位姿")
    forward, left = world_point_to_robot((destination.x_m, destination.y_m), pose)
    return RelativePoseCommand(forward_m=forward, left_m=left,
        yaw_rad=shortest_turn_to_heading(pose.yaw_rad, destination.yaw_rad))
