"""目标线索分派：物体历史定位和场景返回分别由各行为模块负责。"""

from .models import SearchMode


def continue_target_clue(frame, goal, state, localization=None):
    if goal.search_mode is SearchMode.OBJECT:
        from .object_approach import navigate_object_approach
        return navigate_object_approach(frame, state, localization)
    from .scene_target import continue_scene_target
    return continue_scene_target(frame, state)
