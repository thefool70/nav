"""语义感知：与搜索核心平行的独立模块。

对外只提供 :class:`SemanticPerception`，集中承担：

- 取帧选择与扫描批处理（何时拍、拍哪一帧、如何拼成整轮）。
- 异步视觉队列与结果投递（入队、FIFO 分析、完成后交回运行层）。
- 语义判定与方向打分（目标视图、Frontier 分数、场景判断）。
- 历史物体检测、分割与定位编排。

模块内部按职责分开，互不越界：

- :mod:`~robot_nav.perception.semantic_queue`：感知策略与队列状态机。
- :mod:`~robot_nav.perception.object_localizer`：历史定位的模型组合与回退策略。
- :mod:`~robot_nav.perception.snapshot_store`：快照读写与深度编解码。
- :mod:`~robot_nav.perception.analyzer`：HTTP 与模型进程边界。

感知模块不直接控制底盘，也不修改搜索状态；它只运行层交回的
「采集确认、分析结果、进度」以及搜索核心的显式暂停/继续请求。
"""

from .semantic_queue import (
    PerceptionPause,
    SemanticPerception,
    SemanticPerceptionConfig,
)

__all__ = [
    "PerceptionPause",
    "SemanticPerception",
    "SemanticPerceptionConfig",
]
