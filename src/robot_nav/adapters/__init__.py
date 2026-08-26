"""外部系统适配层：连接核心算法与底盘等外部设备。"""

from .chassis import ChassisInterface, MotionInterruptedError
from .openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .perception import (
    ContinuousTargetObserver,
    TargetObserver,
)
from .random_observer import RandomScoreTargetObserver

__all__ = [
    "ChassisInterface",
    "ContinuousTargetObserver",
    "MotionInterruptedError",
    "OpenAIApiFormat",
    "OpenAICompatibleConfig",
    "OpenAICompatibleTargetObserver",
    "RandomScoreTargetObserver",
    "TargetObserver",
]
