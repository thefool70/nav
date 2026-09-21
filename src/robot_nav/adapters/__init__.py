"""外部系统适配层：连接核心算法与底盘、相机等外部设备。"""

from .chassis import (
    ChassisInterface,
    KnownSpaceChassisInterface,
    MotionPathUnknownError,
    MotionStalledError,
    RecoverableMotionError,
)
from .openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .random_observer import RandomScoreTargetObserver

__all__ = [
    "ChassisInterface",
    "KnownSpaceChassisInterface",
    "MotionPathUnknownError",
    "MotionStalledError",
    "OpenAIApiFormat",
    "OpenAICompatibleConfig",
    "OpenAICompatibleTargetObserver",
    "RandomScoreTargetObserver",
    "RecoverableMotionError",
]
