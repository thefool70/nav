"""外部系统适配层：连接核心算法与底盘等外部设备。"""

from .chassis import ChassisInterface
from .openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .perception import TargetObserver

__all__ = [
    "ChassisInterface",
    "OpenAIApiFormat",
    "OpenAICompatibleConfig",
    "OpenAICompatibleTargetObserver",
    "TargetObserver",
]
