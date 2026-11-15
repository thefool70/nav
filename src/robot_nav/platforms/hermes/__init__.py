"""SLAMTEC Hermes 与外接 RealSense D435i 真实设备环境。"""

from robot_nav.platforms.hermes.adapter import HermesAdapter, HermesConfig
from robot_nav.platforms.hermes.camera_extrinsics import (
    DEFAULT_CAMERA_EXTRINSICS_PATH,
    load_camera_extrinsics,
)
from robot_nav.platforms.hermes.rest_client import (
    HermesExploreMap,
    HermesRestClient,
    HermesRobotHealth,
    HermesSlamState,
)

__all__ = [
    "DEFAULT_CAMERA_EXTRINSICS_PATH",
    "HermesAdapter",
    "HermesConfig",
    "HermesExploreMap",
    "HermesRestClient",
    "HermesRobotHealth",
    "HermesSlamState",
    "load_camera_extrinsics",
]
