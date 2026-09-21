"""可被不同底盘 Adapter 复用的 Intel RealSense 设备边界。"""

from .d435i_camera import D435iCalibrationCamera, D435iCamera, D435iConfig
from .rgbd_camera import RgbdCamera, RgbdCameraConfig, RgbdCapture

# D435iCapture 是导航与地图刷新共用的对齐 RGB-D 帧类型。
D435iCapture = RgbdCapture

__all__ = [
    "D435iCalibrationCamera",
    "D435iCamera",
    "D435iCapture",
    "D435iConfig",
    "RgbdCamera",
    "RgbdCameraConfig",
    "RgbdCapture",
]
