"""D435i 本机 RGB-D 采集。"""

from dataclasses import dataclass

from .rgbd_camera import RgbdCamera, RgbdCameraConfig


@dataclass(frozen=True)
class D435iConfig(RgbdCameraConfig):
    """彩色和深度均为 640×480、30 FPS；可显式选择设备公布的其他 profile。"""

    depth_width: int = 640
    depth_height: int = 480


class D435iCamera(RgbdCamera):
    """使用共用 SDK 采集流程，拒绝误连到其他型号的相机。"""

    device_label = "D435i"


    def _start(self) -> None:
        super()._start()
        device = self._pipeline.get_active_profile().get_device()
        name = device.get_info(self._rs.camera_info.name)
        if "D435I" not in name.upper():
            raise RuntimeError(f"需要 D435i，实际连接的相机为 {name}")
