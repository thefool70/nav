"""D435i 本机采集：导航读取 RGB-D，标定时独占切换 IMU。"""

from dataclasses import dataclass, replace

from .l515_camera import L515Camera, L515Config


@dataclass(frozen=True)
class D435iConfig(L515Config):
    """彩色和深度均为 640×480、30 FPS；可显式选择设备公布的其他 profile。"""

    depth_width: int = 640
    depth_height: int = 480


class D435iCamera(L515Camera):
    """使用共用 SDK 采集流程，拒绝误连到其他型号的相机。"""

    device_label = "D435i"

    @property
    def serial_number(self):
        device = self._pipeline.get_active_profile().get_device()
        return device.get_info(self._rs.camera_info.serial_number)

    def _start(self) -> None:
        super()._start()
        device = self._pipeline.get_active_profile().get_device()
        name = device.get_info(self._rs.camera_info.name)
        if "D435I" not in name.upper():
            raise RuntimeError(f"需要 D435i，实际连接的相机为 {name}")


class D435iCalibrationCamera:
    """按需打开本机 RGB-D；IMU 采集前释放图像流，随后按同一序列号重开。"""

    def __init__(self, config):
        self.config = config
        self._camera = None

    def _open_rgbd(self):
        if self._camera is None:
            self._camera = D435iCamera(self.config)
            self.config = replace(self.config, serial_number=self._camera.serial_number)
        return self._camera

    def capture(self):
        return self._open_rgbd().capture()

    def read_imu_samples(self):
        from .d435i_motion import read_d435i_imu_samples

        camera = self._open_rgbd()
        serial = camera.serial_number
        rotation = camera.depth_to_color_rotation
        self.close()
        samples = read_d435i_imu_samples(serial)
        samples["depth_to_color_rotation"] = rotation
        return samples

    def close(self):
        if self._camera is not None:
            self._camera.close()
            self._camera = None
