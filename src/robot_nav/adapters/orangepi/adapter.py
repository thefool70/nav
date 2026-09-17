"""开发机上的 Hermes + 远程 D435i 适配层；香橙派不执行导航或动作控制。"""

from dataclasses import dataclass, field

from ..realsense.d435i_camera import D435iConfig
from ..slamtec_l515.adapter import SlamtecL515Adapter, SlamtecL515Config
from .camera_client import RemoteD435iCamera


@dataclass(frozen=True)
class OrangePiConfig(SlamtecL515Config):
    """两个本地端口分别由 SSH 转发到 Hermes REST 和香橙派相机服务。"""

    base_url: str = "http://127.0.0.1:11448"
    camera: D435iConfig = field(default_factory=D435iConfig)
    camera_url: str = "http://127.0.0.1:18765"
    camera_request_timeout_s: float = 5.0
    camera_max_roundtrip_s: float = 3.0


class OrangePiAdapter(SlamtecL515Adapter):
    """复用开发机上的 FOV 缓存、完整地图、同步 Action、路径检查和运动帧回调。"""

    def __init__(self, config: OrangePiConfig, **callbacks):
        if config.camera is None:
            raise ValueError("OrangePiAdapter 必须启用远程 D435i")
        super().__init__(
            config, **callbacks,
            camera_factory=lambda _: RemoteD435iCamera(
                config.camera_url, timeout_s=config.camera_request_timeout_s,
                max_roundtrip_s=config.camera_max_roundtrip_s,
            ),
        )
