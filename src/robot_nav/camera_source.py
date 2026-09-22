"""为导航装配本地或远程相机来源。远程服务自行配置 USB 序列号。"""

from .adapters.realsense.d435i_camera import D435iCamera


def camera_factory(args):
    """返回供 Adapter 调用的本地或远程 RGB-D 相机构造函数。"""
    if args.camera_source == "local":
        return D435iCamera
    if args.camera_serial:
        raise ValueError("远程相机序列号请在相机服务启动时指定")
    from .adapters.hermes.remote.camera_client import RemoteD435iCamera

    def remote_camera(_config):
        return RemoteD435iCamera(args.camera_endpoint, topic=args.camera_topic,
            timeout_s=args.camera_timeout_s)
    return remote_camera
