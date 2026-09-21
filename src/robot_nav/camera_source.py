"""为导航和标定装配同一种相机来源。远程服务自行配置 USB 序列号。"""

from .adapters.realsense.d435i_camera import D435iCamera, D435iCalibrationCamera


def camera_factory(args, *, calibration=False):
    """返回供 Adapter 调用的相机构造函数；远程源仅支持导航所需的 RGB-D，不提供标定 IMU。"""
    if args.camera_source == "local":
        return D435iCalibrationCamera if calibration else D435iCamera
    if calibration:
        raise ValueError("ZMQ 发布器不提供 IMU，不能远程自动标定；请使用已有有效外参或本地标定")
    if args.camera_serial:
        raise ValueError("远程相机序列号请在相机服务启动时指定")
    from .adapters.hermes.remote.camera_client import RemoteD435iCamera

    def remote_camera(_config):
        return RemoteD435iCamera(args.camera_endpoint, topic=args.camera_topic,
            timeout_s=args.camera_timeout_s)
    return remote_camera
