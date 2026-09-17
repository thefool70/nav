"""无线采集入口；外参求解和 Hermes 运动复用直连版本。"""

from ..slamtec_l515.d435i_calibration import calibrate_hermes_d435i
from .camera_client import RemoteD435iCamera


def calibrate_orangepi(
    *, base_url, camera_url, output_path,
    turn_angle_deg=30.0, drive_distance_m=0.20, action_timeout_s=120.0,
    minimum_localization_quality=1, camera_request_timeout_s=5.0,
    camera_max_roundtrip_s=3.0, progress=print,
):
    camera = RemoteD435iCamera(camera_url, timeout_s=camera_request_timeout_s,
                              max_roundtrip_s=camera_max_roundtrip_s)
    try:
        return calibrate_hermes_d435i(
            base_url=base_url, camera=camera, output_path=output_path,
            turn_angle_deg=turn_angle_deg, drive_distance_m=drive_distance_m,
            action_timeout_s=action_timeout_s,
            minimum_localization_quality=minimum_localization_quality, progress=progress,
        )
    finally:
        camera.close()
