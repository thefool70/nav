"""用 D435i IMU/RGB-D 与 Hermes 停车位姿求解外参；采集源由调用者提供。"""

from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import tempfile
import time

from ...core.models import CameraExtrinsics
from ..realsense.calibration.floor import estimate_floor, mount_angles_from_up
from ..realsense.calibration.planar import solve_planar_extrinsic
from ..realsense.calibration.visual_motion import CapturedFrame, estimate_motion_pairs, require_visual_features
from .adapter import HermesAdapter, HermesConfig
from .calibration import HermesCalibrationMotion
from .mount_config import save_camera_extrinsics


@dataclass(frozen=True)
class D435iCalibrationResult:
    output_path: Path
    capture_directory: Path
    extrinsic_residual_m: float


def calibrate_hermes_d435i(
    *, base_url, camera, output_path,
    turn_angle_deg=30.0, drive_distance_m=0.20, action_timeout_s=120.0,
    minimum_localization_quality=1, progress=print,
):
    """先校验静止/地面/纹理，再转向和平移；全部求解通过后才替换正式外参文件。"""
    if not math.isfinite(turn_angle_deg) or not 10.0 <= turn_angle_deg <= 45.0:
        raise ValueError("标定转角必须在 10–45° 内")
    if not math.isfinite(drive_distance_m) or not 0.10 <= drive_distance_m <= 0.50:
        raise ValueError("标定平移必须在 0.10–0.50 m 内")
    np = importlib.import_module("numpy")
    cv2 = importlib.import_module("cv2")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="calibration-", dir=output_path.parent))
    progress(f"标定采集记录：{folder}")
    config = HermesConfig(
        base_url=base_url, camera=None, action_timeout_s=action_timeout_s,
        minimum_localization_quality=minimum_localization_quality,
        action_arrival_position_m=0.03, yaw_tolerance_rad=math.radians(2.0),
    )
    try:
        with HermesAdapter(config, on_action_progress=progress) as chassis:
            controller = HermesCalibrationMotion(chassis)
            progress("[1/4] 静止采集 D435i IMU……")
            before_imu = controller.read_pose()
            samples = camera.read_imu_samples()
            after_imu = controller.read_pose()
            _require_same_pose(before_imu, after_imu)
            _write_json(folder / "imu.json", samples)
            up_hint, imu_diagnostics = _stationary_up_in_color(samples, np)

            progress("[2/4] 采集起点 RGB-D，检查地面和纹理……")
            initial = _capture_stationary("起点", camera, controller, folder, 0, np)
            floor = estimate_floor(initial.camera, up_hint, np)
            up = up_hint + floor.up_in_color
            up /= np.linalg.norm(up)
            pitch, roll = mount_angles_from_up(up)
            require_visual_features(initial.camera, cv2)
            progress(f"地面与静止检查通过：高度 {floor.height_m:.3f} m，"
                     f"pitch {math.degrees(pitch):.2f}°，roll {math.degrees(roll):.2f}°")

            progress("[3/4] 左转、回正、右转、回正，再向前平移……")
            start = initial.base_pose
            frames = [initial]
            angle = math.radians(turn_angle_deg)
            for label, yaw in (("左转", start.yaw_rad + angle), ("左转后回正", start.yaw_rad),
                               ("右转", start.yaw_rad - angle), ("右转后回正", start.yaw_rad)):
                progress(label)
                controller.turn_to_world_yaw(yaw)
                frames.append(_capture_stationary(label, camera, controller, folder, len(frames), np))
            # 从最后一次回正的实际位置沿起始朝向前进，不补偿此前累计的位置漂移。
            current = controller.read_pose()
            controller.drive_to_world_xy((current.x_m + drive_distance_m * math.cos(start.yaw_rad),
                                          current.y_m + drive_distance_m * math.sin(start.yaw_rad)))
            frames.append(_capture_stationary("前进", camera, controller, folder, len(frames), np))
    finally:
        camera.close()

    progress("[4/4] 对齐视觉运动与底盘位姿，求解并核对外参……")
    pairs = estimate_motion_pairs(frames, up, cv2, np, progress)
    _write_json(folder / "motion_pairs.json", [asdict(pair) for pair in pairs])
    forward, left, yaw, residual = solve_planar_extrinsic(pairs, np)
    extrinsics = CameraExtrinsics(height_m=floor.height_m, forward_m=forward, left_m=left,
                                  yaw_rad=yaw, pitch_down_rad=pitch, roll_rad=roll)
    diagnostics = {
        **imu_diagnostics, "floor_inlier_ratio": floor.inlier_ratio,
        "imu_floor_angle_deg": math.degrees(floor.imu_angle_rad), "extrinsic_residual_m": residual,
        "motion_visual_inliers": [pair.visual_inliers for pair in pairs],
        "capture_directory": str(folder.resolve()), "camera_serial_number": samples["serial_number"],
        "turn_angle_deg": turn_angle_deg, "drive_distance_m": drive_distance_m,
    }
    saved = save_camera_extrinsics(output_path, extrinsics, diagnostics,
                                   device="Intel RealSense D435i on SLAMTEC Hermes 48V")
    progress(f"标定完成，外参已保存：{saved}")
    return D435iCalibrationResult(saved, folder, residual)


def _stationary_up_in_color(samples, np):
    """在开发机核对原始 IMU，再将 SDK 深度光学系的向上方向转到彩色光学系。"""
    expected = {"version": 1, "coordinate_frame": "depth_optical", "timestamp_unit": "ms",
                "acceleration_unit": "m/s^2", "angular_velocity_unit": "rad/s"}
    if not isinstance(samples, dict) or any(samples.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IMU 协议、单位或坐标系不匹配")
    arrays = []
    for name in ("acceleration", "angular_velocity"):
        values = np.asarray(samples.get(name), dtype=float)
        if values.ndim != 2 or values.shape[1] != 4 or len(values) < 100 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} 样本不足或格式无效")
        if np.any(np.diff(values[:, 0]) <= 0) or values[-1, 0] - values[0, 0] < 1500.0:
            raise RuntimeError(f"{name} 时间戳无效或时长不足 1.5 秒")
        arrays.append(values[:, 1:])
    acceleration, gyro = arrays
    mean = acceleration.mean(axis=0)
    acceleration_std = float(acceleration.std(axis=0).max())
    gyro_std = float(gyro.std(axis=0).max())
    norm = float(np.linalg.norm(mean))
    if not 7.0 <= norm <= 12.5:
        raise RuntimeError(f"IMU 加速度模长异常：{norm:.3f} m/s²")
    if acceleration_std > 0.35 or gyro_std > 0.05 or float(np.linalg.norm(gyro.mean(axis=0))) > 0.08:
        raise RuntimeError("IMU 显示机器人仍在运动或振动，请静止后重新标定")
    rotation = np.asarray(samples.get("depth_to_color_rotation"), dtype=float)
    if rotation.shape != (9,) or not np.all(np.isfinite(rotation)):
        raise RuntimeError("缺少有效的深度到彩色相机旋转")
    rotation = rotation.reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
        raise RuntimeError("深度到彩色相机旋转矩阵无效")
    up = rotation @ (mean / norm)
    return up / np.linalg.norm(up), {
        "imu_acceleration_std_mps2": acceleration_std, "imu_angular_velocity_std_radps": gyro_std,
        "imu_acceleration_hz": samples["acceleration_hz"], "imu_angular_velocity_hz": samples["angular_velocity_hz"],
    }


def _capture_stationary(label, camera, controller, folder, index, np):
    """停车后丢弃过渡帧，再用取帧前后两次位姿核对稳定性，避免采集延迟错配。"""
    time.sleep(0.5)
    for _ in range(4):
        camera.capture()
    before = controller.read_pose()
    capture = camera.capture()
    after = controller.read_pose()
    _require_same_pose(before, after)
    np.savez_compressed(folder / f"frame-{index:02d}.npz", rgb=capture.rgb, depth_m=capture.depth_m)
    _write_json(folder / f"frame-{index:02d}.json", {
        "label": label, "pose_before": asdict(before), "pose_after": asdict(after),
        "timestamp_s": capture.timestamp_s, "intrinsics": asdict(capture.camera_intrinsics),
    })
    return CapturedFrame(label, after, capture)


def _require_same_pose(before, after):
    turn = (after.yaw_rad - before.yaw_rad + math.pi) % (2 * math.pi) - math.pi
    if math.hypot(after.x_m - before.x_m, after.y_m - before.y_m) > 0.01 or abs(turn) > math.radians(1.0):
        raise RuntimeError("采集期间底盘位姿仍变化超过 1 cm 或 1°，停止标定")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
