"""D435i 原始 IMU 采集；只传输样本与单位，静止判定和标定留在开发机。"""

import importlib
import time


def read_d435i_imu_samples(serial_number):
    """RGB-D 已关闭时独占采集 IMU；分别选择 accel/gyro 实际公布的频率。"""
    rs = importlib.import_module("pyrealsense2")
    devices = [device for device in rs.context().query_devices()
               if device.get_info(rs.camera_info.serial_number) == serial_number]
    if len(devices) != 1:
        raise RuntimeError("无法找到与 RGB-D 相同的 D435i")
    device = devices[0]
    rates = {}
    for stream, preferred in ((rs.stream.accel, 100), (rs.stream.gyro, 200)):
        supported = {p.fps() for sensor in device.query_sensors() for p in sensor.get_stream_profiles()
                     if p.stream_type() == stream and p.format() == rs.format.motion_xyz32f}
        if not supported:
            raise RuntimeError(f"D435i 没有可用的 {stream} 采样配置")
        rates[stream] = min(supported, key=lambda rate: (abs(rate - preferred), rate))

    config = rs.config()
    config.enable_device(serial_number)
    for stream, rate in rates.items():
        config.enable_stream(stream, rs.format.motion_xyz32f, rate)
    print(f"D435i IMU 配置：accel={rates[rs.stream.accel]} Hz，gyro={rates[rs.stream.gyro]} Hz。", flush=True)
    pipeline = rs.pipeline()
    started = False
    samples = {rs.stream.accel: [], rs.stream.gyro: []}
    last_frame = {}
    domains = {}
    try:
        pipeline.start(config)
        started = True
        warmup_end = time.monotonic() + 0.5
        deadline = warmup_end + 8.0
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            # 模式切换后的短暂缺帧只消耗总期限，不让一次 1s 超时提前终止整批采集。
            arrived, frames = pipeline.try_wait_for_frames(min(1000, remaining_ms))
            if not arrived:
                continue
            if time.monotonic() < warmup_end:
                continue
            for stream, values in samples.items():
                frame = frames.first_or_default(stream)
                if not frame:
                    continue
                number = frame.get_frame_number()
                if last_frame.get(stream) == number:
                    continue
                last_frame[stream] = number
                data = frame.as_motion_frame().get_motion_data()
                values.append([frame.get_timestamp(), float(data.x), float(data.y), float(data.z)])
                domains[stream] = str(frame.get_frame_timestamp_domain())
            if all(len(values) >= 100 and values[-1][0] - values[0][0] >= 1500.0 for values in samples.values()):
                break
            if any(len(values) > 4000 for values in samples.values()):
                raise RuntimeError("D435i IMU 样本时间戳未正常推进")
        else:
            details = []
            for stream, name in ((rs.stream.accel, "accel"), (rs.stream.gyro, "gyro")):
                values = samples[stream]
                duration = (values[-1][0] - values[0][0]) / 1000.0 if len(values) > 1 else 0.0
                details.append(f"{name}={len(values)} 个/{duration:.2f}s")
            raise RuntimeError(
                "D435i IMU 在总采集期限内未收齐样本：" + "，".join(details)
                + "；两路分别要求至少 100 个且覆盖 1.5s。"
                + "若在 WSL 中两路均为 0，请检查 SDK/固件配套及 USB 强制绑定；"
                + "仅显示 Attached 不足以确认 IMU 通道可用。"
            )
    finally:
        if started:
            pipeline.stop()

    # SDK 的 D435i motion_data 已转换到深度光学系，不再次乘 IMU 到 depth 的外参。
    # https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md
    return {
        "version": 1, "serial_number": serial_number, "coordinate_frame": "depth_optical",
        "timestamp_unit": "ms", "acceleration_unit": "m/s^2", "angular_velocity_unit": "rad/s",
        "acceleration": samples[rs.stream.accel], "angular_velocity": samples[rs.stream.gyro],
        "acceleration_hz": rates[rs.stream.accel], "angular_velocity_hz": rates[rs.stream.gyro],
        "acceleration_timestamp_domain": domains[rs.stream.accel],
        "angular_velocity_timestamp_domain": domains[rs.stream.gyro],
    }
