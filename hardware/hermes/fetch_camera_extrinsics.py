"""从随车发布器获取一次固定安装外参，写入现有 config.json 的六项相机配置。"""

import argparse
import json
import math
from pathlib import Path

import msgpack
import numpy as np
import zmq

from robot_nav.adapters.hermes.remote.wire import MAX_FRAME_BYTES, read_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera = config["camera"]
    topic = camera["camera_topic"].encode("utf-8")

    # 只订阅一包，不启动相机、不控制底盘；无有效数据时不改配置。
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.connect(camera["camera_endpoint"])
        if not socket.poll(int(camera["camera_timeout_s"] * 1000), zmq.POLLIN):
            raise RuntimeError("等待同步数据超时，请检查发布器、IPC 隧道和 topic")
        parts = socket.recv_multipart()
    finally:
        socket.close(0)
        context.term()
    if len(parts) != 4 or parts[0] != topic or len(parts[1]) > 64 * 1024:
        raise ValueError("发布器消息格式错误")
    meta = msgpack.unpackb(parts[1], raw=False)
    if meta["version"] != 2 or meta["type"] != "rgbd_pose":
        raise ValueError("需要 rgbd_pose v2 协议")
    if meta["T_map_camera_frame"] != "color_optical":
        raise ValueError("固定外参必须描述彩色相机光学坐标系")
    if meta["pose_valid"] is not True or meta["sync"]["valid"] is not True:
        raise ValueError("本包位姿未同步，未保存外参")

    map_base = read_transform(meta["T_map_base"], "T_map_base", np)
    map_camera = read_transform(meta["T_map_camera"], "T_map_camera", np)
    # 消去完整底盘位姿，只留下固定安装关系；不能用二维底盘位姿代替这里的矩阵。
    base_camera = np.linalg.solve(map_base, map_camera)
    # 光学轴为右/下/前；配置为前/左/上，角度遵循项目 yaw/pitch_down/roll 约定。
    values = {
        "camera_forward_m": float(base_camera[0, 3]),
        "camera_left_m": float(base_camera[1, 3]),
        "camera_height_m": float(base_camera[2, 3]),
        "camera_yaw_deg": math.degrees(math.atan2(base_camera[1, 2], base_camera[0, 2])),
        "camera_pitch_down_deg": math.degrees(math.atan2(
            -base_camera[2, 2], math.hypot(base_camera[0, 2], base_camera[1, 2]))),
        "camera_roll_deg": math.degrees(math.atan2(base_camera[2, 0], -base_camera[2, 1])),
    }
    if values["camera_height_m"] <= 0:
        raise ValueError("相机高度必须为正，请检查发布器外参的坐标约定")
    camera.update(values)
    args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"固定安装外参已保存到 {args.config}：")
    print(json.dumps(values, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
