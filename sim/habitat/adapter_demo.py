"""启动 Habitat Adapter 并读取一帧，确认仿真输入边界可用。

这不是完整目标搜索入口；完整搜索还需要一个 ``TargetObserver`` 实现。
"""

import argparse

from robot_nav.adapters.habitat import HabitatChassisAdapter, HabitatConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="读取一帧 Habitat 导航数据")
    parser.add_argument("--scene", required=True, help="Habitat .glb 场景路径")
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=-1,
        help="Habitat 渲染设备；Mesa/llvmpipe 使用 -1",
    )
    args = parser.parse_args()

    config = HabitatConfig(
        scene_path=args.scene,
        gpu_device_id=args.gpu_device_id,
    )
    with HabitatChassisAdapter(config) as chassis:
        frame = chassis.read_frame()

    map_rows = len(frame.obstacle_map.occupancy)
    map_cols = len(frame.obstacle_map.occupancy[0])
    rgb_shape = (
        (len(frame.rgb), len(frame.rgb[0])) if frame.rgb is not None else None
    )
    depth_shape = (
        (len(frame.depth), len(frame.depth[0])) if frame.depth is not None else None
    )
    print(f"pose={frame.pose}")
    print(f"map_shape=({map_rows}, {map_cols})")
    print(f"rgb_shape={rgb_shape}, depth_shape={depth_shape}")


if __name__ == "__main__":
    main()
