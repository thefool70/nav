# 底盘接口标准

底盘到货前需确认的项目清单。适配层（`adapters/`）负责把厂商原始协议转换为
下面的内部标准，核心算法只面对内部标准。

## 内部标准（已定）

| 概念 | 内部表示 |
| --- | --- |
| 位姿 | `Pose2D(x_m, y_m, yaw_rad)`，米 / 弧度，yaw 逆时针为正 |
| 障碍图 | `ObstacleMap(occupancy, resolution_m, origin, frame_id)`，`occupancy[y][x]` 为 `0.0` 自由、`1.0` 占用、`None` 未知 |
| 感知快照 | `NavigationFrame(timestamp_s, pose, obstacle_map, ...)`，`timestamp_s` 单位为秒 |
| 深度图 | `NavigationFrame.depth`，单位为米，`None` 表示无有效深度 |
| 控制命令 | `RelativePoseCommand(forward_m, left_m, yaw_rad)`，机器人坐标系，向前 / 向左 / 逆时针为正 |
| 状态 | `NavigationStatus`：OK / INVALID_INPUT / NO_SOLUTION / NOT_IMPLEMENTED |

## 坐标系契约

- `NavigationFrame.pose` 与 `NavigationGoal.target` 在进入 core 前必须已转换到
  `NavigationFrame.obstacle_map.frame_id` 坐标系，core 内部不做坐标转换；
  该转换由适配层负责。

## 底盘到货前必须确认

### 坐标
- 机器人坐标系原点的位置（车体中心？轮轴中点？）与各轴指向。
- 深度相机与机器人坐标系之间的外参（安装位置与朝向）。
- 障碍图 frame_id 与机器人 / 地图坐标系的对应关系。

### 单位与方向
- 平移单位是否为米；角度单位是否为弧度。
- yaw 的零点与逆时针为正是否符合内部标准，是否需要取反或偏移。

### 时间戳
- 时间戳的基准（启动时间、墙钟、单调时钟）与单位（秒？）。
- 深度、位姿、障碍图是否同一时刻采集，时间戳对齐方式。

### 障碍图
- 分辨率（米/格）与原点语义：`origin` 对应格 `(0,0)` 的哪个角。
- 行/列到世界坐标的映射；未知区域的表示方式（`None`？）。

### 图像标定
- RGB 与深度图的尺寸、排列（行主序）与对齐关系（是否已配准）。
- RGB 颜色空间（BGR/RGB）与取值范围。

### 命令覆盖与反馈语义
- `send_relative_pose` 的位移/旋转覆盖范围、单位换算与限幅。
- 命令是增量执行还是重置累计误差；执行期间是否返回反馈、如何表示完成或失败。
- 底盘异常、未就绪、目标不可达时的行为如何映射到 `NavigationStatus`。
