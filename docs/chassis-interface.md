# 底盘接口标准

真机接入时需确认的项目清单。适配层（`adapters/`）负责把厂商原始协议转换为
下面的内部标准，核心算法只面对内部标准。

## 内部标准（已定）

| 概念 | 内部表示 |
| --- | --- |
| 位姿 | `Pose2D(x_m, y_m, yaw_rad)`，米 / 弧度，yaw 逆时针为正 |
| 障碍图 | `ObstacleMap(occupancy, resolution_m, origin, frame_id)`，`occupancy[row][col]` 为 `0.0` 自由、`1.0` 占用、`None` 未知 |
| 障碍图原点 | `origin` 为格 `(row=0, col=0)` 中心在世界坐标系中的位姿；列沿 origin 局部 `+x` 方向增长，行沿 origin 局部 `+y` 方向增长 |
| 目标 | `TargetSearchGoal(target_text)`，`target_text` 为对目标的人类可读描述（如 "门口"） |
| 感知快照 | `NavigationFrame(timestamp_s, pose, obstacle_map, ...)`，`timestamp_s` 单位为秒 |
| 深度图 | `NavigationFrame.depth`，单位为米，`None` 表示无有效深度 |
| 相机标定 | `CameraIntrinsics` 与 `CameraExtrinsics`，RGB/深度必须对齐 |
| 控制命令 | `RelativePoseCommand(forward_m, left_m, yaw_rad)`，机器人坐标系，向前 / 向左 / 逆时针为正 |
| 状态 | `NavigationStatus`：OK / INVALID_INPUT / NO_SOLUTION / NEEDS_OBSERVATION / MISSING_DATA |

## 坐标系契约

- `NavigationFrame.pose` 在进入 core 前必须已转换到
  `NavigationFrame.obstacle_map.frame_id` 坐标系，core 内部不做坐标转换；
  该转换由适配层负责。

## 执行契约

当前 `run_navigation_cycle` 采用同步命令语义：`send_relative_pose()` 返回时，
该命令必须已经完成；执行失败则抛出明确异常。下一周期读取的位姿和图像必须
是命令完成后的新状态。

如果厂商 SDK 只提供异步接口，真机 Adapter 需要在内部等待完成反馈，不能在
刚下发命令时就返回。以后若要支持连续速度控制，再统一扩展接口和状态机，
不要只在真机实现中改变语义。

## 真机接入必须确认

### 坐标
- 机器人坐标系原点的位置（车体中心？轮轴中点？）与各轴指向。
- 深度相机与机器人坐标系之间的外参（安装位置与朝向）；RGB 与深度相机的内参
  与畸变模型。
- 障碍图 frame_id 与机器人 / 地图坐标系的对应关系；厂商地图原点换算为内部
  标准（格中心、行/列与 x/y 的对应）的方式。

### 单位与方向
- 平移单位是否为米；角度单位是否为弧度。
- yaw 的零点与逆时针为正是否符合内部标准，是否需要取反或偏移。

### 时间戳
- 时间戳的基准（启动时间、墙钟、单调时钟）与单位（秒？）。
- 深度、位姿、障碍图是否同一时刻采集，时间戳对齐方式；深度帧与 RGB 帧是否同步。

### 障碍图
- 分辨率（米/格）与未知区域的表示方式（`None`？）。
- 厂商地图到内部标准的行/列方向与原点换算。

### 图像标定
- RGB 与深度图的尺寸、排列（行主序）与对齐关系（是否已配准）。
- RGB 颜色空间（BGR/RGB）与取值范围。
- `CameraExtrinsics` 使用机器人前/左/上平移、向左 yaw、向下 pitch 和从相机
  后方向镜头看时图像顺时针为正的 roll；Adapter 必须在生成帧前统一这些方向。

### 命令覆盖与反馈语义
- `send_relative_pose` 的位移/旋转覆盖范围、单位换算与限幅。
- 相对位姿命令是纯运动执行还是自带路径规划 / 避障。
- 命令是增量执行还是重置累计误差；执行期间是否返回反馈、如何表示完成或失败。
- 底盘异常、未就绪、目标不可达时的行为如何映射到 `NavigationStatus`。

只要上述数据、坐标、标定和同步执行契约全部归一化，仿真切换到真底盘时只需
替换 `ChassisInterface` 的实现；`core` 与 `TargetObserver` 不应包含厂商分支。

当前 S100 + L515 实现与仍需实测的安装参数见
[s100-l515.md](s100-l515.md)。
