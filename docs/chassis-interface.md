# 底盘接口标准

真机接入时需确认的项目清单。适配层（`adapters/`）负责把厂商原始协议转换为
下面的内部标准，核心算法只面对内部标准。

## 内部标准（已定）

| 概念 | 内部表示 |
| --- | --- |
| 位姿 | `Pose2D(x_m, y_m, yaw_rad)`，米 / 弧度，yaw 逆时针为正 |
| 障碍图 | `ObstacleMap(occupancy, resolution_m, origin, frame_id)`，`occupancy[row][col]` 为 `0.0` 自由、`1.0` 占用、`None` 未知 |
| 障碍图原点 | `origin` 为格 `(row=0, col=0)` 中心在世界坐标系中的位姿；列沿 origin 局部 `+x` 方向增长，行沿 origin 局部 `+y` 方向增长 |
| 目标 | `TargetSearchGoal(target_text, search_mode)`；模式为具体物体 `object` 或目的场景 `scene` |
| 感知快照 | `NavigationFrame(timestamp_s, pose, obstacle_map, ...)`，`timestamp_s` 单位为秒 |
| 深度图 | `NavigationFrame.depth`，单位为米，`None` 表示无有效深度 |
| 相机标定 | `CameraIntrinsics` 与 `CameraExtrinsics`，RGB/深度必须对齐 |
| 控制命令 | `RelativePoseCommand(forward_m, left_m, yaw_rad)`，机器人坐标系，向前 / 向左 / 逆时针为正 |
| 状态 | `NavigationStatus`：`OK`、`NO_SOLUTION`、输入/数据错误，或请求目标观测、场景判断、Frontier 评分和目标确认 |

## 坐标系契约

- `NavigationFrame.pose` 在进入 core 前必须已转换到
  `NavigationFrame.obstacle_map.frame_id` 坐标系，core 内部不做坐标转换；
  该转换由适配层负责。

## 执行契约

当前 `run_navigation_cycle` 采用同步命令语义：`send_relative_pose()` 返回时，
该命令必须已经完成；执行失败则抛出明确异常。下一周期读取的位姿和图像必须
是命令完成后的新状态。

如果失败只表示规划器无法到达本次目标点，Adapter 抛出
`RecoverableMotionError`；Frontier 探索会淘汰对应目标并继续，返回父节点失败则从
实际位置重新检查并选点。
设备离线、健康异常、数据读取失败等系统错误必须使用普通异常，仍然终止运行。

提供实际规划路径的 Adapter 可实现 `KnownSpaceChassisInterface` 扩展。
`app.py` 对 Frontier 探索、返回父节点与恢复暂存方向调用其
`send_relative_pose_in_known_space(command, obstacle_map, *, reference_pose)`，
显式传入命令、决策地图与 `frame.pose`。Adapter 必须用该参考位姿还原世界目标，
不能用发送时重新读取的位姿解释同一条相对命令；
启动前移、标定、扫描转向与目标接近使用普通发送接口。
当前 Hermes 支持该扩展，Habitat 和 S100 仍使用原接口。

受约束的路径必须与传入地图同坐标系，逐段检查当前位置和剩余路径点之间的连线。
地图内 `None` 格和地图外部都算未知。Hermes 累加本次剩余路径在未知区内的实际
长度，默认超过 1.5 m 才取消；等于或小于上限时继续。多段未知区累加，不按最长
连续段判断，也不跨轮询或跨动作累计已经走过的距离。可用 `--max-unknown-path-m`
调整该上限。超限时 Adapter 取消当前动作并
确认其结束，再抛 `MotionPathUnknownError`（`RecoverableMotionError` 的子类），
附带包含当前位置的被拒绝路径 `path_world_xy` 及未知长度、上限和总路径长度。
`app.py` 将路径显式传给核心，
用于记录取消原因；Frontier 探索对应区域在本次运行中持续屏蔽，不自动恢复。
返回父节点的路径未知长度超限时，核心跳过本次返回节点并重新观察，不据此屏蔽
尚未尝试的暂存区域；该节点方向解除暂存后，继续接受当前地图的候选过滤。
普通目标点失败仍使用 `RecoverableMotionError`；读取路径或取消确认失败则停止运行。
检查不得依赖可视化回调，也不得用运动中扩展的可见地图替换本次决策快照。
测量按线段与栅格的交点计算实际长度，不按未知格数量乘分辨率估算；只接触格角
不增加长度，沿格边行走时任一侧未知即计入一次。碰撞与车体净空仍由底盘负责。

如果移动只是因连续静止达到门槛而主动终止、但当前位姿和传感器仍可继续使用，
Adapter 抛出 `MotionStalledError`。Frontier 探索不会把它当作规划失败，而是以
实际当前位置开始下一轮扫描；返回父节点停滞也从实际位置重新检查。两种异常不得混用。
Hermes 的平移、转向执行失败、停滞和目标检测中断，均须取消并确认 Action 进入
终态后才允许恢复导航；取消或确认失败仍使用普通异常停止运行。

返回父节点的命令结束后，核心还检查实际位置与父节点是否相距不超过 0.25 m；
超出容差以 `motion.backtrack_recovered` 返回 `OK`，将失败节点移出回退栈，
保留其 Frontier 为待重新评估候选，下一周期读取实际位置后继续搜索。正常回退
一次只提交当前分支的上一节点，到达后没有有效探索方向才提交更上一层的位置。

如果厂商 SDK 只提供异步接口，真机 Adapter 需要在内部等待完成反馈，不能在
刚下发命令时就返回。以后若要支持连续速度控制，再统一扩展接口和状态机，
不要只在真机实现中改变语义。

## 感知边界

`TargetObserver` 与底盘接口相互独立。Adapter 只负责把同步的 RGB、深度、内参和
外参放进 `NavigationFrame`；目标检测、场景判断和 Frontier 评分不应写进
`ChassisInterface`。持续检测器可以在底盘运动期间读取 Adapter 提供的新帧，但
最终结果仍通过 `TargetObservation` 进入核心状态机。

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

具体实现见 [Habitat](habitat.md)、[Hermes + L515](slamtec-l515.md) 和
[S100 + L515](s100-l515.md)。
