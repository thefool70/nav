# Hermes 排错

面向 Agent；使用步骤见 [Hermes 使用说明](../hermes.md)。运行和设备检查须遵守 AGENTS.md 的授权规则。

## 地图缓存

- Hermes `platforms/hermes/observed_map.py::HermesObservedMap` 分别缓存世界格网上的
  `_raw_occupancy` 与 `_inflated_occupancy`。只从当前理论水平 FOV（最远 5m）
  读取激光历史占用值，膨胀结果也只写回该区域；不能通过累计 seen 掩码每帧重读
  所有已见格，否则视场外地图会变化。FOV 不读取深度、不检查遮挡。
  启动点 0.50m 区域只初始化一次，扩图不补读该圆内的视场外新格。原点平移时
  按首次格网锚点投回当前数组；frame_id、分辨率或 yaw 变化才清空并重新初始化。
  该规则控制算法使用的 FOV 缓存，不改变 Hermes 内部 SLAM 与避障地图。
  `platforms/hermes/adapter.py::_capture_frame` 还将同次读取的完整未膨胀图保留为
  `navigation_map`，供物体障碍定位和停靠；`navigation_clearance_m=0.36` 只在
  停靠规划时排除障碍邻域。完整图不得回写到 `HermesObservedMap` 的视场外缓存。
  `HermesObservedMap.update` 返回 `(obstacle_map, visibility_map)`；后者在膨胀前冻结，
  与探索图使用相同 FOV 缓存。`scan.py` 的方向筛选、固定网格覆盖及
  首层未知边界均使用视觉图，Frontier 格子仍从探索图转换为世界点。
  JSONL 的 camera 字段记录 `visibility_map_source`、`origin_in_obstacle_map` 和
  `origin_in_visibility_map`。底盘格为自由不代表前置相机光心也在自由格；若本地待查
  数为 0，先检查光心是否落在导航膨胀带，勿直接当作旧视角复用。
- Hermes 的 `KnownSpaceChassisInterface` 扩展在 `explore.select`、
  `backtrack.return`、`backtrack.resume`、`target.revisit`、`object.fallback_return`
  和 `object.approach` 使用。`pipeline/navigation_loop.py::_execute_action` 对物体停靠优先传入
  `frame.navigation_map`，其余动作传入 `frame.obstacle_map`，同时显式传入
  `reference_pose=frame.pose`。不能用 FOV 图检查完整导航图选出的物体停靠路径。
  启动前移与扫描转向仍走普通接口；
  Habitat 也实现该扩展，在离散动作前检查剩余 navmesh 路径。
- Hermes `_send_relative_pose` 用 `reference_pose` 还原世界目标与目标 yaw，
  `start_pose` 仅计算剩余距离和反馈。排查“Action 完成但未到达父节点”时对照
  `cycle_decision.command.target_world_xy`、`Hermes command target` 和实际终点。
  不能用放宽到达容差掩盖参考位姿错误；平移完成的 `target_error` 仅相对底盘收到的目标。
- `interfaces/geometry.py::measure_unknown_path_length` 按格边交点切分每段路径，
  累加未知部分的实际长度；`None` 和地图外算未知。只触碰格角长度为零，沿格边
  行走时任一侧未知即计入一次。不能用格数×分辨率代替斜线/部分穿格的长度。
  Hermes `_check_known_space_path` 将当前位置
  加在剩余路径前，使用选点地图快照；不能改用原始 Hermes 地图、最新运动帧地图，
  或仅检查离散路径点，否则会漏掉未知绕行。
- `HermesConfig.max_unknown_path_m` 与 CLI `--max-unknown-path-m` 默认均为
  1.5 m，非负有限数。只在未知长度 > 上限时取消（比较保留 1e-9 m 浮点容差）；
  多段未知区累加，不取最长连续段，不累加不同轮询或已走过的路。0 禁止正长度
  未知段；地图快照仍固定到选点时刻，不能改成运动中扩展的可见地图。
- Hermes 路径检查在 `_monitor_action` 的图像采集和进度节流之前执行，不能受
  `on_motion_plan` 或 `path_error_reported` 控制。空路径继续轮询，路径读取错误
  则取消并停止运行，不能用可视化路径读取的容错分支掩盖检查失败。

## 人工墙的状态与日志

- Hermes 人工墙：`motion_blockage.FrontObstructionDetector.observe` 用 `_poses` 保存
  窗口内位姿及边界前最近的一帧，不依赖规划路径；动作结束清空。日志 `duration`
  是实际比较时长，`translation` / `heading_change` 是两端净位移 / 最短角差。
  `samples` 为最近窗口内实际保留的近障碍样本数。扫描 RotateTo 不启用该检测。
  `_continuous_frame_loop` 发布停滞事实；首次由 `_RetryFrontObstructionError` 经过取消
  确认后交给 `_execute_action` 重发。检测器 `_retrying` 记录重发复查阶段，有效运动
  或动作结束时清除；停滞分支只有 `confirmed_after_retry` 才允许写墙。
  未知路径超限由 `_commit_front_obstruction(unknown_path=True)` 在取消确认后调用
  `wall_for_unknown_path(stopped_pose)`，复用窗口深度样本和原目标生成墙，不伪造重试确认。
  保留 MotionPathUnknownError 供核心按原动作用途恢复；取消确认失败不写墙。
  日志 phase 区分普通检测与重试复查。
  `_execute_monitored_action` 对重试受阻确认取消终态后调用
  `_commit_front_obstruction`。`HermesObservedMap.navigation_map` 向完整图叠加同一份原始墙，
  `add_permanent_wall` 按停止后 REST 位姿检查净空并拒绝让膨胀墙覆盖当前格。

## USB 与环境

升级 SDK 后需退出并重启已有相机进程，旧进程不会自动切换到新库。
若 IMU 在总期限内两路均为 0，先核对当前加载的 SDK 与相机固件，再检查
Windows 的 `usbipd` 绑定方式。普通共享即使显示 Attached，RGB-D 也能使用，
仍可能在相机流关闭、重开后没有 IMU 数据。此时取消
`ROBOT_NAV_SKIP_USB_PREPARE=1`，让启动脚本执行 `bind --force` 并重新转发；
Windows 枚举名称中的 `435i` 与 `D435i` 均受支持。
恢复后应检查连续两轮 RGB-D/IMU 切换，不能只用一次冷启动读取判定恢复。
不要通过延长等待或放宽静止门槛掩盖无数据的问题。官方配套表见
[RealSense 固件发布说明](https://dev.realsenseai.com/docs/firmware-releases-d400/)。

## 相机协议与时序

接收端只接受 `rgbd_pose` v2：四段依次为 `rgbd.pose`、msgpack 元数据、RGB8、
小端 uint16 深度。深度乘 `depth_scale_m` 转为米；深度已对齐到彩色图，使用
`color_intr`。v2 的 `depth_intr` 描述对齐后的深度图，原始深度内参另存于
`depth_intr_raw`。接收端要求 `depth_aligned_to=color`、`depth_frame=color_optical`
且深度宽高与彩色图一致；一次性外参脚本要求 `T_map_camera_frame=color_optical`。
发布器必须满足接收端要求的 v2 字段。

| 发布数据 | 导航使用方式 |
| --- | --- |
| `T_map_base` | 提取 x、y、yaw，作为该帧的底盘 `pose`，不再另查 REST 替换 |
| `T_map_camera` | 仅一次性获取外参脚本使用，导航接收器不读取 |
| `timestamp_ns` | 转为 `NavigationFrame.timestamp_s`，保留 RGB 采集时刻的 Unix 秒数 |
| `frame_id` | 发布帧序号，不作为地图坐标系名称 |

FOV、图像投影和物体定位继续使用“同步底盘二维位姿 + 固定安装外参”，
保持当前底盘近似水平、地面 `z=0` 的算法假设。

接收端校验版本、图像编码、同步有效标记与变换矩阵。无效同步位姿直接报错，
不拼接“旧图像 + 最新 REST 位姿”。地图和动作执行反馈仍单独读取 REST；
地图不在同步包内，也不保证与图像同一采集时刻。

默认 848×480、30 FPS 的对齐 RGB-D 原始载荷约 61 MB/s；常驻订阅会持续使用链路
带宽。接收线程独占 socket，完整消息解码后原子替换最新包，不排队保存历史帧；
ZMQ 接收高水位为 1。断流超时后报错，不无限沿用最后一帧，也不静默重试。

`--camera-timeout-s` 默认 3 秒，用于首次等帧、动作后等新包及接收断流判断。
快照读取同时检查本机接收龄。接收龄和源端图像时间是两回事：网络缓冲、发布器
延迟可能让刚收到的包也已经滞后；本机检查不证明传感器到接收端的绝对帧龄。
源端 Unix 时间不与开发机单调时间相减，两机未同步时也不能直接相减估计网络延迟。

发布器用 REST 请求/响应的墙钟中点估计底盘采样时刻，再对 RGB 时刻插值。
默认要求前后位姿间隔各不超过 100 ms、各自 RTT 不超过 40 ms，等待后侧位姿
最多 120 ms；无法同步时默认丢帧。因此接收超时也可能由底盘位姿请求或同步失败
造成，应查看发布器的 `skip_reasons`。这属于软件时间对齐；深度与 RGB 的时间差
只在 `camera_debug` 中记录，深度空间对齐不表示二者硬件同一时刻曝光。

导航日志与历史快照沿用原有结构，保存算法使用的拍摄位姿、时间和固定外参。
发布器每次启动会在 `/home/hri/data_publisher/logs/rgbd-publisher-*.jsonl` 独立记录诊断，
不依赖导航进程或 Rerun。每秒记录采集阶段、采集与发布计数、累计及本周期同步拒绝原因、
最近一次拒绝的时间／RTT／位姿时间差、位姿 API 成功与失败次数及最近错误、ZMQ 丢包计数，
以及距最近采集／发布的时间；退出记录异常原因。计数是本次发布器进程累计值，周期差值
用于对照导航断流时段。采集阻塞或连续同步拒绝时仍写心跳；整机调度停顿也会延迟心跳。
修改发布器后需手动重启发布器才生效，不调整同步门槛、采集频率或导航超时。
验收时先做只读预检，再检查图像投影、历史定位和运动后的新帧。

## 随车网络

随车笔记本默认使用本地 Qwen；Wi-Fi 用于远程访问和联网，有线连接用于底盘。底盘 DHCP 提供的
`192.168.11.1` 不接受 DNS 查询，因此有线连接 `Wired connection 1` 已设置
`ipv4.ignore-auto-dns=yes` 和 `ipv6.ignore-auto-dns=yes`，域名由 Wi-Fi DNS 解析。
若重新创建有线连接后出现 `Temporary failure in name resolution`，检查
`resolvectl status`，避免再次将底盘地址用作 DNS；无需修改模型或密钥。


## 执行与退出定位

- Hermes `MoveToAction` 从首帧积累 `blocked_pose_duration_s`（3s）滑动窗口，无起步宽限。
  窗口净位移不足 0.25m 即停滞，角差只记录；首次取消确认后重发原目标。
  重发从首帧复查 3s，位移达到 0.25m 清除重试状态、恢复普通检测；
  仍不足才取消确认并建墙。RotateTo 不检测；未知路径超限提前取消确认后直接建墙，不等待复查。
  近深度只细化墙的位置，不参与触发判断。
  “前向采样区未检出近障碍”指 0.1～`front_blockage_distance_m` 范围内无合格深度簇，
  不能据此判断目标深度缺失；目标定位看 `object_localized` 的 `mask_used/sample_count/reason`。
  `--action-stall-timeout-s` 只控制转向无进展上限。Action 轮询默认 0.2s，
  运动帧间隔默认 0.5s；实际触发时间还受相机采集与 REST 延迟影响。
- `platforms/hermes/adapter.py::_check_stable_arrival` 对 working Action 检查目标
  位置误差 ≤`action_arrival_position_m`（0.30m）或转向误差 ≤`yaw_tolerance_rad`（5°）。
  在容差内连续 `action_arrival_hold_s`（0.001s，实际至少等到后续轮询）相对采样锚点移动 <2cm、转动 <1° 才
  抛出内部到达信号；MoveTo 还要求先有有效平移。`_execute_monitored_action` 捕获后
  保留任务供下一 Action 替换；没有下一动作、进入定位、异常或退出时才取消并确认终态。
  该到位交接不等于固件报告成功。该容差不同于
  `position_tolerance_m`（0.03m），后者仅决定是否下发微小平移命令。
- `platforms/setup.py::_move_hermes_forward_on_start` 在 `run_navigation` 之前直接调用
  底盘；前移 1 m 的可恢复失败/停滞在确认动作结束后继续启动，其他异常仍停止。
- `platforms/hermes/adapter.py::_execute_action` 将 Action 创建、起始位姿读取和监控
  放在同一异常范围；`_cancel_active_action` 与 `close` 负责取消遗留动作。已获得
  ID 时还要确认终态；创建请求失败而没有 ID 时只能尝试取消，原异常仍终止运行。