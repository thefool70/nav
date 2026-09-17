# robot-nav

面向算法研究的机器人语义搜索项目。算法接收位姿、RGB-D 和占用图，输出相对
位姿命令；仿真器和真底盘的差异全部收在 Adapter 中。

当前支持两种任务：

- `object`：寻找一个具体物体。
- `scene`：寻找一个目的场景，例如洗手间或电梯厅。

## 从哪里开始读

建议按下面的顺序阅读：

1. `src/robot_nav/app.py`：一个导航周期如何串联输入、感知、决策和运动。
2. `src/robot_nav/core/navigator.py`：环境无关的搜索状态机。
3. `src/robot_nav/core/models.py`：算法输入、输出和跨周期状态。
4. `src/robot_nav/core/frontier.py`：Frontier 的生成、聚类和排序。
5. 当前使用的 Adapter：Habitat、S100、Hermes 直连或香橙派无线转发。

算法细节见 [算法说明](docs/algorithm.md)，坐标和接口约定见
[底盘接口标准](docs/chassis-interface.md)。

## 架构

```text
设备 / 仿真器
    │
    ▼
ChassisInterface ──► NavigationFrame
                           │
VLM 视觉队列 ─────► TargetObserver
                           │
                           ▼
                run_navigation_cycle()
                           │
                           ▼
                      navigate()
                           │
                           ▼
                RelativePoseCommand
                           │
                           ▼
                  ChassisInterface
```

- `core/` 只做算法计算，不读取设备、不请求模型、不发送控制命令。
- `app.py` 负责单周期编排，并把可恢复的运动结果送回状态机。
- `adapters/` 负责设备协议、坐标转换、地图归一化和视觉模型。
- `visualization/` 与 `run_log.py` 只记录过程，不参与决策。
- `__main__.py` 只负责命令行、组件组装和循环运行。

## 搜索方式

| 模式 | 发现目标 | VLM 的作用 | 后续行为 |
| --- | --- | --- | --- |
| 物体搜索 | 后台检查扫描和移动中采集的固定画面 | 联合检测与评分；接近时可提供目标框 | 历史 RGB-D 优先定位，失败后尝试障碍射线；得到位置后接近 |
| 场景搜索 | 后台检查扫描和移动中采集的固定画面 | 同一次请求判断场景与评分 Frontier | 返回拍摄位置并对齐朝向，搜索完成 |

两种模式都在当前可达自由区内选择 Frontier，一次移动到选定位置，动作结束后
再观察和选择下一目标。新出现的 Frontier 优先探索，未选方向暂存；新候选耗尽
后，沿当前分支逐个返回父节点，直到到达仍有有效探索方向的节点，再继续寻找。
首次环扫后，仅补查局部可见且尚未检查的 Frontier 方向；返回父节点不额外环扫。
地图已知与视觉已检查分别记录；两种模式没有待查方向时仍采集当前画面，不额外转向。
返回节点未完成时保留实际位置，跳过该返回节点并重新检查有效方向，不直接结束搜索。

命令行默认使用异步 FIFO 视觉队列。每轮扫描收齐后，将全部画面拼成一张图，
在同一次请求中提取目标线索并评分 Frontier。扫描转向不额外提交单图，探索和分支回退的
平移动作途中保留提前采样。缺少语义分时按几何分继续走，评分只在下一次决策生效。
后台检测到目标后按模型顺序处理线索。场景模式返回拍摄位置并对齐朝向后完成，
不进行到场视觉复查。物体模式先在历史 RGB-D 上定位，YOLO 或 VLM 任一路
检出即可使用；优先用 SAM2 掩码，分割失败时直接用检测框内深度。无法用深度定位时，
有框就沿框中心方向查询地图中的首个障碍，无框则沿相机光轴，将障碍表面作为目标位置假设。
得到位置后，在目标周围搜索可达自由格，从当前位置一次前往停靠点并对准目标，
停靠命令执行成功后直接完成搜索。
历史定位失败时留在当前位置，继续下一条线索，并处理已采集但尚未分析的画面。
全部历史线索都无法定位时，才保底返回首条线索的拍摄位姿并停止，不再重采或继续探索。
实际运动失败时可换点，每条已定位线索最多尝试三次停靠；均失败且历史队列耗尽后恢复探索。
当前线索定位和接近期间暂停普通队列与运动采样，迟到结果暂存；需要更多历史线索时
只恢复已有队列的分析，不为此新增运动。
有效探索方向耗尽后先等待队列，模型失败不冒充“已经检查”。

Hermes 执行 Frontier 移动时，还会按选点时的算法地图检查实际路径。路径经过
未知区的累计长度超过 1.5 m 才取消动作，在本次运行中持续屏蔽整个连通 Frontier 区域并转向其他候选，
避免在同一片边界内换点反复取消。

## 运行环境

| 环境 | 用途 | 文档 |
| --- | --- | --- |
| `robot-nav` | 核心代码、Hermes + D435i | [Hermes + D435i](docs/slamtec-l515.md) |
| `robot-nav-habitat` | Habitat-Sim 仿真 | [Habitat](docs/habitat.md) |
| `robot-nav-slam` | S100 + L515、ROS 2 SLAM | [S100 + L515](docs/s100-l515.md) |
| 开发机 `robot-nav` + 香橙派 Python | Hermes + D435i 无线数据转发 | [香橙派无线适配](docs/orangepi.md) |

核心环境的最小安装：

```bash
micromamba env create -f environment.yml
micromamba activate robot-nav
python -m pip install -e .
```

Rerun 是可选依赖：

```bash
python -m pip install -e '.[visualization]'
```

## 当前实现

- Habitat：提供 RGB-D、二维位姿和逐步公开的占用图；相对目标由 navmesh 规划并
  离散执行。
- Hermes + D435i（`slamtec-d435i`，由原 L515 直连版本改造）：Hermes 提供位姿、激光地图和自主规划；D435i 提供 RGB-D，
  探索地图只在当前理论水平 FOV 内刷新，视场外保留历史值，不考虑遮挡；
  启动区域仅初始化一次。物体障碍定位和停靠使用完整导航图，选点时保留 0.36 m 净空。
  实际位姿到达并稳定后主动结束 Action，确认终态后继续。
  两种模式与 Habitat、S100 共用探索队列及物体接近状态机。
- S100 + L515：可由 `slam_toolbox` 生成位姿和占用图，也保留小范围直接模式。
- 香橙派 + Hermes + D435i：香橙派传输 RGB-D、标定所需 IMU 及底盘网络通信；导航、地图
  处理、动作控制及模型留在开发机。相机按请求开启，空闲 15 分钟后关闭数据流。
  使用 `python -m robot_nav orangepi`，
  `calibrate-orangepi` 自动求解 D435i 安装外参；部署和 SSH 隧道见 [无线适配说明](docs/orangepi.md)。
- Rerun：显示 RGB、深度、地图、Frontier、机器人轨迹、算法目标、底盘目标和
  规划路径，同时持续写入 `data/run_logs/rerun-*.rrd`。`--rerun-save <PATH>`
  可指定新文件路径；`--no-rerun` 同时关闭界面和录制。
  `VLM summary` 简要关联任务、请求与导航使用；`VLM full` 保留完整会话信息。
  World 将占用图、机器人和任务标记放在一起，同次扫描只画一个点，邻近任务合并显示。
  右侧直接展示当前推理的 RGB 与评分；`Observations` 中点击 J 查看整组、V 查看
  单图评分卡，原图通过 raw 链接查看。完整点位保留在 `World history`。
- Hermes JSONL 日志：记录每周期决策、候选评分和 Action 反馈，供事后复盘。
- 视觉快照：`data/run_logs/semantic-*/job-*` 保存图片、位姿、候选和分析结果；
  普通任务按 FIFO 处理。目录在退出后保留，当前不自动恢复上次队列。
  同帧深度先临时压缩保存，分析后只保留命中画面的正式深度文件；检测失败或
  未完成的任务保留临时深度，等待明确判断。

## 主要目录

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/` | 状态机、Frontier、扫描、定位、历史和数据契约 |
| `src/robot_nav/core/observation_coverage.py` | 局部 Frontier 观察点、RGB-D 覆盖记录和跨位置复用 |
| `src/robot_nav/core/frontier_projection.py` | Frontier 地面点投影与对齐深度核对，供 VLM 图片标注 |
| `src/robot_nav/core/path_validation.py` | 测量实际规划路径在算法未知区内的累计长度 |
| `src/robot_nav/core/object_approach.py` | 历史线索处理、停靠完成与保底返回停止 |
| `src/robot_nav/core/object_grounding.py` | RGB-D 定位与图像方向上的障碍位置假设 |
| `src/robot_nav/core/object_standoff.py` | 在目标周围搜索满足净空与连通条件的停靠点 |
| `src/robot_nav/adapters/habitat/` | Habitat Adapter |
| `src/robot_nav/adapters/slamtec_l515/` | Hermes + D435i Adapter |
| `src/robot_nav/adapters/s100_l515/` | S100 + L515 Adapter |
| `src/robot_nav/adapters/orangepi/` | 无线 Adapter、远程相机客户端和香橙派 RGB-D 数据服务 |
| `src/robot_nav/adapters/realsense/` | RealSense RGB-D 采集、D435i 配置与相机外参标定 |
| `src/robot_nav/adapters/openai_compatible.py` | VLM 请求与结构化结果解析 |
| `src/robot_nav/adapters/queued_semantics.py` | 快照落盘、FIFO 联合分析、提前采样与迟到结果接收 |
| `src/robot_nav/adapters/object_localizer.py` | 接近阶段的 VLM、本地模型与定位编排 |
| `src/robot_nav/adapters/object_detection_worker.py` | 在指定环境中常驻运行 YOLO 或 SAM2，记录阶段与调用栈 |
| `src/robot_nav/adapters/object_model_process.py` | 模型进程的请求、进度、超时与退出 |
| `src/robot_nav/adapters/yolo_world_sam2.py` | 可复用的 YOLO/SAM2 检测器与同步 API 观察器 |
| `src/robot_nav/visualization/` | Rerun 调试界面 |
| `src/robot_nav/visualization/semantic_world.py` | World 节点的固定 RGB、评分与状态关联 |
| `src/robot_nav/visualization/observation_card.py` | 同次观测的 RGB、评分锚点与状态卡片 |
| `sim/habitat/` | Habitat 环境与启动脚本 |
| `hardware/`、`slam/` | 真机、USB 和 ROS 启动配置 |

## 当前边界

物体接近的本地模型按需加载后常驻，同帧 SAM2 编码也复用，默认使用已有
`robot-nav` 模型环境。可用
`--object-python` 指定已安装 PyTorch、ultralytics、SAM2 的解释器，
`--object-class` 指定 YOLO 简短类别，`--object-device` 选择设备；场景模式和
随机评分模式不启动本地模型。详细流程与阈值见 [算法说明](docs/algorithm.md)。

这是算法原型，不是功能安全系统。核心输出高层相对位姿，实际路径规划、避障和
执行反馈由 Adapter 后面的仿真器或底盘负责。真机运行必须有人能够立即急停；
动态障碍预测、通用模型重试和连续速度控制尚未实现。
