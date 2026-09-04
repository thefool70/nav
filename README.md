# robot-nav

面向算法研究的机器人语义搜索项目。算法接收位姿、RGB-D 和占用图，输出相对
位姿命令；仿真器和真底盘的差异全部收在 Adapter 中。

当前支持两种任务：

- `object`：寻找并接近一个具体物体。
- `scene`：寻找一个目的场景，例如洗手间或电梯厅。

## 从哪里开始读

建议按下面的顺序阅读：

1. `src/robot_nav/app.py`：一个导航周期如何串联输入、感知、决策和运动。
2. `src/robot_nav/core/navigator.py`：环境无关的搜索状态机。
3. `src/robot_nav/core/models.py`：算法输入、输出和跨周期状态。
4. `src/robot_nav/core/frontier.py`：Frontier 的生成、聚类和排序。
5. 当前使用的 Adapter：Habitat、S100 或 Hermes。

算法细节见 [算法说明](docs/algorithm.md)，坐标和接口约定见
[底盘接口标准](docs/chassis-interface.md)。

## 架构

```text
设备 / 仿真器
    │
    ▼
ChassisInterface ──► NavigationFrame
                           │
本地检测或 VLM ──► TargetObserver
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

| 模式 | 发现目标 | VLM 的作用 | 完成条件 |
| --- | --- | --- | --- |
| 物体搜索 | Hermes 使用 YOLO-World + SAM2 持续检测；其他环境使用通用视觉观察器 | 多个 Frontier 的批量评分；接近候选后的最终确认 | VLM 确认候选就是目标 |
| 场景搜索 | 每轮扫描后拼接全部 RGB | 判断是否已经位于目的场景；否则批量评分 Frontier | VLM 判断已经到达目的场景 |

两种模式都只在占用图的可达自由区内选择 Frontier，并在当前分支走完后回到仍有
候选方向的历史观测点。

## 运行环境

| 环境 | 用途 | 文档 |
| --- | --- | --- |
| `robot-nav` | 核心代码、Hermes + L515 | [Hermes + L515](docs/slamtec-l515.md) |
| `robot-nav-habitat` | Habitat-Sim 仿真 | [Habitat](docs/habitat.md) |
| `robot-nav-slam` | S100 + L515、ROS 2 SLAM | [S100 + L515](docs/s100-l515.md) |

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
- Hermes + L515：Hermes 提供位姿、激光地图和自主规划；L515 提供 RGB-D；
  物体模式使用 YOLO-World + SAM2。
- S100 + L515：可由 `slam_toolbox` 生成位姿和占用图，也保留小范围直接模式。
- Rerun：显示 RGB、深度、地图、Frontier、机器人轨迹、算法目标、底盘目标和
  规划路径。
- Hermes JSONL 日志：记录每周期决策、候选评分和 Action 反馈，供事后复盘。

## 主要目录

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/` | 状态机、Frontier、扫描、定位、历史和数据契约 |
| `src/robot_nav/adapters/habitat/` | Habitat Adapter |
| `src/robot_nav/adapters/slamtec_l515/` | Hermes + L515 Adapter |
| `src/robot_nav/adapters/s100_l515/` | S100 + L515 Adapter |
| `src/robot_nav/adapters/realsense/` | 共用 L515 采集与外参标定 |
| `src/robot_nav/adapters/openai_compatible.py` | VLM 请求与结构化结果解析 |
| `src/robot_nav/adapters/yolo_world_sam2.py` | YOLO-World + SAM2 持续目标检测 |
| `src/robot_nav/visualization/` | Rerun 调试界面 |
| `sim/habitat/` | Habitat 环境与启动脚本 |
| `hardware/`、`slam/` | 真机、USB 和 ROS 启动配置 |

## 当前边界

这是算法原型，不是功能安全系统。核心输出高层相对位姿，实际路径规划、避障和
执行反馈由 Adapter 后面的仿真器或底盘负责。真机运行必须有人能够立即急停；
动态障碍预测、通用模型重试和连续速度控制尚未实现。
