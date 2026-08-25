# robot-nav

面向算法研究的机器人语义目标搜索项目。给定目标描述（如“门口”），算法会
扫描环境、定位可见目标；未发现目标时选择 Frontier 继续探索，并在走到尽头
时回到仍有候选方向的历史观测点。

## 数据流

```text
ChassisInterface ──► NavigationFrame ──┐
                                      ├─► navigate() ──► RelativePoseCommand
TargetObserver ────► TargetObservation ┘
```

- `ChassisInterface` 统一仿真器和真底盘的数据与控制接口。
- `TargetObserver` 统一视觉/VLM 的目标可见性、方向评分和目标框输出。
- `core` 只处理算法，不依赖 Habitat、底盘 SDK 或具体视觉模型。

`run_navigation_cycle` 串联一个周期：读取帧、调用目标观察器、推进算法、在有
有效命令时发送给底盘。通用命令行入口负责选择 Adapter 并重复执行周期。

## 当前能力

- Habitat Adapter：RGB、深度、二维位姿、局部已知障碍图，以及基于 navmesh
  的相对位姿规划与离散动作执行。
- S100 + L515 Adapter：用 S100 轮速里程计、L515 深度扫描和
  `slam_toolbox` 生成统一位姿/占用图；支持借助 L515 Motion Module 与 RGB-D
  自动估计安装外参，并保留直接深度累计模式用于局部排错。
- Hermes + L515 Adapter：通过 SLAMTEC Robot Agent REST API 读取 Hermes 位姿和
  激光栅格图，并把相对位姿转换为底盘自主规划 Action；外接 L515 只负责对齐
  RGB-D，不参与底盘定位和避障。
- OpenAI-compatible 目标观察器：支持 Chat Completions 和 Responses API，输出
  目标可见性、不可见方向评分和可见目标框。
- 算法主流程：四向扫描、目标框与深度定位、安全距离接近、Frontier 提取与
  排序、观测历史和回退。
- Rerun 实时可视化：记录算法决策帧和 Habitat 动作中间帧，包括 RGB、米制
  深度、三色占用图、机器人位姿与轨迹、控制命令、历史候选点和状态；Habitat
  入口默认启用，用 `--no-rerun` 关闭。
- 通用入口默认使用 OpenCode Zen 和 Muse Spark 1.2；Key 只在运行时读取，不写入
  仓库。未提供观察器时算法会返回明确的 `NEEDS_OBSERVATION`。
- Habitat 入口支持 `--debug-random-score`：不创建也不调用任何视觉模型，观察器
  每次只返回 `NOT_VISIBLE` 和随机方向评分，用于调试扫描、Frontier、移动和回退；
  此模式无法识别或到达语义目标，也不要求 `ROBOT_NAV_VLM_API_KEY`。

算法流程见 [docs/algorithm.md](docs/algorithm.md)，环境接入见
[Habitat](docs/habitat.md)、[S100 + L515](docs/s100-l515.md) 和
[Hermes + L515](docs/slamtec-l515.md)。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/navigator.py` | 算法入口和阶段流转 |
| `src/robot_nav/core/frontier.py` | 可达 Frontier 提取与排序 |
| `src/robot_nav/core/grounding.py` | 目标框与深度的二维定位 |
| `src/robot_nav/core/vision.py` | VLM 提示词和回答解析 |
| `src/robot_nav/core/history.py` | 探索方向历史与回退依据 |
| `src/robot_nav/core/models.py` | 全部输入、输出与状态契约 |
| `src/robot_nav/adapters/chassis.py` | 仿真器/真底盘共同接口 |
| `src/robot_nav/adapters/habitat/adapter.py` | Habitat-Sim Adapter |
| `src/robot_nav/adapters/s100_l515/adapter.py` | S100 + L515 真机 Adapter |
| `src/robot_nav/adapters/slamtec_l515/adapter.py` | Hermes + L515 真机 Adapter |
| `src/robot_nav/adapters/slamtec_l515/rest_client.py` | Hermes REST 与 Action 边界 |
| `src/robot_nav/adapters/realsense/l515_camera.py` | 底盘无关的 L515 RGB-D 采集 |
| `src/robot_nav/adapters/s100_l515/calibration/` | 一次性相机安装外参标定 |
| `src/robot_nav/adapters/s100_l515/ros_slam.py` | ROS SLAM 与统一导航帧的边界 |
| `src/robot_nav/adapters/perception.py` | 视觉/VLM 接口 |
| `src/robot_nav/adapters/openai_compatible.py` | OpenAI-compatible VLM 调用 |
| `src/robot_nav/adapters/random_observer.py` | 调试随机方向评分观察器 |
| `src/robot_nav/visualization/rerun_view.py` | Rerun 导航调试界面 |
| `src/robot_nav/app.py` | 单周期串联入口 |
| `src/robot_nav/__main__.py` | Adapter 选择、循环和终端输出 |
| `slam/s100_l515/` | S100 + L515 的 ROS 环境、启动与 SLAM 参数 |

## Python 环境

核心支持 Python 3.9 及以上，运行时只依赖标准库：

```bash
micromamba env create -f environment.yml
micromamba activate robot-nav
python -m pip install -e .
```

Rerun 可视化是可选依赖，核心环境按需安装：

```bash
python -m pip install -e '.[visualization]'
```

Habitat 使用独立环境，避免其 Python 与图形依赖污染核心环境。安装和启动命令
见 [docs/habitat.md](docs/habitat.md)。

S100 + L515 的 ROS SLAM 同样使用独立的 `robot-nav-slam` 环境，安装、USB
转发和启动命令见 [docs/s100-l515.md](docs/s100-l515.md)。

## 接入真底盘

S100 + L515 的安装、预检和启动方法见 [docs/s100-l515.md](docs/s100-l515.md)；
Hermes + L515 见 [docs/slamtec-l515.md](docs/slamtec-l515.md)。
其他设备只需实现同一个 `ChassisInterface`，算法与 `TargetObserver` 不应改动；
统一坐标、单位和同步语义见 [docs/chassis-interface.md](docs/chassis-interface.md)。
