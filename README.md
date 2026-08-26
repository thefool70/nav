# robot-nav

面向算法研究的机器人语义目标搜索项目。给定目标描述（如“门口”），算法会
扫描环境、定位可见目标；未发现目标时选择 Frontier 继续探索，并在走到尽头
时回到仍有候选方向的历史观测点。

## 数据流

```text
ChassisInterface ──► NavigationFrame ─────────────────────────┐
本地检测器 ────────► 目标框 + SAM2 掩码 ──────────────────────┼─► navigate()
VLM ───────────────► Frontier 分数 / 接近后的最终确认 ────────┘       │
                                                                    ▼
                                                         RelativePoseCommand
```

- `ChassisInterface` 统一仿真器和真底盘的数据与控制接口。
- `TargetObserver` 是感知边界；Hermes 模式下由 YOLO-World + SAM2 持续检测，
  VLM 只做整批 Frontier 评分和最终确认。
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
  激光栅格图，按 L515 水平 FOV 与实际深度视界累计已观察区域并按底盘外形膨胀
  障碍，再把相对位姿转换为底盘自主规划 Action；不可达 Frontier 会被淘汰后
  继续搜索。正式导航启动时先沿当前朝向规划前移 1 m，再开始首次扫描；探索或
  扫描 Action 期间若本地检测发现目标，会终止当前 Action 并从底盘实际位置
  重新决策。
- VLM 目标观察器：支持 Chat Completions、Responses 和 Anthropic Messages；
  单帧输出目标可见性或目标框，一轮扫描结束后把全部 Frontier 标在扫描 RGB
  拼图中，并用一次请求返回全部分数。
- Hermes 语义模式常驻 YOLOv8s-World：用目标类别检测候选，再用本地 SAM2
  验证并细化为掩码；最新帧队列在底盘运动和 VLM 等待期间都继续运行，目标
  距离只取掩码内深度。接近候选后由 VLM 做一次 `confirmed/rejected` 最终
  确认；被否决的位置会被屏蔽，随后恢复 Frontier 探索。
- 算法主流程：开始时连续转动 8 次、每次 45° 并观察，后续只用相机 FOV 覆盖
  当前 Frontier 聚类代表点，并在地图更新后跳过已经消失或变成障碍的聚类；同时
  支持目标掩码与深度定位、安全距离接近、Frontier 排序、观测历史和回退。
- Rerun 实时可视化：记录算法决策帧和底盘动作中间帧，包括 RGB、米制
  深度、叠加 Frontier/机器人/轨迹/命令的占用图，以及世界系扫描朝向、观测
  节点、候选方向虚线和角落状态摘要。界面固定为 RGB、Map、World 三个主视图
  和右侧调试标签页，后台模型结果只更新对应标签页，不重复推进导航时间轴或
  写入机器人轨迹；`model/interaction` 把每次 VLM 的
  完整提示词、实际图片、请求参数、原始回应和解析结果收纳到一张 CJK
  交互卡片，成功定位的目标框会叠加在卡片的输入 RGB 上；决策帧 RGB 使用
  洋红色半透明区域显示 SAM2 掩码，并保留绿色检测框；
  `model/sam2/latest_success` 持久保留最近一次成功分割。空间图形不附着文字标签。
- Hermes 导航自动保存 JSONL 运行日志，记录每周期决策、候选摘要、实际运动
  目标和 Action 位姿反馈，便于在程序退出后复盘异常移动。
- 通用入口默认使用 OpenCode Go 的 Qwen3.7 Plus，提示词为英文，并关闭模型
  thinking；优先读取 `ROBOT_NAV_VLM_API_KEY`，未设置时复用 OpenCode Go 本地
  登录凭据，不把 Key 写入仓库。
- Habitat 入口支持 `--debug-random-score`：不创建也不调用任何视觉模型，观察器
  对扫描帧返回 `NOT_VISIBLE`，再为整批 Frontier 生成随机分数，用于调试扫描、
  Frontier、移动和回退；此模式无法识别或到达语义目标，也不要求 Key。

算法流程见 [docs/algorithm.md](docs/algorithm.md)，环境接入见
[Habitat](docs/habitat.md)、[S100 + L515](docs/s100-l515.md) 和
[Hermes + L515](docs/slamtec-l515.md)。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/navigator.py` | 算法入口和阶段流转 |
| `src/robot_nav/core/frontier.py` | 可达 Frontier 提取与排序 |
| `src/robot_nav/core/grounding.py` | 用目标掩码、深度和完整相机外参估计目标位置 |
| `src/robot_nav/core/vision.py` | VLM 提示词和回答解析 |
| `src/robot_nav/core/history.py` | 探索方向历史与回退依据 |
| `src/robot_nav/core/models.py` | 全部输入、输出与状态契约 |
| `src/robot_nav/adapters/chassis.py` | 仿真器/真底盘共同接口 |
| `src/robot_nav/adapters/habitat/adapter.py` | Habitat-Sim Adapter |
| `src/robot_nav/adapters/s100_l515/adapter.py` | S100 + L515 真机 Adapter |
| `src/robot_nav/adapters/slamtec_l515/adapter.py` | Hermes + L515 真机 Adapter |
| `src/robot_nav/adapters/slamtec_l515/observed_map.py` | L515 FOV、障碍遮挡与 Hermes 障碍膨胀 |
| `src/robot_nav/adapters/slamtec_l515/rest_client.py` | Hermes REST 与 Action 边界 |
| `src/robot_nav/adapters/realsense/l515_camera.py` | 底盘无关的 L515 RGB-D 采集 |
| `src/robot_nav/adapters/realsense/calibration/` | 两种底盘共享的 L515 外参标定算法 |
| `src/robot_nav/adapters/s100_l515/ros_slam.py` | ROS SLAM 与统一导航帧的边界 |
| `src/robot_nav/adapters/perception.py` | 视觉/VLM 接口 |
| `src/robot_nav/adapters/openai_compatible.py` | OpenAI/Anthropic-compatible VLM 调用 |
| `src/robot_nav/adapters/yolo_world_sam2.py` | YOLO-World + SAM2 持续目标检测 |
| `src/robot_nav/adapters/sam2_observer.py` | 可复用的 SAM2 边界框分割器 |
| `src/robot_nav/adapters/frontier_overlay.py` | 构造带 Frontier/候选框标记的 VLM 输入图 |
| `src/robot_nav/adapters/random_observer.py` | 调试用 Frontier 批量随机评分观察器 |
| `src/robot_nav/visualization/rerun_view.py` | Rerun 导航调试界面 |
| `src/robot_nav/run_log.py` | 不含图像和完整地图的 JSONL 运行日志 |
| `src/robot_nav/app.py` | 单周期串联入口 |
| `src/robot_nav/__main__.py` | Adapter 选择、循环和终端输出 |
| `slam/s100_l515/` | S100 + L515 的 ROS 环境、启动与 SLAM 参数 |
| `hardware/realsense/` | WSL L515 USB 权限与 RSUSB 用户态后端 |
| `hardware/slamtec_l515/` | Hermes + L515 的真机启动脚本 |

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
