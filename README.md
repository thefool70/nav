# robot-nav

机器人导航算法的**最小 Python 框架**，正在实现**语义目标搜索**：给定对目标的人类可读描述（如 "门口"），机器人扫描环境、记录可探索方向并逐步探索，直到定位目标。

## 目标与边界

- 运行时只用 Python 标准库（Python 3.11），无第三方运行依赖。
- 核心算法（`core`）不依赖 ROS、底盘 SDK、文件系统或适配层。
- 当前已具备：显式 `SearchState` 周期状态、四向均匀扫描几何、世界/机器人/栅格坐标转换、探索方向历史。
- 尚未迁入：视觉观察、深度定位、Frontier 提取与排序、移动闭环。因此 `navigate` 合法输入返回 `NOT_IMPLEMENTED`、非法输入返回 `INVALID_INPUT`，两种情况均 `command=None`，**不发送任何控制命令**。
- 厂商原始协议不在核心中猜测，由适配层负责转换。

## 数据流

```text
ChassisInterface.read_frame()  →  NavigationFrame（时间戳/位姿/障碍图/可选深度与RGB）
                                        │
                                        ▼
            core.navigator.navigate(frame, goal, state)  →  NavigationResult
                                        │                 (status + command + debug + state)
                   仅当 status==OK 且 command 非 None
                                        ▼
             ChassisInterface.send_relative_pose(command)
```

`run_navigation_cycle` 完成「读取一帧 → 导航 → 条件发送命令」这一个周期，不包含循环或频率假设。搜索状态显式保存在 `result.state`（`SearchState`：阶段、待扫描朝向序列与下标、观测历史），调用方将其作为下一周期的 `navigate` 输入。

目标以 `TargetSearchGoal(target_text)` 表达，`target_text` 为对目标的人类可读描述。算法整体说明见 [docs/algorithm.md](docs/algorithm.md)。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/models.py` | 内部数据契约（Pose2D、ObstacleMap、NavigationFrame、TargetSearchGoal、SearchState 等） |
| `src/robot_nav/core/navigator.py` | 导航主入口 `navigate(frame, goal, state)`，输入校验与扫描状态初始化 |
| `src/robot_nav/core/scan.py` | 均匀扫描朝向与最短转角计算 |
| `src/robot_nav/core/geometry.py` | 世界/机器人/栅格坐标转换与角度归一化 |
| `src/robot_nav/core/history.py` | 观测节点与探索方向历史管理 |
| `src/robot_nav/adapters/chassis.py` | 最薄的底盘接口协议 `ChassisInterface` |
| `src/robot_nav/app.py` | 单周期串联入口 `run_navigation_cycle` |

## Python 环境

用 `environment.yml` 创建名为 `robot-nav` 的 Micromamba/Conda 环境（仅 conda-forge 通道，只含 Python 3.11 与 setuptools）：

```bash
micromamba env create -f environment.yml
micromamba activate robot-nav
python -m pip install -e .
```

环境已存在时，用以下命令按文件内容增量更新：

```bash
micromamba env update -n robot-nav -f environment.yml --prune
```

## 接入底盘

底盘到货后，在 `adapters/` 中实现 `ChassisInterface` 的 `read_frame` 与
`send_relative_pose`，负责把厂商协议转换为内部标准；具体待确认项见
[docs/chassis-interface.md](docs/chassis-interface.md)。
