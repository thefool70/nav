# robot-nav

机器人导航算法的**最小 Python 框架**。目标是让算法结构清晰、可读、便于沿数据流定位问题。当前底盘尚未到货，本仓库只搭框架，**不实现具体导航算法**，也**不会输出任何控制命令**。

## 目标与边界

- 运行时只用 Python 标准库（Python 3.11），无第三方运行依赖。
- 核心算法（`core`）不依赖 ROS、底盘 SDK、文件系统或适配层。
- 契约：`NavigationFrame.pose` 与 `NavigationGoal.target` 在进入 core 前必须已转换到 `NavigationFrame.obstacle_map.frame_id`，core 内部不做坐标转换。
- 当前 `navigate` 固定返回 `NOT_IMPLEMENTED`，`command=None`，仅提供占位与可读 debug 信息（含 `stage` 字段）。
- 厂商原始协议不在核心中猜测，由适配层负责转换。

## 数据流

```text
ChassisInterface.read_frame()  →  NavigationFrame（时间戳/位姿/障碍图/可选深度与RGB）
                                        │
                                        ▼
                    core.navigator.navigate(frame, goal)  →  NavigationResult
                                        │                 (status + command + debug)
                   仅当 status==OK 且 command 非 None
                                        ▼
             ChassisInterface.send_relative_pose(command)
```

顶层 `run_navigation_cycle` 完成「读取一帧 → 导航 → 条件发送命令」这一个周期，不包含循环或频率假设。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/core/models.py` | 内部数据契约（Pose2D、ObstacleMap、NavigationFrame、RelativePoseCommand 等） |
| `src/robot_nav/core/navigator.py` | 导航主入口 `navigate(frame, goal)`，当前为占位 |
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
