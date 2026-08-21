# 算法说明

## 目标

给定对目标的人类可读描述（`TargetSearchGoal.target_text`，如 "门口"），机器人原地四向扫描感知环境、记录可探索方向并移动探索，直到定位目标或判定无法完成。

## 流程

```text
navigate(frame, goal, state)
 ├─ 校验输入 ──非法──► INVALID_INPUT（state 置 FAILED）
 └─ 合法
     ├─ 无扫描朝向 ──► 以当前 yaw 初始化四向扫描
     ├─ 四向扫描，每方向先转向、取得新观测、判断目标    （视觉观察，未迁入）
     │    ├─ 可见 ──► 粗框/精框 ──► 深度定位 ──► 安全接近 ──► 到达后重新确认 ──► COMPLETE
     │    │              （定位与接近闭环，未迁入）
     │    └─ 不可见 ──► 记录方向评分
     ├─ 四向都不可见后 ──► 提取并排序 Frontier          （Frontier 闭环，未迁入）
     │    ├─ 选定候选，出发前才冻结观测历史节点
     │    ├─ 移动到候选后以新地图判断新增 Frontier
     │    └─ 无新增 ──► 回退历史                        （移动/回退闭环，未迁入）
     └─ 输出新 state；合法输入返回 NOT_IMPLEMENTED、非法输入返回 INVALID_INPUT，均不发命令
```

## 模块到排错问题的映射

| 问题现象 | 排查模块 |
| --- | --- |
| 输入被拒但理由不清 | `core/navigator.py` 的输入校验 |
| 扫描朝向 / 最短转角不对 | `core/scan.py` |
| 世界 / 机器人 / 栅格坐标换算错误 | `core/geometry.py` |
| 观测顺序、方向状态错乱 | `core/history.py` |
| 周期状态丢失或错位 | `core/models.py` 的 `SearchState` 与调用方状态传递 |
| 周期不推进、不发命令 | `core/navigator.py`（视觉未迁入，合法输入返回 NOT_IMPLEMENTED） |

## 关键输入输出

- 输入：`NavigationFrame`（时间戳、位姿、障碍图、可选深度与 RGB）、
  `TargetSearchGoal`、上周期 `SearchState`。
- 输出：`NavigationResult`（status、command、debug、显式 `SearchState`）；
  下一周期把 `result.state` 作为 `navigate` 的输入。
- 约定：长度单位米、角度弧度（逆时针为正）；yaw 与栅格映射的内部标准见
  [chassis-interface.md](chassis-interface.md)。

## 当前明确的未实现边界

- 视觉观察、深度定位、Frontier 提取与排序、移动闭环均未迁入，`navigate`
  合法输入返回 `NOT_IMPLEMENTED`、非法输入返回 `INVALID_INPUT`，两种情况均
  `command=None`，不发送控制命令。
- 控制命令不包含行走规划：相对位姿命令是否自带路径规划 / 避障由底盘决定。
