# 算法说明

## 主流程

阅读代码从 `core/navigator.py` 的 `navigate()` 开始。每次调用只推进一个周期，
跨周期信息全部保存在 `SearchState`。

```text
SCANNING
  ├─ 目标可见 ──► LOCALIZING_TARGET ──► 接近并重新观察 ──► COMPLETE
  └─ 四向均不可见
         ↓
     提取并排序可达 Frontier
         ├─ 有候选 ──► 记录观测节点 ──► 移动 ──► SCANNING
         └─ 无候选 ──► BACKTRACKING
                              ├─ 找到历史待探索方向 ──► 移动 ──► SCANNING
                              └─ 搜索空间耗尽 ──► FAILED
```

## 各步骤输入输出

| 步骤 | 输入 | 输出 | 代码位置 |
| --- | --- | --- | --- |
| 目标观察 | RGB 与目标描述 | 可见性、方向评分、目标框 | `core/vision.py`、`adapters/openai_compatible.py` |
| 四向扫描 | 当前 yaw 与逐帧观测 | 四个世界航向及其证据 | `core/navigator.py`、`core/scan.py` |
| 目标定位 | 目标框、对齐深度、相机标定、位姿 | 目标的机器人系/世界系坐标 | `core/grounding.py` |
| Frontier | 局部占用图、位姿、首选方向 | 已排序的可达候选 | `core/frontier.py` |
| 回退 | 观测节点与方向状态 | 最近的待探索方向 | `core/history.py`、`core/navigator.py` |

目标可见时，算法使用目标框中央区域的近端深度中值估计位置，并移动到距目标
约 0.75 m 的观察位置；到达后必须用新帧重新观察。目标不可见时，四个方向的
视觉评分只影响 Frontier 排序，不能绕过地图可达性判断。

目标观察器先用完整 RGB 判断目标是否可见。目标不可见时再请求当前方向的探索
评分；评分失败不会阻塞基于地图的 Frontier 探索。目标可见时再请求千分制目标
框并转换为归一化坐标；没有可靠目标框时返回 `UNCERTAIN`，算法不会移动。

`navigate()` 需要视觉输入时返回 `NEEDS_OBSERVATION`；`run_navigation_cycle()`
调用 `TargetObserver` 后用同一帧再次推进算法。目标可见但缺少目标框、深度或
相机内参时返回 `MISSING_DATA`。这些状态描述所缺输入，不代表算法尚未实现。

选择 Frontier 前会把当前位置及全部候选冻结为一个观测节点：本次选择标为
`COMMITTED`，其余标为 `PENDING`。新位置没有可用 Frontier 时，本次方向变为
`EXPLORED`，算法回到最近仍含 `PENDING` 方向的节点；新地图已判定为障碍的
历史候选会变为 `INVALIDATED`。

Rerun 会同时显示全部历史观测节点中的候选，因此运行越久，画面中的候选点可能
越多；它们不等于当前一轮新生成的候选。启动时增加 `--debug-frontier`，可在
发送移动命令前打印当前一轮候选的栅格坐标、世界坐标、前沿长度、路径距离、
方向奖励和最终分数。

## 当前边界

- 已提供 OpenAI-compatible Chat Completions / Responses 观察器，但 API 地址、
  Key 和模型名由调用方在运行时配置。没有视觉结果、目标框或可靠深度时，算法
  会停止并给出明确原因。
- 控制输出是高层相对位姿，不包含速度控制和实时避障；路径规划、执行完成和
  失败反馈由 Adapter 后面的仿真器或底盘负责。
- 当前只有单次 HTTP 请求超时，没有模型重试、算法级超时策略、动态障碍预测和
  真实运动学。这些应在实际出现需求后补充，不进入第一版算法主干。

所有距离使用米，角度使用弧度且逆时针为正。坐标与地图契约见
[chassis-interface.md](chassis-interface.md)。
