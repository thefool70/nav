# 算法说明

## 主流程

阅读代码从 `core/navigator.py` 的 `navigate()` 开始。每次调用只推进一个周期，
跨周期信息全部保存在 `SearchState`。Hermes 真机入口在进入该状态机前先沿底盘
当前朝向规划前移 1 m；这是设备启动编排，不属于环境无关的导航核心。

```text
SCANNING
  ├─ 本地检测可见 ──► LOCALIZING_TARGET ──► 接近并重新检测
  │                                              ↓
  │                                      VERIFYING_TARGET
  │                                        ├─ VLM 确认 ──► COMPLETE
  │                                        └─ VLM 否决 ──► 屏蔽位置并继续探索
  └─ 本轮全部视角均不可见
         ↓
     提取可达 Frontier
         ↓
     标在扫描 RGB 拼图中，一次请求全部语义分数
         ↓
     几何与语义综合排序
         ├─ 有候选 ──► 记录观测节点 ──► 移动 ──► SCANNING
         └─ 无候选 ──► BACKTRACKING
                              ├─ 找到历史待探索方向 ──► 移动 ──► SCANNING
                              └─ 搜索空间耗尽 ──► FAILED
```

## 各步骤输入输出

| 步骤 | 输入 | 输出 | 代码位置 |
| --- | --- | --- | --- |
| 持续目标检测 | 单帧 RGB 与 YOLO 类别 | 可见性、候选框、SAM2 掩码 | `adapters/yolo_world_sam2.py`、`adapters/sam2_observer.py` |
| 最终目标确认 | 接近候选后的 RGB 与目标描述 | `confirmed` 或 `rejected` | `core/vision.py`、`adapters/openai_compatible.py` |
| 环境扫描 | 当前 yaw、相机水平 FOV、Frontier 与逐帧观测 | 必要的世界航向及其证据 | `core/navigator.py`、`core/scan.py` |
| 目标定位 | 目标掩码、对齐深度、相机标定、位姿 | 目标的机器人系/世界系坐标 | `core/grounding.py` |
| Frontier 评分 | 扫描 RGB 与全部可达 Frontier | Frontier ID 到 0-1 分数 | `adapters/frontier_overlay.py`、`adapters/openai_compatible.py` |
| Frontier 排序 | 当前有效占用图、位姿、语义分数 | 已排序的可达未知边界 | `core/frontier.py` |
| 回退 | 观测节点与方向状态 | 最近的待探索方向 | `core/history.py`、`core/navigator.py` |

Hermes 语义模式中，YOLO-World 对当前决策帧和导航全程采集的最新帧持续检测
目标；即使主循环正在等待 VLM 评分或最终确认，本地检测也不会暂停。SAM2 在
同一 RGB 上验证候选框并生成掩码。YOLO 没有候选或所有候选都无法生成非空掩码
时按 `NOT_VISIBLE` 处理；模型、图像或推理异常才返回 `UNCERTAIN`。后台采用
最新帧队列，旧帧可丢弃，避免推理落后于机器人。探索和扫描期间发现候选会终止
当前 Hermes Action，并在下一周期从实际位置用新帧重新定位；目标接近 Action
不因重复看到同一候选而反复中断，到达后再判断。

掩码有效后算法立即进入目标接近模式，深度只取自掩码像素；只要存在一个有效
深度点，就按当前估计位置一次性移动到距目标约 0.75 m 的观察位置，不限制单次
接近距离。若掩码内完全没有有效深度，则把掩码质心的深度假定为有效上界 5 m
并继续接近；这是一项主动探索假设，不代表 L515 实际测得了 5 m。进入观察距离
后，算法把本地候选框以绿色边框标在完整 RGB 上，交给 VLM 做一次二元最终
确认。确认后完成搜索；否决
后记录候选世界位置并恢复 Frontier 探索，之后落在该位置邻域的检测不再接近。
目标接近动作被拒绝或连续静止达到门槛时，算法保留底盘实际位置，下一周期重新
检测，不会因此直接结束导航。程序开始时连续转动 8 次，
每次 45° 并在转向后观察。此后每次移动只扫描当前有效 Frontier 聚类的代表点，
并依据相机内参、图像宽度和安装 yaw 规划覆盖全部代表点所需的最少视角。每次
执行下一个视角前都会重新提取当前 Frontier：已被本轮视场覆盖，或因后续地图
更新而消失、合并、变成障碍的聚类不再触发旋转和本地观测；新出现且尚未覆盖
的聚类会进入刷新后的扫描计划。没有 Frontier 时直接进入回退。

扫描结束后，感知层把全部候选按水平方位编号到对应 RGB 视角，再拼为一张图片，
一次性请求 VLM 返回所有 0-1 分数且不生成理由。综合分数为“Frontier 长度
- 0.05×路径距离 + 0.75×(2×语义分数-1)”。模型只改变可达候选的排序，不能
绕过地图可达性和底盘规划器；请求或解析失败时退回纯几何排序。

Frontier 搜索从机器人所在自由格进行四邻接 BFS，遍历当前有效地图中全部连通
自由格；只有八邻域接触未知格的可达自由格才是 Frontier。算法不再把固定路径
距离边界伪造为 Frontier。每个聚类的实际移动点仍是该聚类中的 Frontier 自由格：
先在 0.75 m 局部范围内选择离占据格最远的格，再用聚类质心和 BFS 距离打破
并列；不会沿 BFS 路径额外退让。Hermes + L515 模式由 Adapter 的理论水平 FOV
和最远 5 m 范围产生候选视野，再用 L515 对应图像列附近的最远有效深度截断
真正被遮挡的区域。低矮或局部障碍后方仍有深度射线时不会挡住整个方向；邻近列
没有有效深度时保留理论 FOV。底盘保存的完整地图仍保持不变。

Hermes 模式不再逐帧请求 VLM 判断可见性或框选目标。目标框来自 YOLO-World，
掩码来自 SAM2；VLM 只在一轮扫描结束后批量评分 Frontier，以及候选接近后做
最终确认。其他 Adapter 仍可使用通用 VLM 单帧观察器。

`navigate()` 需要单帧视觉输入时返回 `NEEDS_OBSERVATION`，需要整批评分时返回
`NEEDS_FRONTIER_SCORES`，接近候选后返回 `NEEDS_TARGET_CONFIRMATION`；
`run_navigation_cycle()` 调用 `TargetObserver` 后用同一帧和显式结果再次推进
算法。目标可见但缺少目标框、深度或相机内参时返回
`MISSING_DATA`。一次感知结果可能使状态机立即请求下一帧观测，例如目标丢失后
重新进入扫描；命令行循环会继续读取下一帧，不把这些等待输入状态当成失败。
这些状态描述所缺输入，不代表算法尚未实现。

选择 Frontier 前会把当前位置及全部候选冻结为一个观测节点：本次选择标为
`COMMITTED`，其余标为 `PENDING`。新位置没有可用 Frontier 时，本次方向变为
`EXPLORED`，算法回到最近仍含 `PENDING` 方向的节点；新地图已判定为障碍的
历史候选会变为 `INVALIDATED`。如果底盘规划器明确拒绝 Frontier，或者移动
Action 失败，该方向也会变为 `INVALIDATED`，算法从同一节点继续尝试其他
`PENDING` 方向。Hermes 探索移动连续静止 15 秒是另一种结果：Action 会被终止，
但算法把实际当前位置当作本次移动终点，下一周期直接重新扫描，不立即返回旧节点。
运动期间本地检测触发的中断也不是导航失败：算法保留 SearchState，下一周期从
底盘实际位置处理最新目标观测；尚未到达的 Frontier 方向恢复为 `PENDING`，
不会被误记为已经探索。

Rerun 会把当前 Frontier 直接叠加在占据图中：Frontier 为绿色，最终选中的代表
点为黄色；机器人使用朝向三角形表示，同一地图还显示轨迹和命令目标。世界视图
显示当前 Frontier、计划扫描朝向、全部历史观测节点和候选点，并用与方向状态
同色的虚线连接节点和候选。空间图形不显示附着标签，位姿、扫描和命令摘要集中
在 world 左上角，完整信息保留在 `navigation/status` 面板。Rerun 使用固定布局：
RGB、Map 和 World 始终作为主视图，状态、本地检测、SAM2、VLM 和深度位于右侧
标签页。后台推理完成时只更新模型标签页，不重新记录来源帧的地图、位姿和轨迹，
因此延迟结果不会让导航时间轴倒退。启动时增加
`--debug-frontier`，可在发送移动命令前打印当前候选的栅格坐标、世界坐标、
前沿长度、路径距离、VLM 分数、语义奖励和最终分数。Rerun 的
`model/interaction` 在时间轴上用一张 CJK 交互卡片保留每次 VLM 的实际
RGB、完整提示词与请求参数、assistant 文本、完整 HTTP JSON、解析结果和
错误；目标框解析成功时同时叠加显示。导航决策帧的 `camera/rgb` 用洋红色
半透明区域显示 SAM2 掩码，绿色框表示 YOLO-World 候选；
`model/sam2/latest_success` 会跨越后续运动帧保留最近一次成功分割，
`model/sam2/status` 显示当前观测是否取得掩码；`model/yolo_world/status` 显示
运动期间每次已处理帧的候选数、置信度、耗时和失败原因。

## 当前边界

- 已提供 Chat Completions、Responses 和 Anthropic Messages 观察器。默认英文
  提示词只要求结构化结果，不要求理由。Hermes 模式的目标检测依赖本地
  YOLOv8s-World 与 SAM2；没有候选框或掩码时不会进入目标接近，掩码内没有有效
  深度时按掩码质心 5 m 的假定距离接近并重新检测。
- 控制输出是高层相对位姿，不包含速度控制和实时避障；路径规划、执行完成和
  失败反馈由 Adapter 后面的仿真器或底盘负责。
- 当前 VLM 只有单次 HTTP 请求超时，本地决策帧也只有单次推理等待上限；尚无
  通用模型重试、算法级超时策略、动态障碍预测和真实运动学。

所有距离使用米，角度使用弧度且逆时针为正。坐标与地图契约见
[chassis-interface.md](chassis-interface.md)。
