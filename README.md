# robot-nav

面向算法研究的机器人语义搜索项目。算法接收位姿、RGB-D 和占用图，输出相对
位姿命令；仿真器和真底盘的差异全部收在 Adapter 中。

当前支持两种任务：

- `object`：寻找一个具体物体。
- `scene`：寻找一个目的场景，例如洗手间或电梯厅。

## 从哪里开始读

建议按下面的顺序阅读：

1. `src/robot_nav/__main__.py`：启动分派；参数定义见 `cli.py`。
2. `src/robot_nav/launch.py`：两种环境共用的组件装配；环境创建与准备见 `environment.py`。
3. `src/robot_nav/app.py`：完整运行循环，以及单周期的输入、感知、决策和运动。
4. `src/robot_nav/core/navigator.py`：行为分派与公共输入检查，四类行为的入口。
5. `src/robot_nav/core/models.py`：算法输入、输出和跨周期状态。
6. 当前使用的 Adapter：Habitat 或 Hermes + D435i。

启动、配置和主要算法模块按“入口 → 主要步骤 → 内部辅助实现”阅读；
Adapter 先列公共操作，再列内部实现。数据类型仍先于使用它们的函数定义。`launch.py` 的
`_assemble_and_run` 展示组件连接，`app.py` 从 `run_navigation` 总循环读到
`run_navigation_cycle` 单周期；计时与显示细节集中在 `runtime_reporting.py`。
函数前缀 `_` 表示模块内部接口，不表示它是否属于算法计算。
行为入口保留重要分支和恢复规则，计算函数直接表达具体算法；不为未使用的
调用方式维护通用选项或转发接口。深度定位只保留当前导航使用的正深度采样与
中位数估计，背景过滤和假定距离回退不再作为 Python 参数提供。

算法细节见 [算法说明](docs/algorithm.md)，坐标和接口约定见
[底盘接口标准](docs/chassis-interface.md)。

## 架构

```text
设备 / 仿真器
    │
    ▼
ChassisInterface ──► NavigationFrame
                           │
    SemanticPerception ───►感知结果 / 语义判定 / 历史定位
                           │
                           ▼
                run_navigation_cycle()      （app.py：单周期编排）
                           │
                           ▼
                      navigate()            （core/navigator.py：行为分派）
                           │
                           ▼
                  NavigationAction          （显式动作类型 + 约束）
                           │
                           ▼
                  ChassisInterface
```

- `core/` 只做算法计算，不读取设备、不请求模型、不发送控制命令。
- `app.py` 负责导航循环与单周期编排，并把可恢复的运动结果送回状态机。
- 搜索核心按四类行为组织：`core/scan_behavior.py`（观察方向规划与采集）、
  `core/exploration.py`（Frontier 探索）、`core/backtracking.py`（分支回退）、
  目标处理（`core/scene_target.py`、`core/target_clue.py`、`core/object_approach.py`）。
- `perception/` 组织取帧、视觉队列、语义判定与历史物体定位；不直接控制底盘，
  也不修改搜索状态。
- `adapters/` 负责设备协议、坐标转换、地图归一化和视觉模型请求。
- `visualization/` 与 `run_log.py` 只记录过程，不参与决策。
  `rerun_view.py` 组织 Rerun 记录，`panels.py` 生成状态文本与模型卡片，
  `view_geometry.py` 计算显示坐标、机器人轮廓与路径线段。

## 搜索方式

| 模式 | 发现目标 | VLM 的作用 | 后续行为 |
| --- | --- | --- | --- |
| 物体搜索 | 后台检查前沿扫描采集的固定画面 | 联合检测与评分；接近时可用本地模型提供目标框 | 历史 RGB-D 优先定位，失败后尝试障碍射线；得到位置后接近 |
| 场景搜索 | 后台检查前沿扫描采集的固定画面 | 同一次请求判断场景与评分 Frontier | 返回拍摄位置并对齐朝向，搜索完成 |

两种模式都在当前可达自由区内选择 Frontier，一次移动到选定位置，到位后
再观察和选择下一目标。新出现的 Frontier 优先探索，未选方向暂存；新候选耗尽
后，沿当前分支逐个返回父节点，直到到达仍有有效探索方向的节点，再继续寻找。
启动与后续扫描共用同一规则，只观察局部可见且尚未检查的 Frontier 方向。
返回父节点后直接检查剩余探索方向，不额外扫描。
扫描使用移动筛选前的完整边界；边界过近、跨度不足或移动尝试被排除，都不影响
原地观察该方向。观察范围、地图遮挡与已有视觉覆盖仍决定是否需要补查。
地图已知与视觉已检查分别记录；两种模式没有待查方向时仍采集当前画面，不额外转向。
返回节点未完成时保留实际位置，跳过该返回节点并重新检查有效方向，不直接结束搜索。

命令行默认使用异步 FIFO 视觉队列。每轮扫描收齐后，将全部画面拼成一张图，
在同一次请求中提取目标线索并评分 Frontier。扫描转向不额外提交单图，探索和分支回退的
模型任务只来自前沿扫描，平移动作途中不再选帧入队。缺少语义分时按几何分继续走，评分只在下一次决策生效。
后台检测到目标后按模型顺序处理线索。场景模式返回拍摄位置并对齐朝向后完成，
不进行到场视觉复查。物体模式先在历史 RGB-D 上定位，YOLO 或 VLM 任一路
检出即可使用；优先用 SAM2 掩码，分割失败时直接用检测框内深度。无法用深度定位时，
有框就沿框中心方向查询地图中的首个障碍，无框则沿相机光轴，将障碍表面作为目标位置假设。
得到位置后，在目标周围搜索可达自由格，从当前位置一次前往停靠点并对准目标，
停靠命令执行成功后直接完成搜索。
历史定位失败时留在当前位置，继续下一条线索，并处理已采集但尚未分析的画面。
全部历史线索都无法定位时，才保底返回首条线索的拍摄位姿并停止，不再重采或继续探索。
实际运动失败时可换点，每条已定位线索最多尝试三次停靠；均失败且历史队列耗尽后恢复探索。
当前线索定位和接近期间暂停普通队列，迟到结果暂存；需要更多历史线索时
只恢复已有队列的分析，不为此新增运动。
有效探索方向耗尽后先等待队列，模型失败不冒充“已经检查”。

Hermes 与 Habitat 执行 Frontier 移动时，还会按选点时的算法地图检查实际路径。路径经过
未知区的累计长度超过 1.5 m 才取消动作，在本次运行中持续禁止向整个连通 Frontier 区域探索移动并转向其他候选，
避免在同一片边界内换点反复取消。

Hermes 平移收到新的路径受阻事件后等待恢复；超过配置时长仍未恢复有效平移，
取消并确认停止。Frontier 探索因此结束时，本次运行屏蔽选中的整个区域；
短暂等待后恢复移动则继续原任务。具体门槛与事件采集见 Hermes 文档。

两种环境只在 `environment.py` 中创建各自 Adapter；感知、日志、回调和导航循环
统一在 `launch.py` 装配。只读预检独立运行，启动前移仅用于真机。

内部模块直接使用 `NavigationFrame`、`SearchState` 等明确的数据契约，不逐层重复
验证对象类型或尝试转换任意输入。通信、配置和模型响应在边界处理；深度缺测、
未知地图与不可达目标仍是正常的算法分支。内部程序错误向上传播，后台感知错误
在主循环取结果时重新抛出，不伪装成“未检出目标”。

## 运行环境

| 环境 | 用途 | 文档 |
| --- | --- | --- |
| `robot-nav` | 核心代码、Hermes + D435i | [Hermes + D435i](docs/hermes.md) |
| `robot-nav-habitat` | Habitat-Sim 仿真 | [Habitat](docs/habitat.md) |

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

底盘手动操作面板：`python -m robot_nav.chassis_gui`，浏览器打开
`http://127.0.0.1:8088`。支持状态查看、小步移动、转向、回桩和取消任务；
连接方式与限制见 [Hermes 文档](docs/hermes.md#底盘操作面板)。

## 统一运行配置

项目根目录的 `config.json` 集中保存导航的常用参数，按 navigation、habitat、
hermes、camera、perception、logging 分组。默认读取当前目录下的该文件；
切换工作目录时通过 `--config /绝对路径/config.json` 指定，配置文件必须完整。
每组的 `_comments` 保存参数中文说明，加载时忽略；文件保持标准 JSON 格式。
命令行显式参数优先于文件，配置中的相对文件路径以配置文件所在目录为基准，
命令行相对路径仍以当前工作目录为基准。

```bash
python -m robot_nav habitat --target "chair"
python -m robot_nav --config config.json hermes --preflight-only
python -m robot_nav hermes --config config.json --target "chair" --enable-motion
```

随车笔记本方式将 `hermes.base_url` 改为 `http://127.0.0.1:11448`，
`camera.camera_source` 改为 `remote`；相机 IPC 地址已在 camera 组中。
远程主题为 `rgbd.pose`，接收随车发布器打包的 RGB-D 与同步底盘位姿。
仓库默认仍保持本地 USB 与底盘直连，不因加载配置自动改变连接方式。

`navigation.target` 的 null 表示本次需要指定目标；`perception.object_python` 的 null
表示自动查找已有 robot-nav 模型环境。`hermes.startup_forward_m` 默认 1 米，设为 0
可跳过真机启动前移。固定外参保存在 camera 组已有的六个高度、偏移和角度字段中；
远程方式先用 `hardware/hermes/fetch_camera_extrinsics.py` 获取一次，导航启动时读取。
本地 USB 未配置齐六项时仍可用 `camera.camera_calibration` 指定外参文件。
密钥继续通过环境变量或已有凭据读取。
`navigation.max_unknown_path_m` 是两种环境共用的未知路径长度上限（默认 1.5 米），
从原来的 `hermes.max_unknown_path_m` 移到此处；自定义配置文件也需同步移动该字段。
`--enable-motion`、`--preflight-only`、`--base-only` 只接受命令行设置，不能写入配置。

`logging.rerun_viewer` 选择网页 `web` 或桌面 App `native`；可用
`--rerun-viewer native` 临时覆盖，录制方式不变。

配置布尔值可临时覆盖：`--rerun` / `--no-rerun`、`--debug-random-score` /
`--no-debug-random-score`、`--debug-frontier` / `--no-debug-frontier`。
未知字段、重复字段、缺少字段或非法值会在装配组件前报错，避免配置拼错后悄悄使用默认值。

`config.py` 校验文件字段与类型，`cli.py` 合并命令行覆盖并集中检查参数组合；
`__main__.py` 按需读取凭据和分派入口，`launch.py` 使用已校验参数装配公共组件。
真机运动授权和设备准备条件由 `environment.py` 检查。
各 Config 保留独立调用时的缺省值；算法内部常量、SSH 脚本环境变量与独立底盘 GUI
参数不由这个运行配置文件接管。

## 运行命令

```bash
# Habitat 仿真
python -m robot_nav habitat --scene sim/habitat/scene.glb --target "chair"

# Hermes + D435i 真机（本地 USB 相机）
python -m robot_nav hermes --target "chair" --enable-motion

# 随车笔记本转发（先启动相机服务和 SSH 隧道，见 Hermes 文档）
python -m robot_nav hermes --camera-source remote \
  --camera-endpoint ipc:///tmp/robot-nav-camera.sock --base-url http://127.0.0.1:11448 \
  --target "chair" --enable-motion

# 只读预检，不发送运动命令
python -m robot_nav hermes --preflight-only
```

## 当前实现

- Habitat：提供 RGB-D、二维位姿和逐步公开的占用图；相对目标由 navmesh 规划并
  离散执行。
- Hermes + D435i：Hermes 提供位姿、激光地图和自主规划；D435i 提供 RGB-D，
  探索地图只在当前理论水平 FOV 内刷新，视场外保留历史值，不考虑遮挡；
  启动区域仅初始化一次。物体障碍定位和停靠使用完整导航图，选点时保留 0.36 m 净空。
  观测方向和覆盖使用同一 FOV 缓存的未膨胀视觉图，避免把导航净空带误判为遮挡。
  该视觉图中的封闭小未知孔洞不产生探索与补查候选，地图本身仍保留未知状态。
  实际位姿到达并稳定后交回决策，下一 Action 直接替换旧任务；无下一动作及退出时取消收尾。
  本地直连与经随车笔记本转发共用同一套地图处理、运动监控与导航实现；
  随车发布器完成 RGB-D 与底盘位姿的时间对齐；相机安装外参启动时从配置读取。
  算法、地图处理和动作控制仍在开发机。地图和实时运动反馈继续读取 Hermes REST。
- Rerun：显示 RGB、深度、地图、Frontier、机器人轨迹、算法目标、底盘目标和
  规划路径，同时持续写入 `data/run_logs/rerun-*.rrd`。`--rerun-save <PATH>`
  可指定新文件路径；`--no-rerun` 同时关闭界面和录制。
  `VLM summary` 简要关联任务、请求与导航使用；`VLM full` 保留完整会话信息。
  World 将占用图、机器人和任务标记放在一起，同次扫描只画一个点，邻近任务合并显示。
  扫描期间显示本轮全部计划朝向，并用箭头突出当前待执行方向；扫描计划结束后清除。
  右侧直接展示当前推理的 RGB 与评分；`Observations` 中点击 J 查看整组、V 查看
  单图评分卡，原图通过 raw 链接查看。完整点位保留在 `World history`。
- 导航 JSONL 日志（Hermes / Habitat）：记录每周期决策、候选评分和 Action 反馈，供事后复盘。
- 视觉快照：`data/run_logs/semantic-*/job-*` 保存图片、位姿、候选和分析结果；
  普通任务按 FIFO 处理。目录在退出后保留，当前不自动恢复上次队列。
  同帧深度先临时压缩保存，分析后只保留命中画面的正式深度文件；检测失败或
  未完成的任务保留临时深度，等待明确判断。

## 主要目录

| 路径 | 职责 |
| --- | --- |
| `src/robot_nav/__main__.py`、`cli.py` | 启动分派、参数定义与组合校验 |
| `src/robot_nav/launch.py` | 两种环境共用的组件装配、日志与可视化接线 |
| `src/robot_nav/environment.py` | Adapter 创建、真机预检与启动前移 |
| `src/robot_nav/app.py` | 完整导航循环、单周期编排与显式动作执行 |
| `src/robot_nav/core/navigator.py` | 行为分派与公共输入检查 |
| `src/robot_nav/core/scan_behavior.py` | Frontier 观察方向规划、补扫与画面采集 |
| `src/robot_nav/core/exploration.py` | Frontier 选点、提交、淘汰与暂存方向恢复 |
| `src/robot_nav/core/backtracking.py` | 分支节点回退与到点后恢复方向 |
| `src/robot_nav/core/scene_target.py` | 场景线索返回拍摄位姿并完成 |
| `src/robot_nav/core/target_clue.py` | 目标线索分派与丢弃 |
| `src/robot_nav/core/object_approach.py` | 历史线索处理、停靠完成与保底返回停止 |
| `src/robot_nav/core/frontier_regions.py` | Frontier 区域刷新、候选预览与区域屏蔽 |
| `src/robot_nav/core/perception_flow.py` | 感知增量归并、采样上下文与目标处理状态 |
| `src/robot_nav/core/actions.py` | 执行与日志共用的目标位姿转换 |
| `src/robot_nav/runtime_reporting.py` | 周期日志回调与终端摘要 |
| `src/robot_nav/core/navigation_io.py` | 动作构造与状态/边界校验的公共输入检查 |
| `src/robot_nav/core/observation_coverage.py` | 局部 Frontier 观察点、RGB-D 覆盖记录和跨位置复用 |
| `src/robot_nav/core/frontier_projection.py` | Frontier 地面点投影与对齐深度核对，供 VLM 图片标注 |
| `src/robot_nav/core/path_validation.py` | 测量实际规划路径在算法未知区内的累计长度 |
| `src/robot_nav/core/object_grounding.py` | RGB-D 定位与图像方向上的障碍位置假设 |
| `src/robot_nav/core/object_standoff.py` | 在目标周围搜索满足净空与连通条件的停靠点 |
| `src/robot_nav/perception/semantic_queue.py` | 扫描快照、FIFO 联合分析与迟到结果接收 |
| `src/robot_nav/perception/analyzer.py` | 语义分析的模型边界协议 |
| `src/robot_nav/perception/snapshot_store.py` | 语义快照的写入与读取 |
| `src/robot_nav/adapters/habitat/` | Habitat Adapter |
| `src/robot_nav/adapters/hermes/` | Hermes + D435i Adapter、REST 客户端与外参读取 |
| `src/robot_nav/adapters/realsense/` | RealSense RGB-D 采集、D435i 配置 |
| `src/robot_nav/adapters/openai_compatible.py` | VLM 请求与结构化结果解析 |
| `src/robot_nav/perception/object_localizer.py` | 历史定位的 YOLO/VLM 组合、SAM2 分割与定位回退策略 |
| `src/robot_nav/adapters/object_detection_worker.py` | 在指定环境中常驻运行 YOLO 或 SAM2，记录阶段与调用栈 |
| `src/robot_nav/adapters/sam2_segmenter.py` | SAM2 边界框到像素掩码的分割 |
| `src/robot_nav/adapters/object_model_process.py` | 模型进程的请求、进度、超时与退出 |
| `src/robot_nav/adapters/yolo_world.py` | YOLO-World 模型加载与框检测 |
| `src/robot_nav/visualization/` | Rerun 调试界面 |
| `src/robot_nav/visualization/semantic_world.py` | World 节点的固定 RGB、评分与状态关联 |
| `src/robot_nav/visualization/observation_card.py` | 同次观测的 RGB、评分锚点与状态卡片 |
| `sim/habitat/` | Habitat 环境与启动脚本 |
| `hardware/hermes/`、`hardware/realsense/` | D435i USB 转发、RSUSB 构建与 udev 规则 |

## 当前边界

物体接近的本地模型按需加载后常驻，同帧 SAM2 编码也复用，默认使用已有
`robot-nav` 模型环境。可用
`--object-python` 指定已安装 PyTorch、ultralytics、SAM2 的解释器，
`--object-class` 指定 YOLO 简短类别，`--object-device` 选择设备；场景模式和
随机评分模式不启动本地模型。详细流程与阈值见 [算法说明](docs/algorithm.md)。

这是算法原型，不是功能安全系统。核心输出高层相对位姿，实际路径规划、避障和
执行反馈由 Adapter 后面的仿真器或底盘负责。真机运行必须有人能够立即急停；
动态障碍预测、通用模型重试和连续速度控制尚未实现。
