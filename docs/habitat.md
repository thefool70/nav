# Habitat 仿真

`HabitatChassisAdapter` 把 Habitat-Sim 转换为与真底盘相同的
`ChassisInterface`。导航算法不包含 Habitat 分支；`sim/habitat/` 只保存环境、
渲染包装和简单的 Adapter 验证脚本。
## 安装

Habitat 使用独立的 Python 3.9 环境，避免图形依赖污染核心环境：

```bash
micromamba env create -f sim/habitat/environment.yml
micromamba activate robot-nav-habitat
python -m pip install -e .
python -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes \
  --data-path data/habitat \
  --no-replace
```

环境已存在时可按项目配置同步：

```bash
micromamba env update -n robot-nav-habitat \
  -f sim/habitat/environment.yml \
  --prune
```

当前环境固定 Habitat-Sim 0.3.3、Python 3.9、NumPy 1.26.4、Pillow 10.4.0 和
Rerun 0.22.1。

## 先检查 Adapter

下面的命令读取一帧并打印位姿、地图、RGB 和深度尺寸，不调用 VLM：

```bash
micromamba activate robot-nav-habitat
env HABITAT_RENDERER=gpu sim/habitat/run.sh \
  python sim/habitat/adapter_demo.py \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --seed 1
```

成功时应看到：

```text
Habitat 渲染后端：WSL Mesa D3D12（GPU）。
Renderer: D3D12 (NVIDIA GeForce RTX 3060)
```

`run.sh` 默认使用 `HABITAT_RENDERER=auto`：先验证 WSL D3D12，失败才回退到
conda `llvmpipe`。当前机器建议显式使用 `gpu`，这样 GPU 链路异常会直接报错，
不会静默使用 CPU。GPU 模式统一加载 Arch 的 Mesa、GLVND 与 DRM 库，避免
conda 中同名库覆盖 D3D12 所需版本。

## 运行导航

默认读取项目根目录 `config.json` 中的运行参数，命令行参数优先。场景路径可在
`habitat.scene` 中设置；配置规则见 [统一运行配置](../README.md#统一运行配置)。


没有 VLM 凭据时，先运行随机评分调试模式：

```bash
env HABITAT_RENDERER=gpu sim/habitat/run.sh \
  python -m robot_nav habitat \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --target "门口" \
  --seed 1 \
  --debug-random-score \
  --max-cycles 100
```

该模式不调用模型，只能检查扫描、Frontier、路径规划、运动和重新选点，不能识别
目标。相同场景和 `--seed` 会使用相同随机起点。

正式语义搜索先执行 `opencode auth login`，再去掉
`--debug-random-score`。也可以用 `ROBOT_NAV_VLM_API_KEY` 显式覆盖凭据。默认
模型为 OpenCode Go 的 `qwen3.7-plus`，提示词使用英文并关闭 thinking。
CLI 自动为本次导航设置稳定的 `x-opencode-session` 请求头。若旧进程出现
`HTTP 400 / MissingSessionID`，更新代码并重启导航；请求失败原因会直接打印在终端。

场景搜索额外增加：

```bash
--search-mode scene --target "洗手间"
```

场景模式必须使用 VLM，不能与 `--debug-random-score` 同时使用。

默认启用异步 FIFO 队列：扫描和部分运动画面保存为固定快照，后台在同一次请求
中检查目标并评分 Frontier；没有语义分时按几何分继续探索。随机模式也经过同一
队列，但分析只返回随机分和未发现目标。结果在下一次决策使用，后台检测到目标后
按顺序处理线索。场景模式返回拍摄位姿即完成；物体模式先用历史 RGB-D 的
YOLO/VLM 检测与 SAM2 分割定位；深度定位失败时沿框中心方向或无框时的光轴
查询已公开地图的首个障碍作为位置假设。得到位置后搜索目标周围可达停靠点。历史线索
失败时原地继续处理其他历史画面；全部都无法定位才保底返回并停止，以退出码
`3` 表示未完成搜索。停靠命令执行成功后直接完成搜索，不再进行到达后的检测或测距。具体规则见
[算法说明](algorithm.md#物体接近)。

物体接近默认通过独立 Python 进程复用 `robot-nav` 环境中的模型；
`robot-nav-habitat` 继续负责仿真。可用 `--object-python <PATH>` 指定已有模型
环境的解释器，`--object-class "chair"` 提供简短 YOLO 类别，`--object-device cuda`
选择设备。两个模型分别常驻复用；终端显示当前加载或推理步骤与耗时。
场景和随机评分模式不加载 YOLO/SAM2。

快照与结果保存在 `data/run_logs/semantic-*/job-*`，退出后保留，当前不会自动
恢复上次队列。有效方向耗尽时先停止移动并等待队列；等待不占 `--max-cycles`
额度，达到决策上限仍会退出。

## Rerun

导航启动 Rerun Web 服务但不自动打开浏览器，请手动访问：

[http://127.0.0.1:9090/?url=ws://127.0.0.1:9877](http://127.0.0.1:9090/?url=ws://127.0.0.1:9877)

启动时依次显示服务启动、界面布局发送和 Viewer 地址，之后才初始化 Habitat。
若尚未显示 Viewer 地址就停住，应先检查 Rerun 启动阶段。

主要视图含义：

- RGB、深度和当前占用图。
- 绿色 Frontier、黄色选中点、橙色算法命令。
- 红色 Adapter 实际目标、紫色 navmesh 路径、蓝色机器人轨迹。
- `VLM full` 保留实际输入图、完整提示词、原始输出、HTTP JSON 和解析结果。
- `VLM summary` 显示排队／运行／返回／导航接收状态、有序目标线索、分数和耗时。

提供给 VLM 的 F 编号位于图片下方，通过细引线连接图内的 Frontier 地面锚点。
锚点由拍摄时的位姿、相机标定与对齐深度生成，并随快照保存；简表显示所属 V，
完整卡片记录原始像素坐标。无可靠投影的候选回退到几何分，画面仍参与目标检测。
输入 RGB 仍使用当前 320 像素宽度上限。V 编号使用蓝底白字，页脚按 F 标签
行数收缩；没有 F 时不预留页脚，图间距为 2 像素。

World 占据主要空间，直接在占用图上显示机器人、路径和拍摄任务。同一次扫描只
显示一个 J 点，0.45 m 内的任务合并显示，`+N` 表示另有 N 个邻近任务。点的位置
仍是代表任务的真实拍摄位置。主图保留排队、推理和目标线索，以及最近三个普通
完成任务；完整视角、历史评分点、探索连线和扫描方向保留在 `World history`。
节点颜色表示：橙色排队、紫色推理、蓝色已分析、绿色检出目标、红色失败。

右侧 `Inference RGB + scores` 自动放大当前任务的一张画面：返回前显示首张，
返回后优先首条目标线索，否则显示语义分最高的画面；物体定位时显示正在处理的
具体画面。卡片同时显示 F 锚点、VLM 语义分、拍摄时的综合分与导航接收状态。
`Live camera` 标签页显示机器人当前画面。

`Observations` 保留全部任务：点击 J 在 Selection 面板查看整组 RGB 与评分，
点击 V 查看单图评分卡，raw 查看无标注原图。也可悬浮 World 点查看状态与评分，
点击查看整组卡片；若选中点实例，双击选择整个实体。选择历史记录只改变 Selection
预览，不切换自动跟随推理的面板。没有 Pillow 时卡片退回原图，评分仍保留在节点字段。

下方 `Live` 显示实时位姿与当前阶段，物体推理期间显示模型步骤与等待耗时。
`Motion details` 保留完整运动与分支摘要，`Frontiers` 显示算法候选及排名，
完整决策在 `Status`。

异步调用使用 `J` 表示 FIFO 任务、`R` 表示一次模型请求；图中 `V` 是该请求的
画面编号，`F` 是该请求里的 Frontier 编号。简表显示 F 与拍摄时区域 ID 的映射，
`received C…` 表示导航周期已接收，`ranked C…` 表示分数已交给该周期排序，
不表示一定选中了该候选。目标列表显示为 `V3 → V1`，V 表格的 `Clue order`
列显示线索优先顺序；先处理 V3，无法完成才尝试 V1。场景返回拍摄位姿后完成，
物体优先处理历史定位，停靠成功后完成。`Motion details` 显示物体定位来源、位置和停靠次数。
线索处理期间，简表显示 `Ordinary VLM queue paused; motion prefetch disabled.`。
此时保留待处理任务；已发出的普通请求仍可能返回，但结果先暂存。
现有物体线索用完后，原地恢复已有画面的分析，以处理剩余历史线索。
历史物体定位会产生专用模型请求，停靠后不再追加请求。

`World history` 的 `J/V` 节点对应固定拍摄画面，`J/F` 点对应当时请求评分的 Frontier。
返回结果只更新所属节点，旧节点与原始 RGB 保留；停靠完成后标记对应历史节点为完成。
简表中的 V/F 链接直接选中对应实体。历史评分点不代表当前仍有效的 Frontier；
实时算法候选与目标继续单独显示。

完整窗口显示最近发生的请求或返回事件；简表另保留最近返回结果，因此可能与
完整窗口中正在发送的新请求编号不同。请求和返回按实际记录时刻进入 `frame`
时间轴；在时间面板输入简表的发送／返回帧号即可回看完整会话。R/J 链接选中
`model/vlm/requests` 或 `model/vlm/jobs` 下的归档实体，较早记录也保留在那里。
本次界面变化只对新录制生效。

开启界面时自动从启动开始持续录制到 `data/run_logs/rerun-*.rrd`，终端显示
完整路径。`--rerun-save <PATH>` 可指定新文件，不覆盖已有文件；`--no-rerun`
同时关闭界面和录制。录制包含图像、地图、状态及默认布局，可在 Rerun 0.22.1
中打开回放。

Web Viewer 默认内存上限为 2.5 GB（约 2.33 GiB）；WebSocket 服务端缓存另有
系统总内存 25% 的上限。内存淘汰旧数据不影响独立写入的 RRD，但界面不会自动
从磁盘补回已淘汰帧。正常退出时 SDK 刷新并关闭录制，强制杀进程或断电仍可能
丢失最后尚未写出的数据；磁盘文件会随运行持续增长。

前往 Frontier 时，黄色点与橙色命令指向同一个最终位置，等待本次移动结束后再决策。
返回父节点时，橙色命令指向本次返回节点，`Motion details` 显示节点编号、分支深度与
该节点暂存方向数。
区域用 `region:N` 标识，状态面板显示局部 Frontier 待检查点与复用点数；启动与
后续扫描都只检查局部可见且覆盖不可复用的 Frontier 方向。`scan basis` 显示
观察依据，具体规则见 [算法说明](algorithm.md)。
`frontier choice` 的 `source=new` 表示优先探索新方向，`source=deferred` 表示
新候选耗尽后逐个返回父节点，遇到仍有有效方向的节点再继续探索，对应 `backtrack.return` 与
`backtrack.resume`；`--debug-frontier` 同时显示返回目标、新旧数量和暂存顺序。
`Motion details` 与状态详情还显示待处理视觉工作、失败批次和目标线索。达到
`--max-cycles` 不代表搜索已完成。

## Adapter 行为

- 导航帧显式携带相机离地高度 `sensor_height_m`（默认 1 m），与仿真传感器安装
  位置一致；地面锚点投影使用该外参。相机默认采集分辨率仍为 320×240。
- 内部二维坐标为 `x = Habitat x`、`y = -Habitat z`；top-down map 的行方向在
  Adapter 内完成转换。
- 地图只公开机器人附近和相机视野中具有 navmesh 视线的区域，其余格为未知，
  供 Frontier 算法逐步探索。
- 相对位姿目标先投影到 navmesh，再使用 `GreedyGeodesicFollower` 按 0.25 m
  前进和 10° 转向的离散动作执行。
- 目标无法投影、没有路径或 follower 无法生成动作时，Adapter 报告可恢复运动
  失败；Frontier 移动会淘汰当前目标并继续，返回父节点失败则跳过该返回节点，
  从实际位置重新观察，其有效 Frontier 继续作为候选。
- 返回动作结束后距父节点须不超过 0.25 m；超出容差时以 `motion.backtrack_recovered`
  记录原因并继续搜索，不重复发送同一失败节点的返回命令。
- 动作中间帧送入 Rerun 和独立采样线程，选取新观察位置的画面进入后台队列；
  不额外推进核心状态机，也不因普通评分返回而中断当前动作。

## 常见输出

测试场景可能提示缺少 `.scn` 或 `info_semantic.json`。这表示场景没有 Habitat
语义标注，不影响本项目使用 RGB、深度、navmesh、位姿和占用图。

如果输出停在渲染后端之前，先确认已经激活 `robot-nav-habitat`；如果强制 GPU
时 EGL 自检失败，检查 `/dev/dxg`、WSLg、Arch Mesa 和 Windows NVIDIA 驱动。

## 公共导航与路径约束

Habitat 与 Hermes 共用 `launch.py` 的感知、日志、回调和导航循环装配；
`environment.py` 只负责创建各自 Adapter，真机启动前移不用于仿真。
仿真导航也自动保存 `data/run_logs/habitat-*.jsonl`，可用 `--run-log` 指定路径。

受约束动作在开始执行和每个离散动作前，按选点时的地图检查当前位置到目标的
navmesh 剩余路径。未知长度超过 `navigation.max_unknown_path_m`（默认 1.5 米）
就停止下发动作，并通过与 Hermes 相同的失败结果交给核心恢复；
`--max-unknown-path-m` 可临时覆盖。不使用运动中新扩展的可见地图放宽本次约束。
检查的是 navmesh 规划折线；GreedyGeodesicFollower 的离散轨迹可能存在偏差，
不代表两种环境的实际运动轨迹完全相同。
