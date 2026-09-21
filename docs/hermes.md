# Hermes + D435i 真机 Adapter

这是当前主要真机链路，由原 Hermes + L515 直连版本改为 D435i。Hermes 提供地图位姿、激光栅格图、自主路径规划和运动
控制；外接 D435i 提供对齐 RGB-D。`HermesAdapter` 把两者组合成统一的
`NavigationFrame`，核心算法不包含思岚或 RealSense 分支。

```text
Hermes 位姿 + 激光地图 ───────────────┐
D435i RGB-D + 标定参数 ─► 有效地图处理 ─┼─► NavigationFrame ─► core
D435i RGB ─► VLM 视觉队列 ─────────────┘

core 相对位姿 ─► 地图系目标 ─► Hermes MoveTo / Rotate Action
```

Adapter、REST 客户端和标定入口都在 `src/robot_nav/adapters/hermes/`，包名与
CLI 子命令均为 `hermes`。默认在本机采集 USB 图像和 IMU；
相机连接随车笔记本时使用文末的远程采集方式。本机直连前关闭占用 D435i 的服务。旧 L515 安装外参不能用于这台相机。

导航及标定参数统一在根目录 `config.json`，命令行可以临时覆盖；规则见
[统一运行配置](../README.md#统一运行配置)。底盘 GUI 和 SSH 脚本仍使用各自的参数。

## 首次安装

```bash
micromamba activate robot-nav
python -m pip install -e '.[hermes,visualization]'
hardware/realsense/build_rsusb.sh
hardware/realsense/setup_usb_permissions.sh
```

D435i 固件 5.17.0.10 使用配套的 librealsense 2.56.5；本机 RSUSB 构建默认安装该版本。
该后端直接访问 USB，不依赖 WSL 内核的相机 HID/IIO 驱动。构建
脚本需要系统已有 `git`、`cmake` 和 C++ 编译器；权限脚本会请求一次 WSL
`sudo`。

以后通过 `hardware/hermes/run.sh` 启动带相机的命令。脚本会识别 Windows
侧 D435i、通过 `usbipd` 转发到 WSL，并注入 RSUSB 库。首次强制绑定会弹出
Windows UAC；绑定期间 Windows 侧不能同时使用相机。只有已经完成强制绑定并
转发后，才可跳过这一步：

```bash
export ROBOT_NAV_SKIP_USB_PREPARE=1
```

需要恢复 Windows 使用时，在管理员 PowerShell 执行：

```powershell
usbipd unbind --busid <BUSID>
```

升级 SDK 后需退出并重启已有相机进程，旧进程不会自动切换到新库。
若 IMU 在总期限内两路均为 0，先核对当前加载的 SDK 与相机固件，再检查
Windows 的 `usbipd` 绑定方式。普通共享即使显示 Attached，RGB-D 也能使用，
仍可能在相机流关闭、重开后没有 IMU 数据。此时取消
`ROBOT_NAV_SKIP_USB_PREPARE=1`，让启动脚本执行 `bind --force` 并重新转发；
Windows 枚举名称中的 `435i` 与 `D435i` 均受支持。
恢复后应检查连续两轮 RGB-D/IMU 切换，不能只用一次冷启动读取判定恢复。
不要通过延长等待或放宽静止门槛掩盖无数据的问题。官方配套表见
[RealSense 固件发布说明](https://dev.realsenseai.com/docs/firmware-releases-d400/)。

## 预检

只检查 Hermes，不连接 D435i：

```bash
python -m robot_nav hermes \
  --preflight-only \
  --base-only
```

检查 Hermes、D435i、位姿和地图的完整链路：

```bash
hardware/hermes/run.sh \
  python -m robot_nav hermes \
  --preflight-only
```

预检不会创建运动 Action。输出中的 SLAM 状态含义如下：

- `mode=mapping`：底盘正在建图，可以使用地图位姿探索；此时 `quality=0` 不作为
  禁止运动条件。
- `mode=localization`：底盘在已有地图中定位，质量必须达到
  `--min-localization-quality`，默认 1。
- `mode=odometry`、`health=error` 或 `health=fatal`：拒绝运动。

## 标定 D435i

相机固定后标定一次；安装位置改变后必须重做。标定是独立入口
`calibrate-hermes`，程序会左右旋转并向前移动约
0.20 m，执行前清空四周及前方至少 0.5 m，并确保可以立即急停。
画面应同时包含地面和约 1–3 m 的静止纹理物体。程序在运动前检查至少 80 个
RGB 特征点有 0.25–4 m 的可用深度；空旷地面或远景纹理不能替代这一条件。

```bash
hardware/hermes/run.sh \
  python -m robot_nav calibrate-hermes \
  --enable-motion
```

标定使用 Motion Module、深度地面拟合、RGB-D 视觉运动和 Hermes 位姿，估计
相机的前、左、高度、yaw、pitch 和 roll。结果默认保存到
`data/hermes_d435i/extrinsics.json`，导航时自动读取。它是算法开发所需的初值，
不是计量级标定。修改安装位置后重做，或把新外参写到同一路径。

无法自动标定时，至少用 `--camera-height-m` 提供手测高度；其余字段可用
`--camera-forward-m`、`--camera-left-m`、`--camera-yaw-deg`、
`--camera-pitch-down-deg` 和 `--camera-roll-deg` 覆盖。

## 运行物体搜索

先执行 `opencode auth login`，然后运行：

```bash
hardware/hermes/run.sh \
  python -m robot_nav hermes \
  --target "门口" \
  --enable-motion \
  --max-cycles 100
```

物体模式与 Habitat 共用后台 VLM 队列，联合检测目标与评分 Frontier。
收到线索后先用历史 RGB-D 做 YOLO/VLM 检测、SAM2 分割与定位，直接前往目标
附近的可达停靠点，YOLO 或 VLM 任一路检出即可使用。深度定位失败时，有框沿
框中心方向、无框沿拍摄时光轴，在完整导航图中查询首个障碍作为位置假设。
历史线索失败时原地继续
其他画面，全部无法定位才保底返回拍摄位姿并停止。停靠命令执行成功后直接完成；
实际运动失败才有限换点或继续下一条线索，详见 [物体接近](algorithm.md#物体接近)。

本地模型按需启动后常驻，默认复用已有 `robot-nav` 环境。`--object-python` 可指定
模型环境解释器，`--object-class` 可提供简短 YOLO 类别，`--object-device` 默认
为 `cuda`；`--object-yolo-model` 与 `--object-sam-checkpoint` 可指定权重。

## 运行场景搜索

```bash
hardware/hermes/run.sh \
  python -m robot_nav hermes \
  --search-mode scene \
  --target "洗手间" \
  --enable-motion \
  --max-cycles 100
```

场景模式不加载 YOLO-World 或 SAM2。扫描和部分运动画面进入 FIFO 队列，VLM
在同一次请求中检查目的场景与评分 Frontier。缺少分数时按几何分继续探索；
旧图检测到目标场景后返回拍摄位置并对齐朝向，到位即结束。未选方向暂存，新候选耗尽后
逐个返回父节点，遇到有效方向再按原顺序恢复；返回父节点本身不额外扫描。

默认 VLM 是 OpenCode Go 的 `qwen3.7-plus`，使用英文提示词并关闭 thinking。
`ROBOT_NAV_VLM_API_KEY` 可以显式覆盖 OpenCode 本地凭据。

## 调试运动链路

```bash
hardware/hermes/run.sh \
  python -m robot_nav hermes \
  --target "门口" \
  --debug-random-score \
  --debug-frontier \
  --enable-motion
```

随机模式不调用 VLM，也不能识别目标，只用于检查地图、Frontier、Action 和重新选点。
`--debug-frontier` 会在移动前打印候选坐标、跨度、路径距离和分数组成。

默认每次正式导航都会先沿当前朝向用 `MoveToAction` 规划前移 1 m（可用
`hermes.startup_forward_m` 或 `--startup-forward-m` 调整，0 表示跳过），然后才开始首次
8×45° 扫描。预检和标定不执行这个启动动作。
启动前移规划失败或停滞且已确认动作结束时，从实际位置开始搜索；扫描转向的
可恢复失败也从实际朝向重新规划观察。

后续探索一次下发选定的最终 Frontier，等待 Hermes 动作结束后，仅补查局部可见
且尚未检查的 Frontier 方向，再选择下一目标；两种模式没有待查方向时仍采集
当前画面检查目标。新候选耗尽后，沿当前分支逐个返回；每到一个节点，有有效方向就继续探索，
没有则再退一层。日志分别为 `backtrack.return` 和 `backtrack.resume`，
父节点到达容差为 0.25 m。
核心不插入固定两米停靠。实际路径仍由 Hermes 规划，Frontier 探索与返回父节点
及返回目标线索位置都会检查路径是否经过算法未知区。后台结果在下一周期接收，
不因本地检测中断扫描或移动。

## 地图处理

Hermes 原始栅格值在 Adapter 中转换为：

- `0`：未知。
- `1..127`：自由。
- `128..255`：占用。

探索图 `obstacle_map` 独立缓存占用值，不会写回或修改 Hermes 内部地图：

1. 只读取并更新 D435i **当前**理论水平 FOV 内、最远 5 m 的地图格，不按深度或障碍遮挡截断。
2. 启动位置周围 0.50 m 仅初始化一次；之后也只有进入当前 FOV 才更新。
3. 视场外保留上次记录的占用值，不跟随底盘新地图刷新；从未公开的区域保持未知。
4. 障碍仍按 0.36 m 半径膨胀，膨胀结果也只写回本次允许更新的区域。

因此，机器人转开后，即使 Hermes 改变了背后的障碍格，探索图仍保留旧值；
再次转回并覆盖该格时才刷新。地图扩展、原点平移时按世界位置保留缓存；坐标系、
分辨率或地图方向变化时重新初始化。缓存仅存在于当前进程，重启后重新建立。

同一 FOV 缓存还输出未膨胀的 `visibility_map`，用于观测方向筛选和视觉覆盖记录。
视觉射线不使用 0.36 m 净空带判断遮挡；Frontier 与探索移动继续使用膨胀的
`obstacle_map`。两图的未知范围和刷新规则一致，视觉图不会提前读取视场外的新地图。

Frontier 提取还用 `visibility_map` 识别面积不超过 0.05 m² 的八邻接封闭未知孔洞，
不让其边界触发移动或回头补查。连接到大块未知区或地图边缘的方向仍保留。
该过滤不改变三种地图的占用值与避障净空；日志记录忽略的孔洞数和面积。

同一导航帧还保留转换后的完整原始图 `navigation_map`。物体定位失败后的障碍
射线使用这张未膨胀图；停靠选点按 0.36 m 净空计算可达自由区，并搜索目标周围
0.60–2.0 m 的自由格。物体停靠的实际路径检查也使用该完整图，不受探索图 5 m
公开范围限制；其他探索与返回动作仍使用探索图。三张图各司其职，完整图不会
写入 FOV 缓存。历史射线使用拍摄时位姿查询当前地图，两者时间分别写入定位记录。

## Hermes Action 与异常

Adapter 把相对平移转换成地图坐标后交给 `MoveToAction`，再用 `RotateToAction`
达到目标朝向。终端约每 2 秒输出 Action ID、阶段、位姿、运行时间和连续静止
时间；位姿到达与停滞判断在每次 Action 轮询执行，轮询间隔默认 0.2 秒。

若 Action 仍为 `working`，但位置误差已不超过 0.30 m（平移）或角度误差已不超过
5°（转向），且连续 0.001 秒内位姿变化小于 2 cm、1°，Adapter 主动结束该 Action。
平移还必须已经产生有效进展，防止尚未起步就被当作到达。取消后确认 Action
进入终态，并重新检查实际位姿，再开始下一步；终止或确认失败仍停止程序。
日志会显示“按位姿确认到达，已主动结束并确认终态”。固件先报告完成时直接沿用
正常完成流程。0.001 秒为配置的稳定门槛，实际至少等待后续轮询样本；
轮询、网络与取消确认耗时另计。

Frontier、返回父节点与返回目标线索的相对命令使用决策帧位姿还原世界目标，发送前的最新位姿
只用于剩余距离和反馈。日志分别记录 `reference_pose` 与 `start_pose`，避免
两次读取之间朝向变化导致底盘目标偏离算法目标；平移完成日志同时显示 `target_error`。

Frontier 移动使用选点时的有效地图快照检查 Hermes 返回的剩余路径，包括当前
位置到首个路径点，以及各路径点之间的完整线段。落在 `None` 格或地图外部的
实际长度累计超过 1.5 m 时，
先取消并确认 Action 结束，将本次尝试标为 `INVALIDATED`，并屏蔽整个连通
Frontier 区域后选择其他区域。不会沿同一片边界换代表点逐个重试。未知判定使用
D435i 筛选后的算法地图，
对应选点时 Rerun 中的灰色区域，不使用 Hermes 原始完整地图替代。

多段未知区长度累加，例如 0.8 m + 0.9 m 会触发取消；累计不超过 1.5 m 时继续。
每次轮询独立检查当前位置和当前剩余路径，重规划后重新测量，不累加不同轮询
的结果或已走过的距离。长度按线段在栅格内的实际部分计算；只擦过格角不计长度，
沿格边行走时任一侧未知就计入一次。可通过 `--max-unknown-path-m 1.5` 调整上限。

区域屏蔽按世界边界保存，换编号、边界一格移动、分裂、合并或短暂退出候选都不会
直接清除记录。本次运行中不自动恢复或重试该区域；被拒绝路径仅保留在取消日志，
后续不复查旧路径来解除屏蔽。普通规划失败和停滞仍按各自规则恢复；目标检测不抢占正在执行的运动。

返回父节点也检查本次决策地图上的实际路径；若未知长度超限、规划失败或停滞，
确认动作结束后以 `motion.backtrack_recovered` 继续，不将尚未尝试的暂存 Frontier
标为不可达。动作结束后仍距父节点超过 0.25 m，也进入相同恢复流程：跳过本次
返回节点，释放该节点的暂存方向，从实际位置重新观察和选点，保留其他节点的回退顺序。
执行失败、停滞和目标检测中断均在取消后确认 Action 终态；确认失败仍停止程序。
场景线索返回失败时尝试下一条，到位即完成。物体接近失败换停靠点；所有历史
线索都无法定位时才保底返回，返回到位或失败均直接停止，不再重新采集。

路径检查在每次 Action 轮询中执行，重规划后也会复查，不依赖 Rerun 或 2 秒
进度输出间隔。路径尚未发布（空列表）时继续等待；路径读取或取消确认失败则
停止程序。Hermes 当前接口在 Action 创建后才提供路径，因此取消前可能已有
位移。物体保底返回与停靠命令也使用这项路径检查：保底返回用探索图，停靠用
完整导航图；两者失败均不屏蔽 Frontier 区域。
启动前移、标定和扫描转向不启用它。

- 单个 Action 默认总超时 120 秒。
- `MoveToAction` 连续 1 秒平移不足 2 cm 时结束本次动作；原地转向不重置该计时。
- `RotateToAction` 以 1° 为有效进展，进入目标朝向 5° 内并达到上述稳定门槛即可主动收尾。
- Frontier 明确规划失败时淘汰当前方向并继续其他候选。
- Frontier 路径未知长度超过上限时，记录累计长度、上限、首个未知格和被拒绝路径，取消并屏蔽整个区域。
- 探索移动停滞时记录 `STALLED`，检查实际位置并在重新选点时避开本次停滞目标。
- 物体模式停靠命令成功后直接完成，不再复检或测距；实际运动失败时每条线索最多尝试三次停靠。
- 网络、相机、地图或健康状态异常仍会停止程序。

普通 VLM 推理在后台进行，不阻塞 Action 监控；1 秒静止门槛仍只依据底盘动作
的实际进展。Action 创建和后续读取均在异常收尾范围内，Adapter 退出也会取消
尚未结束的动作；取消或终态确认失败时停止程序。

## Rerun 与日志

Rerun 默认开启，使用 `--no-rerun` 关闭。主要图形含义：

- 绿色：Frontier；黄色：本轮选中点；橙色：算法命令。
- 红色：实际提交给 Hermes 的地图目标。
- 紫色：Hermes 返回的剩余规划路径。
- 蓝色：机器人实际轨迹。

World 隐藏候选的浮动标签；下方 `Live` 显示实时位姿和当前阶段，`Frontiers` 表格
显示编号与暂存顺序，编号链接到对应点。VLM 和状态页显示当前分析与导航结果；
物体接近阶段在 `Motion details` 中显示定位来源、目标位置、深度支持点数及停靠次数，
选点时还显示地图来源、净空、候选数和计划目标距离。完整掩码、当次定位地图与
本地推理记录保存在视觉队列目录的 `object-localization/`。
`VLM full` 保留实际输入图、完整提示词、原始输出、HTTP JSON 与
解析结果；新增 `VLM summary` 简要显示 FIFO 任务 J、模型请求 R、目标判断、
评分与导航接收／排序周期。请求和返回各自记录到发生时刻，不将回包写回旧帧。
`Frontiers` 还列出被未知路径屏蔽的区域，状态栏用 `blocked` 显示数量。
`Motion details` 与状态详情显示待处理视觉工作、失败批次和目标线索。
World 在占用图上按任务显示拍摄点，聚合邻近任务，完整视角与评分点在 `World history`。
右侧直接展示当前推理的单图与评分；`Observations` 中点 J 查看整组、V 查看单图
评分卡，raw 查看原图。历史选择的结果显示在 Selection，自动推理面板继续跟随当前任务。
物体推理期间 Live 显示当前步骤和耗时。
图例与交互说明见 [Habitat 的 Rerun 说明](habitat.md#rerun)。

每次导航还会在 `data/run_logs/` 创建 JSONL 日志，记录每周期决策、候选摘要、
世界目标和 Action 位姿反馈。使用 `--run-log <PATH>` 可以指定 JSONL 文件。
排查到点后停留时，按 `cycle` 关联以下事件（耗时单位均为秒，使用单调时钟）：

- `cycle_start`：开始下一轮取帧，可与上一 Action 完成时刻比较循环间隔。
- `cycle_timing`：从本轮入口到执行动作前的总耗时与 `spans`；不含底盘运动。
  `frame.*` 区分取帧锁等待、相机采集、位姿/地图请求、地图转换、观察图更新和组帧；
  `frontier.*` 区分栅格准备、可达距离、边界与孔洞过滤、聚类、代表点生成、区域匹配、
  评分排序及选点提交。`semantic.*` 记录主线程感知/评分调用，后台 VLM 耗时仍看队列事件。
- `callback_timing`：分别记录决策日志写入、可视化与终端调试回调的耗时。

每个 span 带 `started_monotonic_s`、`ended_monotonic_s`、`duration_s` 和 `completed`。
`snapshot.*` 进一步拆分主线程观测中的前沿预览、观察点、图像投影、覆盖计算、深度
编码和整轮提交；提交包含队列锁等待、RGB 压缩、文件写入及队列事件回调。
`frontier.cache_hit` 表示复用了本周期同帧、同排除集的提取结果；前沿计时也包含
快照预览中的调用。命中缓存时不会出现该次提取的准备、BFS、聚类等子阶段。
同名阶段多次调用会逐条保留；`cycle.*` 包含内部的 `frame.*`、`frontier.*` 等子阶段，
`frontier.extract` 也包含提取子阶段，统计时不能把父子耗时相加。
`cycle_timing.ended_monotonic_s` 到首条 Action 创建事件的间隔，包含计时日志写入、
执行前准备和 REST 下发；它不等于单次网络请求耗时。只有走到执行前的周期才输出
完整 `cycle_timing`，取帧或决策异常时结合 `cycle_start` 和错误事件定位。

取消结果的 `rejection_scope=region` 表示整片屏蔽，`rejected_path_world_xy`
保存被拒绝路径；状态中的 `blocked_frontier_regions` 列出屏蔽记录摘要。
Action 进度中的 `unknown_path=当前长度/允许上限` 使用米；取消结果另保存
`unknown_path_length_m`、`unknown_path_limit_m` 和 `checked_path_length_m`，
运行配置记录 `max_unknown_path_m`。

视觉队列另外保存到 `data/run_logs/semantic-*/job-*`：压缩 RGB、拍摄位姿与
候选快照在 `snapshot.json`，分析结果在 `result.json`。图片文件随运行增长；
退出后保留，当前不自动恢复。有效方向耗尽后停止移动并等待任务和线索处理完，
等待不占 `--max-cycles` 额度；中断、决策上限或设备故障仍会结束运行。

运行中的 Rerun 回调或 JSONL 写入失败会停用对应记录功能并提示，不终止导航。
语义快照是待检测输入：正式扫描快照写入失败仍会停止，运动提前采样失败只跳过
该次采样并记录原因；二者不能按普通日志故障处理。

Rerun 开启时，还会从启动开始持续写入 `data/run_logs/rerun-*.rrd`，保存图像、
深度、完整地图、界面状态和默认布局，终端打印完整路径。
`--rerun-save <PATH>` 可指定新文件，不覆盖已有文件；`--no-rerun` 同时关闭
界面和录制。RRD 可用 Rerun 0.22.1 打开回放。

Web Viewer 默认内存上限为 2.5 GB（约 2.33 GiB），WebSocket 服务缓存默认是
系统总内存的 25%。内存淘汰旧帧不影响独立写入的 RRD，界面不会自动从磁盘
补回旧帧。正常退出时 SDK 刷新并关闭录制；强制杀进程或断电可能丢失最后
尚未写出的数据。磁盘文件会随运行持续增长。

## 随车笔记本转发

随车端只做 D435i 采集、深度对齐和 ZMQ 发布，开发机运行导航、地图处理、
本地模型与动作监控。底盘仍经 SSH 转发；两条通道共用 `HermesAdapter`。

开发机安装远程相机可选依赖（不需要为远程采集安装 RealSense SDK）：

```bash
python -m pip install -e '.[remote-camera]'
```

随车端使用原有 `/home/hri/camera/realsense_publisher.py`，本项目不维护或修改
发布器副本。随车环境需要 `pyrealsense2`、`numpy`、`pyzmq`、`msgpack`。
在随车笔记本启动原发布器（默认 IPC 地址）：

```bash
python3 /home/hri/camera/realsense_publisher.py
```

如果随车端有本仓库，也可运行 `bash hardware/hermes/run_camera.sh`，脚本仅调用上述原文件。
两种启动方式选其一；多相机时添加 `--serial <SERIAL>`。导航要求保持默认深度到彩色
对齐，不能使用 `--no-align`。原消息没有对齐状态字段，接收端无法自动判断是否关闭了对齐。

开发机另开终端建立 SSH 转发并保持运行：

```bash
bash hardware/hermes/tunnel.sh
```

默认随车主机 `hri@10.113.45.27`，底盘 `192.168.11.1:1448`。
可用 `ROBOT_NAV_ONBOARD_HOST`、`ROBOT_NAV_HERMES_HOST` 覆盖。
相机使用 SSH Unix 套接字转发：开发机 `/tmp/robot-nav-camera.sock` →
随车端 `/tmp/ngd_frames.ipc`；底盘仍以 TCP 转发到 `11448`。
可用 `ROBOT_NAV_LOCAL_CAMERA_SOCKET`、`ROBOT_NAV_REMOTE_CAMERA_SOCKET` 改两端路径；
修改本地路径时同步修改导航的 `--camera-endpoint`。SSH 服务端须允许 StreamLocal 转发。
隧道不自动覆盖已有本地 socket；退出后若残留该文件，确认旧隧道已关闭再移除它后重建。
SSH 交互认证或使用现有密钥，仓库不保存密码。

```bash
python -m robot_nav hermes --camera-source remote \
  --camera-endpoint ipc:///tmp/robot-nav-camera.sock --base-url http://127.0.0.1:11448 \
  --preflight-only

python -m robot_nav hermes --camera-source remote \
  --camera-endpoint ipc:///tmp/robot-nav-camera.sock --base-url http://127.0.0.1:11448 \
  --target "chair" --enable-motion
```

`--camera-url` 和旧 HTTP 服务已移除。远程方式无需在开发机运行 USB 转发脚本。
发布器不提供 IMU，`calibrate-hermes --camera-source remote` 在连接设备前拒绝执行；
导航使用已有有效安装外参。需要重新自动标定时使用本地相机方式，先释放发布器对相机的占用。

原协议依次为 topic、msgpack 元数据、RGB8、本机字节序 uint16 深度；
随车 x86 主机为小端，接收端按小端读取。深度乘 `depth_scale_m`
变成米，采用彩色内参。占位的 tracking/pose/map 字段不参与导航，位姿与地图只来自 Hermes。
默认 640×480、30 FPS 的 RGB-D 原始载荷约 46 MB/s；PUB 持续采集，但导航按需订阅。
每次读取创建独立订阅，避免导航暂停期间积压消息；断流超时失败，不返回缓存画面。
发布器重启后下一次读取重新建立订阅，正在进行的读取仍可能超时并结束导航。

帧时间戳采用开发机接收时的单调时钟。`--camera-timeout-s` 默认 3 秒，仅限制
连接和等待消息的时间，不证明传感器到接收端的绝对帧龄。Global Time 对齐到随车主机，
不能替代两台主机的时钟同步；未同步时不能用 Unix 时间相减验收网络延迟。
相机和底盘位姿仍顺序读取，不是硬件同步。`receiver.py` 是独立诊断脚本，
随车端继续使用默认 `ipc:///tmp/ngd_frames.ipc`，开发机使用
`--endpoint ipc:///tmp/robot-nav-camera.sock`。其中丢帧和延迟统计不能直接当作
导航验收结论。上述链路仍需实际运行验收。

## 底盘操作面板

开发机运行（保持底盘 SSH 隧道，不需要相机）：

```bash
python -m robot_nav.chassis_gui
```

浏览器打开 `http://127.0.0.1:8088`。直接连接底盘可改为
`python -m robot_nav.chassis_gui --base-url http://192.168.11.1:1448`；
`--port` 改本地页面端口。程序只用 Python 标准库，不需要安装 GUI 或前端依赖。

界面显示电量、充电/对桩、位姿、SLAM、健康与当前任务，约每次查询完成后两秒刷新。
启动仅查询状态，勾选“允许运动”后点击按钮才创建动作。平移单次最多 0.5 m，
转向最多 90°；前后按钮设置机器人前方/后方的地图目标，由底盘自主规划，
不保证直线行驶或倒车，也不保证平移结束保持原朝向。界面不执行导航算法的未知区域路径检查。
发送前检查健康、定位及当前任务；已有未结束任务时拒绝新动作。应停止导航及其他控制程序，
固件的检查与提交不是跨进程互斥锁。

“停止”或 Esc 取消当前任务（包括其他程序创建的任务），不受允许运动复选框限制。
请求已受理不代表停稳；网络失败时使用物理急停。关闭页面或服务不会自动取消任务。
回桩成功以实际对接和 `isCharging` 为准；任务状态 4 表示结束，结果 0 表示成功。
操作请求通信失败时可能已被底盘受理，不自动重试，应先查看实际状态。

GUI 入口与操作编排在 `chassis_gui.py`，页面在 `chassis_gui.html`，协议调用仍在
`adapters/hermes/rest_client.py`。回桩采用
[官方 REST 回桩示例](https://wiki.slamtec.com/pages/viewpage.action?pageId=122618038)，
状态接口参照 [官方 REST 文档](https://docs.slamtec.com/)。本面板尚未进行运行与真机验收。

公共导航装配位于 `launch.py`，环境创建、预检和真机专属启动前移位于
`environment.py`。未知路径上限已移至 `config.json` 的
`navigation.max_unknown_path_m`，与 Habitat 共用；旧自定义配置中的同名字段
需从 `hermes` 组移动到 `navigation` 组，命令行 `--max-unknown-path-m` 不变。

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--base-url` | Robot Agent 地址，默认 `http://192.168.11.1:1448` |
| `--search-mode` | `object` 或 `scene`，默认 `object` |
| `--action-timeout-s` | 单 Action 总超时，默认 120 秒 |
| `--action-stall-timeout-s` | 连续静止门槛，默认 1 秒 |
| `--max-unknown-path-m` | 当前剩余路径允许的累计未知长度，默认 1.5 m；超过才取消，0 表示不允许正长度未知段 |
| `--min-localization-quality` | 定位模式最低质量，默认 1 |
| `--camera-source` | `local`（默认）或 `remote`，不改变导航实现 |
| `--camera-endpoint` / `--camera-topic` | ZMQ 地址和主题，默认 `ipc:///tmp/robot-nav-camera.sock` / `ngd.frame` |
| `--camera-timeout-s` | 等待新消息的上限，默认 3 秒 |
| `--camera-serial` | 本地 USB 模式的 D435i 序列号；远程模式在服务端指定 |
| `--camera-calibration` | 完整相机外参 JSON，默认 `data/hermes_d435i/extrinsics.json` |
| `--debug-frontier` | 打印本轮 Frontier 评分明细 |
| `--rerun-save` | 指定 RRD 录制路径，默认自动创建 |
| `--no-rerun` | 关闭 Rerun 界面及录制 |

完整参数以 `python -m robot_nav hermes --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。标定和导航
期间必须有人能够立即急停。
