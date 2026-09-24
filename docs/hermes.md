# Hermes + D435i 真机 Adapter

这是当前主要真机链路，由原 Hermes + L515 直连版本改为 D435i。Hermes 提供地图位姿、激光栅格图、自主路径规划和运动
控制；外接 D435i 提供对齐 RGB-D。`HermesAdapter` 把两者组合成统一的
`NavigationFrame`，核心算法不包含思岚或 RealSense 分支。

```text
本地 USB：RGB-D + 安装外参 + REST 位姿 ─┐
随车 IPC：RGB-D + 同步底盘位姿 + 固定外参 ┼─► HermesAdapter ─► NavigationFrame ─► core
Hermes 激光地图 ──────────────────────┘

core 相对位姿 ─► 地图系目标 ─► Hermes MoveTo / Rotate Action
```

Adapter、REST 客户端和外参读取都在 `src/robot_nav/adapters/hermes/`，包名与
CLI 子命令均为 `hermes`。默认在本机采集 USB RGB-D；
相机连接随车笔记本时使用文末的远程采集方式。本机直连前关闭占用 D435i 的服务。旧 L515 安装外参不能用于这台相机。

导航参数统一在根目录 `config.json`，命令行可以临时覆盖；规则见
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

## 相机安装外参

项目不再提供自动标定程序。本地 USB 模式读取
`data/hermes_d435i/extrinsics.json`，也可用 `--camera-calibration` 指定已有外参文件。
安装位置改变后，应通过外部测量或标定更新外参，不能沿用旧安装参数。

没有外参文件时，至少用 `--camera-height-m` 提供手测高度；其余字段可用
`--camera-forward-m`、`--camera-left-m`、`--camera-yaw-deg`、
`--camera-pitch-down-deg` 和 `--camera-roll-deg` 提供或覆盖。
已有外参文件与历史标定数据不受程序移除影响。

远程模式使用 `config.json` 的 `camera_height_m`、`camera_forward_m`、`camera_left_m`、
`camera_yaw_deg`、`camera_pitch_down_deg`、`camera_roll_deg` 六项固定外参。
启动随车发布器及 SSH 隧道后，在开发机获取一次：

```bash
python hardware/hermes/fetch_camera_extrinsics.py --config config.json
```

脚本从一包有效数据计算 `inverse(T_map_base) @ T_map_camera`，只更新这六个配置值。
它不重新标定、不控制底盘；以后导航启动时读取配置，运行中不再计算安装外参。
安装位置或发布器标定改变后重新获取。远程模式要求六项齐全；本地 USB 配齐六项时
也直接使用它们，不再读取外参文件。单套配置应对应同一套相机安装。

## 运行物体搜索

先在当前终端加载独立保存的 SiliconFlow 凭据，然后运行：

```bash
source data/credentials/siliconflow.env
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

导航状态机、Frontier、地图处理与快照编码在 CPU 上执行；`--object-device`
只控制本地视觉模型，不改变导航计算的设备。VLM 通过远程接口调用。
`--debug-random-score` 不启动 YOLO/SAM2，不能用该模式评估 GPU 推理性能。

## 运行场景搜索

```bash
hardware/hermes/run.sh \
  python -m robot_nav hermes \
  --search-mode scene \
  --target "洗手间" \
  --enable-motion \
  --max-cycles 100
```

场景模式不加载 YOLO-World 或 SAM2。前沿扫描画面进入 FIFO 队列，VLM
在同一次请求中检查目的场景与评分 Frontier。缺少分数时按几何分继续探索；
旧图检测到目标场景后返回拍摄位置并对齐朝向，到位即结束。未选方向暂存，新候选耗尽后
逐个返回父节点，遇到有效方向再按原顺序恢复；返回父节点本身不额外扫描。

当前 VLM 配置为 SiliconFlow 的 `Qwen/Qwen3.8-27B`，请求地址为
`https://api.siliconflow.cn/v1/chat/completions`，格式为 `chat_completions`。
使用英文提示词，当前模型请求显式设置 `enable_thinking=false`，并通过
`response_format={"type":"json_object"}` 约束 JSON 输出。响应先正常解析 JSON，
语法损坏时使用 `json_repair` 修复，不补造缺失的业务字段。返回后仍校验目标画面编号
和评分字段；格式错误不当作“没有目标”。凭据文件导出 `ROBOT_NAV_VLM_API_KEY`，
需在每个新终端中手动 `source`；该文件不纳入 Git，也不会被程序自动加载。
这是云端服务，无需在随车笔记本启动本地 VLM 进程。

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
`hermes.startup_forward_m` 或 `--startup-forward-m` 调整，0 表示跳过），然后按局部可见且
尚未观察的 Frontier 规划扫描；启动与后续扫描使用同一规则。预检不执行这个启动动作。
启动前移规划失败或停滞且已确认动作结束时，从实际位置开始搜索；扫描转向的
可恢复失败也从实际朝向重新规划观察。

后续探索一次下发选定的最终 Frontier，等待 Hermes 到位后，仅补查局部可见
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
   计算时只搜索更新区域包围框及其外扩膨胀半径内的历史障碍，避免每帧遍历全图；
   视场外的邻近障碍仍参与净空计算。

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
5°（转向），且连续达到 `action_arrival_hold_s`（当前 0.001 秒）的位姿变化小于
2 cm、1°，Adapter 返回到位结果，保留旧任务，由下一 Action 直接替换。
首次进入容差后，按剩余稳定时长尽早复查，不再固定多等一个 0.2 秒轮询间隔；
实际时长包含接口请求，0.001 秒不是实时性保证。平移仍须已经产生有效进展。
日志区分“到位交接”和固件报告的“完成”，并记录新旧 Action 的替换关系。
下一目标仍在下一周期计算，因此这不等于提前规划或连续速度控制。
没有下一动作、进入物体定位、执行异常或退出时，取消残留任务并确认终态。
任务替换语义依据 [SLAMTEC SDK2.0 接口说明](https://wiki.slamtec.com/display/SD/SDK2.0%2BCommon%2BInterface%2BGuide)。

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
启动前移和扫描转向不启用它。

- 单个 Action 默认总超时 120 秒。
- `MoveToAction` 期间连续观察位姿：默认在 0.5 米半径内停留 10 秒后，
  建立横向人工墙并取消动作；离开半径则重新计时。近深度只辅助墙的位置，
  缺少近深度仍会触发，此时墙放在观察起点朝目标方向 0.6 米处。
  人工墙保存在本次运行的观察图中，不写入底盘地图；后续路径穿墙也会取消。
  关闭 Rerun 不关闭这项连续组帧与检测。
- `RotateToAction` 以 1° 为有效进展，进入目标朝向 5° 内并达到上述稳定门槛即可主动收尾。
- Frontier 明确规划失败时淘汰当前方向并继续其他候选。
- Frontier 路径未知长度超过上限时，记录累计长度、上限、首个未知格和被拒绝路径，取消并屏蔽整个区域。
- Frontier 持续受阻超时，取消并确认结束后屏蔽整个区域，本次运行不再尝试；
  返回父节点、目标接近和启动前移走各自的失败恢复，不屏蔽 Frontier。
- 物体模式停靠命令成功后直接完成，不再复检或测距；实际运动失败时每条线索最多尝试三次停靠。
- 网络、相机、地图或健康状态异常仍会停止程序。

普通 VLM 推理在后台进行，不阻塞 Action 监控。平移采用上述位姿停滞门槛，
同时保留整体 Action 超时。Action 创建和后续读取均在异常收尾范围内，Adapter 退出也会取消
尚未结束的动作；取消或终态确认失败时停止程序。

## Rerun 与日志

当前配置默认开启 Rerun。常规导航与延迟对照使用 `--no-rerun`，需要可视化排错时
通过 `--rerun` 按需开启，允许开启时有更高延迟。
`logging.rerun_viewer` 选择 `web` 或 `native`，可用 `--rerun-viewer` 临时覆盖。
随车配置为 `native`：在笔记本桌面终端运行时自动启动 Rerun App，
通过本机 9878 端口传输实时数据，无需浏览器。纯 SSH 终端没有桌面显示环境时，
请改用 `--rerun-viewer web`。两种方式都保存 RRD。关闭 Rerun 会同时关闭界面与 RRD
录制，但保留 JSONL 决策与计时日志；终端 Frontier 调试输出
由 `--debug-frontier` 单独控制。比较延迟时应保持这些开关一致。

远程相机由常驻订阅线程持续接收完整 RGB-D/位姿包，只保留最新包；地图由另一
线程独立请求并转换，每轮完成后等待 `motion_frame_interval_s` 再更新。导航与
可视化读取这些快照，仅观察图更新和组帧串行执行。本地 USB 仍使用 SDK 同步
取帧。模型仅分析前沿扫描选帧；相机持续接收和运动可视化不生成模型任务。
动作返回后，远程组帧要求相机包在本机的接收时刻晚于动作结束，必要时只等待
下一包；此条件不证明源端曝光发生于动作结束后。RGB-D 与位姿始终取自同一个包，
地图独立更新，不保证与图像同时采集。地图更新失败会停止运行；最近一次成功更新
超过 `request_timeout_s + motion_frame_interval_s` 也会拒绝继续使用。

主要图形含义：

- 绿色：Frontier；黄色：本轮选中点；橙色：算法命令。
- 红色：实际提交给 Hermes 的地图目标。
- 紫色：Hermes 返回的剩余规划路径。
- 蓝色：机器人实际轨迹。

`Chassis` 面板记录 Hermes 原始健康、Action 与平台事件反馈，随 RRD 保存和回放。
`health` 包含健康标志、`baseError` 错误列表及固件附加字段；开启 Rerun 时，
现有后台帧采集线程每轮额外读取一次健康信息，周期受 REST、相机采集和
`motion_frame_interval_s` 共同影响。诊断读取失败显示 `read_error`，不沿用旧的正常状态。
运动前的健康检查仍按原规则执行，读取失败或 error/fatal 仍会停止导航。
`action` 复用运动轮询响应，保留任务编号、stage、state.status、result 和 reason，
包括正常结束、失败与取消确认。它表示最后一次任务反馈，不保证此刻仍有活跃任务。
当前版本不再轮询平台事件或底盘时间水位。受阻判断依赖连续帧位姿，
计时、位移、辅助深度及人工墙位置写入 Action 日志，区域屏蔽原因写入周期结果。
健康的后台诊断请求仍仅在开启 Rerun 时执行。

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
  `frame.*` 区分组帧锁等待、相机快照读取、位姿请求、地图快照读取、观察图更新和组帧；
  `frontier.*` 区分栅格准备、可达距离、边界与孔洞过滤、聚类、代表点生成、区域匹配、
  评分排序及选点提交。`semantic.*` 记录主线程感知/评分调用，后台 VLM 耗时仍看队列事件。
- `callback_timing`：分别记录决策日志写入、可视化与终端调试回调的耗时。

`frame.convert_rgb`、`frame.convert_depth` 是 `frame.build` 的子阶段。
Action 创建后输出“下发计时”：`motion.ready_check` 为健康与定位许可检查，
`motion.start_pose` 为发送前位姿，
`motion.create_action` 为创建请求，`motion.monitor_pose` 为监控初始位姿。
这些诊断不参与控制；比较动作衔接时，将“到位交接”或“完成”作为上一任务交回控制的时刻。

`frame.build_lock_wait` 记录等待观察图更新锁的时间；`frame.map_snapshot` 只记录
取得地图快照的时间，附带 `map_age_s`（本机距上次地图更新完成的秒数，不是执行
耗时）。地图请求与转换已移到后台，不再计入主循环。旧的 `frame.schedule_wait`
不再输出。所有本轮取帧阶段均包含在 `cycle.read_frame` 内。
采集锁保护相机读取和历史观察图更新；固定快照后的 RGB-D 转换（`frame.build`）
在锁外进行。前台和后台仍可能在采集、地图更新阶段互相等待。

每个 span 带 `started_monotonic_s`、`ended_monotonic_s`、`duration_s` 和 `completed`。
主线程的 `snapshot.*` 记录前沿预览、观察点、覆盖计算和整轮写盘任务提交；
`snapshot.submit` 不再包含后台文件写入。扫描图像打包、投影与深度压缩移到
串行快照线程，其阶段计时保存在 `semantic_queue` 的 `scan_prepared` 事件中，
通过 `timestamp_s` 关联拍摄帧。该事件仅记录编码完成，实际写盘并发布模型任务
仍以 `queued` 事件为准。`scan_prepared` 仅写 JSONL，不刷新 Rerun；
编码完成和入队都不表示模型已经分析。
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
语义快照是待检测输入：正式扫描快照写入失败仍会停止，不能按普通日志故障处理。

Rerun 开启时，还会从启动开始持续写入 `data/run_logs/rerun-*.rrd`，保存图像、
深度、完整地图、界面状态和默认布局，终端打印完整路径。
`--rerun-save <PATH>` 可指定新文件，不覆盖已有文件；`--no-rerun` 同时关闭
界面和录制。RRD 可用 Rerun 0.22.1 打开回放。

Web Viewer 默认内存上限为 2.5 GB（约 2.33 GiB），WebSocket 服务缓存默认是
系统总内存的 25%。内存淘汰旧帧不影响独立写入的 RRD，界面不会自动从磁盘
补回旧帧。正常退出时 SDK 刷新并关闭录制；强制杀进程或断电可能丢失最后
尚未写出的数据。磁盘文件会随运行持续增长。

## 在随车笔记本运行导航

随车部署目录为 `/home/hri/nav`，使用独立环境 `/home/hri/nav/.venv`（Python 3.11）。
代码与开发机共用同一实现；发布器仍使用原来的环境，不安装到导航环境中。
该 venv 基于随车已有 Conda Python 创建，因此需保留
`/home/hri/miniconda3/envs/robot-nav` 的基础解释器。

仓库用 `config.onboard.json` 保存随车配置快照，运行时通过
`python -m robot_nav --config config.onboard.json hermes ...` 显式选择。
现有随车部署目录仍使用自己的 `config.json`，本次归档不改变部署文件。该配置使用 `camera_source=remote`、
`camera_endpoint=ipc:///tmp/rgbd_pose.ipc` 和 `base_url=http://192.168.11.1:1448`，
默认 `no_rerun=true`。这里 `remote` 表示读取发布器协议，不要求跨机器。
固定安装外参沿用开发机配置，YOLO、SAM2 和 CLIP 权重保存在随车目录。
无需开发机相机或底盘转发；发布器按原方式启动，勿重复启动。

随车笔记本通过 Wi-Fi 访问云端模型，有线连接用于底盘。底盘 DHCP 提供的
`192.168.11.1` 不接受 DNS 查询，因此有线连接 `Wired connection 1` 已设置
`ipv4.ignore-auto-dns=yes` 和 `ipv6.ignore-auto-dns=yes`，域名由 Wi-Fi DNS 解析。
若重新创建有线连接后出现 `Temporary failure in name resolution`，检查
`resolvectl status`，避免再次将底盘地址用作 DNS；无需修改模型或密钥。

登录随车笔记本后：

```bash
cd /home/hri/nav
source .venv/bin/activate
python -m robot_nav hermes --preflight-only
```

确认现场可运动后，复现随机评分导航（会执行配置中的启动前移）：

```bash
python -m robot_nav hermes --target chair --debug-random-score --enable-motion --max-cycles 100
```

正式搜索去掉 `--debug-random-score`，并先执行
`source data/credentials/siliconflow.env`。本次配置的 SiliconFlow 凭据已独立保存，
后续部署代码时不要将凭据纳入代码包。日志保存在随车
`/home/hri/nav/data/run_logs/`。依赖安装与文件迁移不代表模型推理或导航验收通过。

## 随车笔记本转发

随车端完成 D435i 采集、深度对齐、底盘位姿时间插值和相机位姿计算，并将它们
打包发布。开发机运行导航、地图处理、本地模型与动作监控；地图和实时控制
仍通过 SSH 转发后的 REST 访问 Hermes，两条通道共用 `HermesAdapter`。

开发机安装远程相机可选依赖（不需要为远程采集安装 RealSense SDK）：

```bash
python -m pip install -e '.[remote-camera]'
```

随车端使用 `/home/hri/data_publisher/rgbd_pose_publisher.py`，接口文档为同目录下的
`README_rgbd_pose_protocol.md`。本项目不维护发布器副本。随车环境使用
`pyrealsense2`、`numpy`、`pyzmq`、`msgpack`、`scipy`，开发机不需要 RealSense SDK 或 scipy。
在随车笔记本启动发布器：

```bash
cd /home/hri/data_publisher
python rgbd_pose_publisher.py
```

如果随车端有本仓库，也可运行 `bash hardware/hermes/run_camera.sh`，脚本仅调用上述文件。
两种启动方式选其一。新发布器不解析命令行参数，序列号通过其 `CAMERA_SERIAL`
配置；导航要求保持 `ALIGN_DEPTH_TO_COLOR=True`。协议未携带对齐开关，接收端
不能仅凭图像尺寸确认对齐。发布器的 `ROBOT_IP` 应与底盘转发指向同一台 Hermes。

开发机另开终端建立 SSH 转发并保持运行：

```bash
bash hardware/hermes/tunnel.sh
```

默认随车主机 `hri@10.113.45.27`，底盘 `192.168.11.1:1448`。
可用 `ROBOT_NAV_ONBOARD_HOST`、`ROBOT_NAV_HERMES_HOST` 覆盖。
相机使用 SSH Unix 套接字转发：开发机 `/tmp/robot-nav-camera.sock` →
随车端 `/tmp/rgbd_pose.ipc`；底盘仍以 TCP 转发到 `11448`。
可用 `ROBOT_NAV_LOCAL_CAMERA_SOCKET`、`ROBOT_NAV_REMOTE_CAMERA_SOCKET` 改两端路径；
修改本地路径时同步修改导航的 `--camera-endpoint`。SSH 服务端须允许 StreamLocal 转发。
隧道不自动覆盖已有本地 socket；退出后若残留该文件，确认旧隧道已关闭再移除它后重建。
SSH 交互认证或使用现有密钥，仓库不保存密码。
更换发布器后须退出旧隧道并重新建立；自定义 `config.json` 的 `camera.camera_topic`
也需改成 `rgbd.pose`。
首次使用或安装标定变化后，先按“相机安装外参”一节获取并保存固定外参，再运行导航。

```bash
python -m robot_nav hermes --camera-source remote \
  --camera-endpoint ipc:///tmp/robot-nav-camera.sock --base-url http://127.0.0.1:11448 \
  --preflight-only

python -m robot_nav hermes --camera-source remote \
  --camera-endpoint ipc:///tmp/robot-nav-camera.sock --base-url http://127.0.0.1:11448 \
  --target "chair" --enable-motion
```

远程方式无需在开发机运行 USB 转发脚本，也不要求发布 IMU。

接收端只接受 `rgbd_pose` v2：四段依次为 `rgbd.pose`、msgpack 元数据、RGB8、
小端 uint16 深度。深度乘 `depth_scale_m` 转为米；深度已对齐到彩色图，使用
`color_intr`。v2 的 `depth_intr` 描述对齐后的深度图，原始深度内参另存于
`depth_intr_raw`。接收端要求 `depth_aligned_to=color`、`depth_frame=color_optical`
且深度宽高与彩色图一致；一次性外参脚本要求 `T_map_camera_frame=color_optical`。
随车协议 README 若仍写 v1，以当前发布器源码中的 v2 字段为准。

| 发布数据 | 导航使用方式 |
| --- | --- |
| `T_map_base` | 提取 x、y、yaw，作为该帧的底盘 `pose`，不再另查 REST 替换 |
| `T_map_camera` | 仅一次性获取外参脚本使用，导航接收器不读取 |
| `timestamp_ns` | 转为 `NavigationFrame.timestamp_s`，保留 RGB 采集时刻的 Unix 秒数 |
| `frame_id` | 发布帧序号，不作为地图坐标系名称 |

FOV、图像投影和物体定位继续使用“同步底盘二维位姿 + 固定安装外参”，
保持当前底盘近似水平、地面 `z=0` 的算法假设。

接收端校验版本、图像编码、同步有效标记与变换矩阵。无效同步位姿直接报错，
不拼接“旧图像 + 最新 REST 位姿”。地图和动作执行反馈仍单独读取 REST；
地图不在同步包内，也不保证与图像同一采集时刻。

默认 640×480、30 FPS 的 RGB-D 原始载荷约 46 MB/s；常驻订阅会持续使用链路
带宽。接收线程独占 socket，完整消息解码后原子替换最新包，不排队保存历史帧；
ZMQ 接收高水位为 1。断流超时后报错，不无限沿用最后一帧，也不静默重试。

`--camera-timeout-s` 默认 3 秒，用于首次等帧、动作后等新包及接收断流判断。
快照读取同时检查本机接收龄。接收龄和源端图像时间是两回事：网络缓冲、发布器
延迟可能让刚收到的包也已经滞后；本机检查不证明传感器到接收端的绝对帧龄。
源端 Unix 时间不与开发机单调时间相减，两机未同步时也不能直接相减估计网络延迟。

发布器用 REST 请求/响应的墙钟中点估计底盘采样时刻，再对 RGB 时刻插值。
默认要求前后位姿间隔各不超过 100 ms、各自 RTT 不超过 40 ms，等待后侧位姿
最多 120 ms；无法同步时默认丢帧。因此接收超时也可能由底盘位姿请求或同步失败
造成，应查看发布器的 `skip_reasons`。这属于软件时间对齐；深度与 RGB 的时间差
只在 `camera_debug` 中记录，深度空间对齐不表示二者硬件同一时刻曝光。

导航日志与历史快照沿用原有结构，保存算法使用的拍摄位姿、时间和固定外参。
发布器同步诊断在随车端查看。验收时先做只读预检，再检查图像投影、历史定位和运动后的新帧。

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
| `--action-stall-timeout-s` | 转向无进展上限，当前根配置 1 秒 |
| `--blocked-pose-radius-m` / `--blocked-pose-duration-s` | 平移停滞观察半径和时长，默认 0.5 m / 10 s |
| `--front-blockage-distance-m` | 辅助人工墙定位的近深度上限，默认 0.5 m |
| `--max-unknown-path-m` | 当前剩余路径允许的累计未知长度，默认 1.5 m；超过才取消，0 表示不允许正长度未知段 |
| `--min-localization-quality` | 定位模式最低质量，默认 1 |
| `--camera-source` | `local`（默认）或 `remote`，不改变导航实现 |
| `--camera-endpoint` / `--camera-topic` | ZMQ 地址和主题，默认 `ipc:///tmp/robot-nav-camera.sock` / `rgbd.pose` |
| `--camera-timeout-s` | 等待新消息的上限，默认 3 秒 |
| `--camera-serial` | 本地 USB 模式的 D435i 序列号；远程模式在服务端指定 |
| `--camera-calibration` | 本地 USB 的外参文件；六项固定外参配齐时不再读取，远程使用六项配置 |
| `--debug-frontier` | 打印本轮 Frontier 评分明细 |
| `--rerun-save` | 指定 RRD 录制路径，默认自动创建 |
| `--no-rerun` | 关闭 Rerun 界面及录制 |

完整参数以 `python -m robot_nav hermes --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。导航
期间必须有人能够立即急停。
