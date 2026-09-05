# Hermes + L515 真机 Adapter

这是当前主要真机链路。Hermes 提供地图位姿、激光栅格图、自主路径规划和运动
控制；外接 L515 提供对齐 RGB-D。`SlamtecL515Adapter` 把两者组合成统一的
`NavigationFrame`，核心算法不包含思岚或 RealSense 分支。

```text
Hermes 位姿 + 激光地图 ───────────────┐
L515 RGB-D + 标定参数 ─► 有效地图处理 ─┼─► NavigationFrame ─► core
                          │             │
                          └─► YOLO + SAM2（物体模式）

core 相对位姿 ─► 地图系目标 ─► Hermes MoveTo / Rotate Action
```

## 首次安装

```bash
micromamba activate robot-nav
python -m pip install -e '.[slamtec-l515,visualization]'
hardware/realsense/build_rsusb.sh
hardware/realsense/setup_usb_permissions.sh
```

RSUSB 用来绕过标准 WSL 内核缺少 L515 Motion Module HID/IIO 枚举的问题。构建
脚本需要系统已有 `git`、`cmake` 和 C++ 编译器；权限脚本会请求一次 WSL
`sudo`。

以后通过 `hardware/slamtec_l515/run.sh` 启动带相机的命令。脚本会识别 Windows
侧 L515、通过 `usbipd` 转发到 WSL，并注入 RSUSB 库。首次强制绑定会弹出
Windows UAC；绑定期间 Windows 侧不能同时使用相机。若已手动转发，可以设置：

```bash
export ROBOT_NAV_SKIP_USB_PREPARE=1
```

需要恢复 Windows 使用时，在管理员 PowerShell 执行：

```powershell
usbipd unbind --busid <BUSID>
```

## 预检

只检查 Hermes，不连接 L515：

```bash
python -m robot_nav slamtec-l515 \
  --preflight-only \
  --base-only
```

检查 Hermes、L515、位姿和地图的完整链路：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --preflight-only
```

预检不会创建运动 Action。输出中的 SLAM 状态含义如下：

- `mode=mapping`：底盘正在建图，可以使用地图位姿探索；此时 `quality=0` 不作为
  禁止运动条件。
- `mode=localization`：底盘在已有地图中定位，质量必须达到
  `--min-localization-quality`，默认 1。
- `mode=odometry`、`health=error` 或 `health=fatal`：拒绝运动。

## 标定 L515

相机固定后标定一次；安装位置改变后必须重做。程序会左右旋转并向前移动约
0.20 m，执行前清空四周及前方至少 0.5 m，并确保可以立即急停。

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav calibrate-slamtec-l515 \
  --enable-motion
```

标定使用 Motion Module、深度地面拟合、RGB-D 视觉运动和 Hermes 位姿，估计
相机的前、左、高度、yaw、pitch 和 roll。结果默认保存到
`data/slamtec_l515/extrinsics.json`，导航时自动读取。它是算法开发所需的初值，
不是计量级标定。

无法自动标定时，至少用 `--camera-height-m` 提供手测高度；其余字段可用
`--camera-forward-m`、`--camera-left-m`、`--camera-yaw-deg`、
`--camera-pitch-down-deg` 和 `--camera-roll-deg` 覆盖。

## 运行物体搜索

先执行 `opencode auth login`，然后运行：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --target "门口" \
  --yolo-class "doorway" \
  --enable-motion \
  --max-cycles 100
```

`--yolo-class` 是可选参数。默认直接使用 `--target`；只有目标是中文长描述或不适合
开放词汇检测时，才需要额外给出简短英文类别。

物体模式会加载：

- `data/models/yolo-world/yolov8s-world.pt`
- `data/models/sam2/sam2.1_hiera_small.pt`

YOLO-World 持续检测候选，SAM2 用候选框生成掩码。VLM 不负责日常框选，只负责
多个新 Frontier 的批量评分，以及机器人接近候选后的最终确认。默认 YOLO 与 SAM2
都使用 CUDA；可分别用 `--yolo-device cpu`、`--sam2-device cpu` 排错。

## 运行场景搜索

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --search-mode scene \
  --target "洗手间" \
  --enable-motion \
  --max-cycles 100
```

场景模式不加载 YOLO-World 或 SAM2。每轮扫描后，VLM 先用无标记拼图判断是否已
进入目的场景；若没有，再用带新 Frontier 编号的拼图评分。未选方向暂存，新候选
耗尽后逐个返回父节点，遇到仍有有效暂存方向的节点再按原顺序恢复。只有一个新 Frontier 或恢复旧方向时
跳过评分；每轮探索扫描仍需场景判断，返回父节点本身不额外扫描。

默认 VLM 是 OpenCode Go 的 `qwen3.7-plus`，使用英文提示词并关闭 thinking。
`ROBOT_NAV_VLM_API_KEY` 可以显式覆盖 OpenCode 本地凭据。

## 调试运动链路

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --target "门口" \
  --debug-random-score \
  --debug-frontier \
  --enable-motion
```

随机模式不调用 VLM，也不能识别目标，只用于检查地图、Frontier、Action 和重新选点。
`--debug-frontier` 会在移动前打印候选坐标、跨度、路径距离和分数组成。

每次正式导航都会先沿当前朝向用 `MoveToAction` 规划前移 1 m，然后才开始首次
8×45° 扫描。预检和标定不执行这个启动动作。

后续探索一次下发选定的最终 Frontier，等待 Hermes 动作结束后，仅补查局部可见
且尚未检查的 Frontier 方向，再选择下一目标；场景模式仍需当前画面确认所在
场景。新候选耗尽后，沿当前分支逐个返回；每到一个节点，有有效方向就继续探索，
没有则再退一层。日志分别为 `backtrack.return` 和 `backtrack.resume`，
父节点到达容差为 0.25 m。
核心不插入固定两米停靠。实际路径仍由 Hermes 规划，Frontier 探索与返回父节点
都会检查路径是否经过算法未知区。目标接近沿用原有距离规则，YOLO-World + SAM2 持续检测
不受扫描方向限制。

## 地图处理

Hermes 原始栅格值在 Adapter 中转换为：

- `0`：未知。
- `1..127`：自由。
- `128..255`：占用。

算法使用的地图是原始地图的临时视图，不会写回或修改 Hermes 内部地图：

1. 累计 L515 理论水平 FOV 内、最远 5 m 的地图格；不按深度或障碍遮挡截断。
2. 始终加入程序启动位置周围 0.50 m 的区域。
3. 未观察区域设为未知。
4. 对已观察障碍膨胀 0.36 m，避免把机器人中心规划到车体无法进入的位置。

该累计只存在于当前进程，重新启动后从头建立。

## Hermes Action 与异常

Adapter 把相对平移转换成地图坐标后交给 `MoveToAction`，再用 `RotateToAction`
达到目标朝向。终端约每 2 秒输出 Action ID、阶段、位姿、运行时间和连续静止
时间。

Frontier 与返回父节点的相对命令使用决策帧位姿还原世界目标，发送前的最新位姿
只用于剩余距离和反馈。日志分别记录 `reference_pose` 与 `start_pose`，避免
两次读取之间朝向变化导致底盘目标偏离算法目标；平移完成日志同时显示 `target_error`。

Frontier 移动使用选点时的有效地图快照检查 Hermes 返回的剩余路径，包括当前
位置到首个路径点，以及各路径点之间的完整线段。经过 `None` 格或地图外部时，
先取消并确认 Action 结束，将本次尝试标为 `INVALIDATED`，并屏蔽整个连通
Frontier 区域后选择其他区域。不会沿同一片边界换代表点逐个重试。未知判定使用
L515 筛选后的算法地图，
对应选点时 Rerun 中的灰色区域，不使用 Hermes 原始完整地图替代。

区域屏蔽按世界边界保存，换编号、边界一格移动、分裂、合并或短暂退出候选都不会
直接清除记录。本次运行中不自动恢复或重试该区域；被拒绝路径仅保留在取消日志，
后续不复查旧路径来解除屏蔽。普通规划失败、停滞及目标检测中断仍按各自规则恢复。

返回父节点也检查本次决策地图上的实际路径；若经过未知区、规划失败或停滞，
确认动作结束后以 `motion.backtrack_recovered` 继续，不将尚未尝试的暂存 Frontier
标为不可达。动作结束后仍距父节点超过 0.25 m，也进入相同恢复流程：跳过本次
返回节点，释放该节点的暂存方向，从实际位置重新观察和选点，保留其他节点的回退顺序。
执行失败、停滞和目标检测中断均在取消后确认 Action 终态；确认失败仍停止程序。

路径检查在每次 Action 轮询中执行，重规划后也会复查，不依赖 Rerun 或 2 秒
进度输出间隔。路径尚未发布（空列表）时继续等待；路径读取或取消确认失败则
停止程序。Hermes 当前接口在 Action 创建后才提供路径，因此取消前可能已有
位移。启动前移、标定、扫描转向和目标接近不启用这项 Frontier 路径约束。

- 单个 Action 默认总超时 120 秒。
- `MoveToAction` 连续 15 秒平移不足 2 cm 时结束本次动作。
- `RotateToAction` 以 1° 为有效进展，进入目标朝向 5° 内即可结束。
- Frontier 明确规划失败时淘汰当前方向并继续其他候选。
- Frontier 路径经过算法未知区时，记录首个未知格坐标和被拒绝路径，取消并屏蔽整个区域。
- 探索移动停滞时记录 `STALLED`，检查实际位置并在重新选点时避开本次停滞目标。
- 目标接近失败或停滞时，保留实际位置并重新检测目标。
- 网络、相机、地图或健康状态异常仍会停止程序。

VLM 推理发生在 Action 创建之前，因此模型等待时间不计入 15 秒静止门槛。

## Rerun 与日志

Rerun 默认开启，使用 `--no-rerun` 关闭。主要图形含义：

- 绿色：Frontier；黄色：本轮选中点；橙色：算法命令。
- 红色：实际提交给 Hermes 的地图目标。
- 紫色：Hermes 返回的剩余规划路径。
- 蓝色：机器人实际轨迹。

World 隐藏自动浮动标签；侧栏 `Live` 显示实时位姿和最近决策，`Frontiers` 表格
显示编号与暂存顺序，编号链接到对应点。右侧其他标签页显示 YOLO、SAM2、VLM
和状态详情。VLM 页面保留实际输入图、完整提示词、原始输出和解析结果。
`Frontiers` 还列出被未知路径屏蔽的区域，状态栏用 `blocked` 显示数量。

每次导航还会在 `data/run_logs/` 创建 JSONL 日志，记录每周期决策、候选摘要、
世界目标和 Action 位姿反馈。使用 `--run-log <PATH>` 可以指定 JSONL 文件。
取消结果的 `rejection_scope=region` 表示整片屏蔽，`rejected_path_world_xy`
保存被拒绝路径；状态中的 `blocked_frontier_regions` 列出屏蔽记录摘要。

Rerun 开启时，还会从启动开始持续写入 `data/run_logs/rerun-*.rrd`，保存图像、
深度、完整地图、界面状态和默认布局，终端打印完整路径。
`--rerun-save <PATH>` 可指定新文件，不覆盖已有文件；`--no-rerun` 同时关闭
界面和录制。RRD 可用 Rerun 0.22.1 打开回放。

Web Viewer 默认内存上限为 2.5 GB（约 2.33 GiB），WebSocket 服务缓存默认是
系统总内存的 25%。内存淘汰旧帧不影响独立写入的 RRD，界面不会自动从磁盘
补回旧帧。正常退出时 SDK 刷新并关闭录制；强制杀进程或断电可能丢失最后
尚未写出的数据。磁盘文件会随运行持续增长。

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--base-url` | Robot Agent 地址，默认 `http://192.168.11.1:1448` |
| `--search-mode` | `object` 或 `scene`，默认 `object` |
| `--action-timeout-s` | 单 Action 总超时，默认 120 秒 |
| `--action-stall-timeout-s` | 连续静止门槛，默认 15 秒 |
| `--min-localization-quality` | 定位模式最低质量，默认 1 |
| `--camera-serial` | 多台 RealSense 时选择 L515 |
| `--debug-frontier` | 打印本轮 Frontier 评分明细 |
| `--rerun-save` | 指定 RRD 录制路径，默认自动创建 |
| `--no-rerun` | 关闭 Rerun 界面及录制 |

完整参数以 `python -m robot_nav slamtec-l515 --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。标定和导航
期间必须有人能够立即急停。
