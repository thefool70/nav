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

YOLO-World 持续检测候选，SAM2 用候选框生成掩码。后台 VLM 对选取的固定画面
联合检查目标与评分 Frontier，发现目标后提供拍摄位置线索；机器人接近候选后
仍使用新画面做最终确认。默认 YOLO 与 SAM2
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

场景模式不加载 YOLO-World 或 SAM2。扫描和部分运动画面进入 FIFO 队列，VLM
在同一次请求中检查目的场景与评分 Frontier。缺少分数时按几何分继续探索；
旧图检测到目标场景后返回拍摄位置并对齐朝向，到位即结束。未选方向暂存，新候选耗尽后
逐个返回父节点，遇到有效方向再按原顺序恢复；返回父节点本身不额外扫描。

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
启动前移规划失败或停滞且已确认动作结束时，从实际位置开始搜索；扫描转向的
可恢复失败也从实际朝向重新规划观察。

后续探索一次下发选定的最终 Frontier，等待 Hermes 动作结束后，仅补查局部可见
且尚未检查的 Frontier 方向，再选择下一目标；场景模式仍采集当前画面判断所在
场景。新候选耗尽后，沿当前分支逐个返回；每到一个节点，有有效方向就继续探索，
没有则再退一层。日志分别为 `backtrack.return` 和 `backtrack.resume`，
父节点到达容差为 0.25 m。
核心不插入固定两米停靠。实际路径仍由 Hermes 规划，Frontier 探索与返回父节点
及返回目标线索位置都会检查路径是否经过算法未知区。目标接近沿用原有距离规则，YOLO-World + SAM2 持续检测
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

Frontier、返回父节点与返回目标线索的相对命令使用决策帧位姿还原世界目标，发送前的最新位姿
只用于剩余距离和反馈。日志分别记录 `reference_pose` 与 `start_pose`，避免
两次读取之间朝向变化导致底盘目标偏离算法目标；平移完成日志同时显示 `target_error`。

Frontier 移动使用选点时的有效地图快照检查 Hermes 返回的剩余路径，包括当前
位置到首个路径点，以及各路径点之间的完整线段。落在 `None` 格或地图外部的
实际长度累计超过 1.5 m 时，
先取消并确认 Action 结束，将本次尝试标为 `INVALIDATED`，并屏蔽整个连通
Frontier 区域后选择其他区域。不会沿同一片边界换代表点逐个重试。未知判定使用
L515 筛选后的算法地图，
对应选点时 Rerun 中的灰色区域，不使用 Hermes 原始完整地图替代。

多段未知区长度累加，例如 0.8 m + 0.9 m 会触发取消；累计不超过 1.5 m 时继续。
每次轮询独立检查当前位置和当前剩余路径，重规划后重新测量，不累加不同轮询
的结果或已走过的距离。长度按线段在栅格内的实际部分计算；只擦过格角不计长度，
沿格边行走时任一侧未知就计入一次。可通过 `--max-unknown-path-m 1.5` 调整上限。

区域屏蔽按世界边界保存，换编号、边界一格移动、分裂、合并或短暂退出候选都不会
直接清除记录。本次运行中不自动恢复或重试该区域；被拒绝路径仅保留在取消日志，
后续不复查旧路径来解除屏蔽。普通规划失败、停滞及目标检测中断仍按各自规则恢复。

返回父节点也检查本次决策地图上的实际路径；若未知长度超限、规划失败或停滞，
确认动作结束后以 `motion.backtrack_recovered` 继续，不将尚未尝试的暂存 Frontier
标为不可达。动作结束后仍距父节点超过 0.25 m，也进入相同恢复流程：跳过本次
返回节点，释放该节点的暂存方向，从实际位置重新观察和选点，保留其他节点的回退顺序。
执行失败、停滞和目标检测中断均在取消后确认 Action 终态；确认失败仍停止程序。
目标线索返回失败时尝试下一条，均无法返回才恢复原搜索分支；到位后不再请求新图确认。

路径检查在每次 Action 轮询中执行，重规划后也会复查，不依赖 Rerun 或 2 秒
进度输出间隔。路径尚未发布（空列表）时继续等待；路径读取或取消确认失败则
停止程序。Hermes 当前接口在 Action 创建后才提供路径，因此取消前可能已有
位移。启动前移、标定、扫描转向和目标接近不启用这项 Frontier 路径约束。

- 单个 Action 默认总超时 120 秒。
- `MoveToAction` 连续 8 秒平移不足 2 cm 时结束本次动作。
- `RotateToAction` 以 1° 为有效进展，进入目标朝向 5° 内即可结束。
- Frontier 明确规划失败时淘汰当前方向并继续其他候选。
- Frontier 路径未知长度超过上限时，记录累计长度、上限、首个未知格和被拒绝路径，取消并屏蔽整个区域。
- 探索移动停滞时记录 `STALLED`，检查实际位置并在重新选点时避开本次停滞目标。
- 目标接近失败或停滞时，保留实际位置并重新检测目标。
- 网络、相机、地图或健康状态异常仍会停止程序。

普通 VLM 推理在后台进行，不阻塞 Action 监控；8 秒静止门槛仍只依据底盘动作
的实际进展。Action 创建和后续读取均在异常收尾范围内，Adapter 退出也会取消
尚未结束的动作；取消或终态确认失败时停止程序。

## Rerun 与日志

Rerun 默认开启，使用 `--no-rerun` 关闭。主要图形含义：

- 绿色：Frontier；黄色：本轮选中点；橙色：算法命令。
- 红色：实际提交给 Hermes 的地图目标。
- 紫色：Hermes 返回的剩余规划路径。
- 蓝色：机器人实际轨迹。

World 隐藏自动浮动标签；侧栏 `Live` 显示实时位姿和最近决策，`Frontiers` 表格
显示编号与暂存顺序，编号链接到对应点。右侧其他标签页显示 YOLO、SAM2、VLM
和状态详情。`VLM full` 保留实际输入图、完整提示词、原始输出、HTTP JSON 与
解析结果；新增 `VLM summary` 简要显示 FIFO 任务 J、模型请求 R、目标判断、
评分与导航接收／排序周期。请求和返回各自记录到发生时刻，不将回包写回旧帧。
`Frontiers` 还列出被未知路径屏蔽的区域，状态栏用 `blocked` 显示数量。
`Live` 与状态详情显示待处理视觉工作、失败批次和目标线索。
World 中青色拍摄位姿／朝向和粉色 Frontier 快照对应最近返回的 R/J；青色虚线
仅连接当前位置与参考拍摄点，不是规划路径。V/F 编号及区域映射在简表查看；
发送／返回帧号可用于回放完整会话。图例与交互说明见 [Habitat 的 Rerun 说明](habitat.md#rerun)。

每次导航还会在 `data/run_logs/` 创建 JSONL 日志，记录每周期决策、候选摘要、
世界目标和 Action 位姿反馈。使用 `--run-log <PATH>` 可以指定 JSONL 文件。
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

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--base-url` | Robot Agent 地址，默认 `http://192.168.11.1:1448` |
| `--search-mode` | `object` 或 `scene`，默认 `object` |
| `--action-timeout-s` | 单 Action 总超时，默认 120 秒 |
| `--action-stall-timeout-s` | 连续静止门槛，默认 8 秒 |
| `--max-unknown-path-m` | 当前剩余路径允许的累计未知长度，默认 1.5 m；超过才取消，0 表示不允许正长度未知段 |
| `--min-localization-quality` | 定位模式最低质量，默认 1 |
| `--camera-serial` | 多台 RealSense 时选择 L515 |
| `--debug-frontier` | 打印本轮 Frontier 评分明细 |
| `--rerun-save` | 指定 RRD 录制路径，默认自动创建 |
| `--no-rerun` | 关闭 Rerun 界面及录制 |

完整参数以 `python -m robot_nav slamtec-l515 --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。标定和导航
期间必须有人能够立即急停。
