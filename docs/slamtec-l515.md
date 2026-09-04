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
多个 Frontier 的批量评分，以及机器人接近候选后的最终确认。默认 YOLO 与 SAM2
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
进入目的场景；若没有，再用带 Frontier 编号的拼图统一评分。只有一个 Frontier
时跳过评分，但不会跳过场景判断。

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

随机模式不调用 VLM，也不能识别目标，只用于检查地图、Frontier、Action 和回退。
`--debug-frontier` 会在移动前打印候选坐标、跨度、路径距离和分数组成。

每次正式导航都会先沿当前朝向用 `MoveToAction` 规划前移 1 m，然后才开始首次
8×45° 扫描。预检和标定不执行这个启动动作。

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

- 单个 Action 默认总超时 120 秒。
- `MoveToAction` 连续 15 秒平移不足 2 cm 时结束本次动作。
- `RotateToAction` 以 1° 为有效进展，进入目标朝向 5° 内即可结束。
- Frontier 明确规划失败时淘汰当前方向并继续其他候选。
- 探索移动停滞时，把实际位置视为本次终点并继续扫描。
- 目标接近失败或停滞时，保留实际位置并重新检测目标。
- 网络、相机、地图或健康状态异常仍会停止程序。

VLM 推理发生在 Action 创建之前，因此模型等待时间不计入 15 秒静止门槛。

## Rerun 与日志

Rerun 默认开启，使用 `--no-rerun` 关闭。主要图形含义：

- 绿色：Frontier；黄色：本轮选中点；橙色：算法命令。
- 红色：实际提交给 Hermes 的地图目标。
- 紫色：Hermes 返回的剩余规划路径。
- 蓝色：机器人实际轨迹。

右侧标签页显示 YOLO、SAM2、VLM 和状态详情。VLM 页面保留实际输入图、完整
提示词、原始输出和解析结果。

每次导航还会在 `data/run_logs/` 创建 JSONL 日志，记录每周期决策、候选摘要、
世界目标和 Action 位姿反馈；图像、深度和完整地图仍只放在 Rerun。使用
`--run-log <PATH>` 可以指定日志文件。

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
| `--no-rerun` | 关闭 Rerun |

完整参数以 `python -m robot_nav slamtec-l515 --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。标定和导航
期间必须有人能够立即急停。
