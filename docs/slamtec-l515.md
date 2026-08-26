# Hermes + L515 真机 Adapter

Hermes 提供地图位姿、激光障碍图、自主规划和运动控制；外接 L515 提供对齐
RGB-D，并限定算法当前真正观察过的地图区域。两者在 `slamtec_l515/adapter.py`
中组合成统一 `NavigationFrame`，算法核心没有思岚或 RealSense 分支。

## 数据链路

```text
Hermes 位姿 + 激光栅格图 ─────────────┐
L515 RGB-D + 内参 + 安装外参 ─► 可见图筛选/障碍膨胀 ─┼─► NavigationFrame ─► core
                         └─► YOLO-World ─► SAM2 掩码 ────────────────┘
core 相对位姿 ─► 地图系目标点/朝向 ───────────────┴─► Hermes MoveTo/Rotate Action
```

`rest_client.py` 只处理 Robot Agent HTTP 协议；`adapter.py` 负责坐标转换、设备
组合和运动安全检查；L515 采集与外参算法位于 `adapters/realsense/`。标定结果
包含前、左、高度、yaw、向下 pitch 和 roll，目标深度投影会实际使用这六项。

Adapter 根据 L515 内参和安装 yaw，先取理论水平 FOV 内最远 5 m 的格，再从当前
深度图计算对应方向邻近 3 列跨全部高度的最远有效深度，只截断确实位于该深度
视界之后的格子，并保留 `0.15 m` 深度余量。低矮或只占局部画面的障碍不会被
当成无限高墙；邻近列完全没有有效深度时退回理论 FOV，不凭空制造遮挡。启动后
首次机器人位姿周围半径 `0.50 m` 的圆形区域仍始终有效。有效障碍按 Hermes
车体 `465 × 545 mm` 的半对角线向上取整为 `0.36 m` 膨胀，使 Frontier 目标不会
落在机器人中心无法进入的区域。Hermes 保存的原始地图不会被修改；程序重启后
重新累计。无 L515 的 `--base-only` 预检仍返回原图。

## 环境与 WSL 相机

安装 Python 依赖：

```bash
micromamba activate robot-nav
python -m pip install -e '.[slamtec-l515,visualization]'
```

该 extra 只管理底盘、相机和可视化依赖，不会重装现有的 PyTorch、Ultralytics
或 SAM2 GPU 环境；语义导航还要求当前环境能够导入 `ultralytics` 和官方
`sam2`，并具备下文所述模型文件。YOLO-World 第一次设置开放词汇类别时还会
加载 CLIP `ViT-B/32`。当前已将经官方 SHA-256 校验的权重缓存到
`weights/clip/ViT-B-32.pt`；硬件启动脚本会固定项目根目录和项目内的
Ultralytics 配置目录，避免因启动位置不同而重复下载。

标准 WSL 内核缺少 L515 Motion Module 所需的 HID Sensor Hub/IIO 枚举。项目用
librealsense 2.54.1 的 RSUSB 用户态后端绕过该限制，首次使用构建一次：

```bash
hardware/realsense/build_rsusb.sh
hardware/realsense/setup_usb_permissions.sh
```

第一条命令需要系统已有 `git`、`cmake` 和 C++ 编译器；第二条会请求一次 WSL
sudo 密码。以后通过 `hardware/slamtec_l515/run.sh` 启动带 L515 的命令。脚本
会自动找到 Windows 侧的 L515、调用 `usbipd` 转发到 WSL，并注入 RSUSB 动态库；
首次共享设备时会弹出 Windows UAC，由用户确认。它也能在 WSLInterop 未注册时
直接通过 `/init` 调用 Windows PowerShell。

若已手动转发设备，可设置 `ROBOT_NAV_SKIP_USB_PREPARE=1`。USB 2 和 USB 3
均可使用，当前 RGB-D 配置为彩色 `640×480@30`、深度 `320×240@30`。

## 预检底盘与相机

只检查 Hermes 时不需要 L515、RSUSB 或视觉模型 Key：

```bash
python -m robot_nav slamtec-l515 --preflight-only --base-only
```

检查完整数据链路时使用启动脚本：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 --preflight-only
```

预检输出包含型号、固件、位姿、地图尺寸、SLAM 模式、定位质量、健康状态和
L515 状态，不会创建运动 Action。

`mode=mapping` 表示底盘正在建图，此时地图位姿可用于探索，`quality=0` 不再被
误判为禁止运动；`mode=localization` 表示在已有地图中定位，此时运动仍受
`--min-localization-quality` 约束。`health=error/fatal` 或建图、定位均未启用
时始终拒绝运动。

## 标定 L515 安装位置

固定或重新移动相机后执行一次标定。相机需看到平整地面和有纹理的静止物体；
清空四周及前方至少 0.5 m，并确保可以立即急停或断电。

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav calibrate-slamtec-l515 \
  --enable-motion
```

程序依次采集静止 IMU、用深度拟合地面、通过 Hermes Action 左转/回正/右转/
回正并前进约 0.20 m，最后用 RGB-D 视觉运动与 Hermes 位姿求解安装外参。结果
默认保存到 `data/slamtec_l515/extrinsics.json`，导航时自动读取；机器人最后
停在起点前方约 0.20 m，不会自动倒回。

这是一组适合算法开发的初始外参，不是计量级标定。结果异常时先检查 JSON 中的
平移残差、地面内点比例和视觉内点数，再改善地面可见范围与环境纹理后重做。

## 导航

每次启动正式 `slamtec-l515` 导航，Adapter 连接和视觉模型加载成功后，Hermes
会先沿当前底盘朝向通过 `MoveToAction` 规划前移 1 m，再进入首次 8×45° 扫描。
预检和标定不会执行这一步；启动前需同时保证前方路线安全并能够立即急停。

先用随机评分检查状态机和真实运动链路；它不调用大模型，也不能识别语义目标：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --target "门口" \
  --debug-random-score \
  --debug-frontier \
  --enable-motion
```

该模式为每轮整批 Frontier 生成随机分数，因此优先级会随随机数变化；只适合
验证数据、状态机和运动反馈，不能用它评价真实目标搜索路径是否合理。

完整语义搜索先用 `opencode auth login` 登录 OpenCode Go，再移除
`--debug-random-score`。入口会自动复用本地凭据；
`ROBOT_NAV_VLM_API_KEY` 仍可用于显式覆盖。Rerun 默认启用，可用
`--no-rerun` 关闭。默认模型是 OpenCode Go 的 `qwen3.7-plus`，使用英文提示词，
并关闭 thinking。

语义模式会自动加载 `data/models/yolo-world/yolov8s-world.pt` 和官方
SAM2.1 Hiera Small `data/models/sam2/sam2.1_hiera_small.pt`，默认都在 CUDA
运行；YOLO 默认置信度为 `0.25`、输入边长为 `640`。YOLO-World 持续检测候选，
SAM2 用候选框生成掩码，目标距离只使用掩码内的对齐深度；候选无法生成非空
掩码时不会接近。接近候选后，Qwen3.7 Plus 对当前完整 RGB 中绿色标出的候选做
最终二元确认。若 `--target` 是中文或较长描述，应另外提供适合开放词汇检测的
简短英文 `--yolo-class`，例如：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --target "门口" \
  --yolo-class "doorway" \
  --enable-motion
```

可用 `--yolo-world-model`、`--sam2-checkpoint` 指定同架构模型，用
`--yolo-device cpu`、`--sam2-device cpu` 临时改为 CPU 推理。

运动命令不会直接发轮速。Adapter 将局部相对平移转换成地图坐标，交给
`MoveToAction` 使用底盘自身规划与避障，再用 `RotateToAction` 达到目标朝向。
每个 Action 都会等待成功、失败或超时；中断和超时时请求终止当前 Action。
运行期间终端每约 2 秒显示 Action ID、状态、已执行时间、连续静止时间、位姿和
底盘返回的阶段。`MoveToAction` 默认连续 15 秒没有超过 2 cm 的平移时终止；
`RotateToAction` 则以 1° 的旋转为有效进展，实际朝向在目标 `5°` 内即满足导航
需要。若实际朝向已经到达但 Action 状态仍停在 `working`，Adapter 会终止该僵住
的 Action 并继续；明显未到达目标的旋转停滞仍停止程序。该计时只覆盖已经创建的
Hermes Action；VLM 推理发生在 Action 创建前，即使耗时很长也不会被判定为底盘
停滞。导航期间每 0.5 秒采集一次最新 L515 帧送入 YOLO-World + SAM2；旧待处理
帧可被覆盖，因此在底盘运动和 VLM 等待期间都不会停止本地检测。扫描或探索中
发现目标会终止当前 Action，并在下一周期从实际位置重新决策。目标接近 Action
会完整执行，避免同一候选反复触发中断，到达后再用新帧检测和最终确认。

Frontier 的 `MoveToAction` 被规划器拒绝、执行失败、总超时，或成功结束但总
平移不足 2 cm 时，会淘汰对应 Frontier，随后从同一观测节点尝试其他候选；
连续静止达到 15 秒门槛则按实际位置结束本次探索移动，下一周期直接扫描。
目标接近的 `MoveToAction` 遇到上述可恢复失败或连续静止时，不淘汰 Frontier、
也不停止导航，而是保留实际位置并在下一周期重新观测目标。网络、相机、地图、
底盘健康状态，以及未达到目标朝向的旋转 Action 异常仍会停止程序。

正式导航每次启动都会在 `data/run_logs/` 自动创建一份 JSONL 日志，并在终端
打印绝对路径。日志逐条保存运行参数、每周期位姿与机器人所在栅格、地图统计、
视觉结果、Frontier 候选摘要、相对命令与世界目标，以及 Hermes Action 的实际
起点、目标、阶段、连续静止时间、位姿、失败原因和最终位移。为控制体积，日志
不复制 RGB、深度、完整占据图和每条 Frontier 的全部格子；这些仍在 Rerun 中
查看。需要固定路径时使用 `--run-log data/run_logs/my-run.jsonl`；若文件已存在，
新记录会追加到文件末尾。

Rerun 的 map 视图用朝向三角形实时表示机器人，并叠加轨迹、当前 Frontier 和
本轮命令；world 视图固定为 Y 轴向上，并显示计划扫描朝向、当前 Frontier、历史
观测节点及其候选方向。空间图形不附着文字，简要状态合并在 world 左上角，完整
信息位于 `navigation/status` 面板。默认布局只保留 RGB、Map、World 三个主视图，
右侧用标签页收纳状态、本地检测、SAM2、VLM 和深度；不会再由 Rerun 自动为每个
实体生成独立面板。后台 YOLO/SAM2 的延迟结果只更新模型标签页，不重复写入
相机主画面、地图、机器人位姿和轨迹，也不单独推进导航时间轴。节点到候选点的
虚线颜色与方向状态一致：
待探索为黄色、执行中为蓝色、已探索为灰色、不可达为红色。
启用 VLM 时，选中对应时间点后，`model/interaction` 使用单张 CJK 卡片显示
完整提示词、实际输入 RGB、模型与推理参数、assistant 文本、完整 HTTP JSON、
解析结果或错误。最终确认时，本地候选框会辅助叠加在卡片 RGB 上；Frontier
评分时，该 RGB 就是实际发送的编号拼图。导航决策帧的 `camera/rgb` 以洋红色
半透明区域显示 SAM2 掩码，并保留绿色 YOLO-World 候选框；状态面板同时显示
掩码尺寸和前景像素数。`model/sam2/latest_success` 单独保留最近一次成功分割，不会被
后续运动帧或无目标帧清除；`model/sam2/status` 显示当前观测的掩码状态。
`model/yolo_world/status` 显示运动帧检测置信度、候选数和推理耗时。

常用参数：

- `--base-url`：Robot Agent 地址，默认 `http://192.168.11.1:1448`。
- `--action-timeout-s`：单个 Action 超时，默认 120 秒。
- `--action-stall-timeout-s`：活跃 Action 连续静止终止时间，默认 15 秒；不会
  计算模型推理时间；探索移动触发后从当时的真实位置继续扫描。
- `--run-log`：指定 JSONL 运行日志路径；默认在 `data/run_logs/` 自动命名。
- `--debug-frontier`：在下发移动命令前打印当前一轮 Frontier 候选及评分组成；
  用于区分本轮候选和 Rerun 中累积显示的历史候选。
- `--min-localization-quality`：仅定位模式使用的最低质量，默认 1。
- `--camera-serial`：连接多台 RealSense 时选择 L515。
- `--yolo-world-model`：YOLOv8s-World 模型文件路径。
- `--yolo-device`：YOLO-World 推理设备，默认 `cuda`。
- `--yolo-class`：可选的开放词汇类别；不指定时复用 `--target`。
- `--yolo-confidence`：YOLO-World 最低置信度，默认 0.25。
- `--yolo-image-size`：YOLO-World 推理边长，默认 640。
- `--yolo-timeout-s`：决策帧等待本地推理的上限，默认 10 秒。
- `--sam2-checkpoint`：SAM2.1 Hiera Small 模型文件路径。
- `--sam2-device`：SAM2 推理设备，默认 `cuda`。
- `--camera-calibration`：指定另一份外参 JSON。
- `--camera-height-m`、`--camera-forward-m`、`--camera-left-m`、
  `--camera-yaw-deg`、`--camera-pitch-down-deg`、`--camera-roll-deg`：
  逐项覆盖标定文件，主要用于排错。
- `--base-only`：仅可与 `--preflight-only` 一起使用。

该链路使用 Hermes 自带激光避障与规划，但仍不是独立的功能安全系统。首次标定
和导航必须有人能立即急停。
