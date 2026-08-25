# Hermes + L515 真机 Adapter

Hermes 提供地图位姿、激光障碍图、自主规划和运动控制；外接 L515 只提供对齐
RGB-D。两者在 `slamtec_l515/adapter.py` 中组合成统一 `NavigationFrame`，算法
核心没有思岚或 RealSense 分支。

## 数据链路

```text
Hermes 位姿 + 激光栅格图 ─────────────┐
L515 RGB-D + 内参 + 安装外参 ──────────┼─► NavigationFrame ─► core
core 相对位姿 ─► 地图系目标点/朝向 ───┴─► Hermes MoveTo/Rotate Action
```

`rest_client.py` 只处理 Robot Agent HTTP 协议；`adapter.py` 负责坐标转换、设备
组合和运动安全检查；L515 采集与外参算法位于 `adapters/realsense/`。标定结果
包含前、左、高度、yaw、向下 pitch 和 roll，目标深度投影会实际使用这六项。

## 环境与 WSL 相机

安装 Python 依赖：

```bash
micromamba activate robot-nav
python -m pip install -e '.[slamtec-l515,visualization]'
```

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

先用随机评分检查状态机和真实运动链路；它不调用大模型，也不能识别语义目标：

```bash
hardware/slamtec_l515/run.sh \
  python -m robot_nav slamtec-l515 \
  --target "门口" \
  --debug-random-score \
  --debug-frontier \
  --enable-motion
```

完整语义搜索需设置 `ROBOT_NAV_VLM_API_KEY`，再移除
`--debug-random-score`。Rerun 默认启用，可用 `--no-rerun` 关闭。

运动命令不会直接发轮速。Adapter 将局部相对平移转换成地图坐标，交给
`MoveToAction` 使用底盘自身规划与避障，再用 `RotateToAction` 达到目标朝向。
每个 Action 都会等待成功、失败或超时；中断和超时时请求终止当前 Action。
运行期间终端每约 2 秒显示 Action ID、状态、已执行时间、连续静止时间、位姿和
底盘返回的阶段。默认连续 30 秒没有超过 2 cm 或 1° 的位姿变化时终止当前
Action。该计时只覆盖已经创建的 Hermes Action；VLM 推理发生在 Action 创建前，
即使耗时很长也不会被判定为底盘停滞。

常用参数：

- `--base-url`：Robot Agent 地址，默认 `http://192.168.11.1:1448`。
- `--action-timeout-s`：单个 Action 超时，默认 120 秒。
- `--action-stall-timeout-s`：活跃 Action 连续静止终止时间，默认 30 秒；不会
  计算模型推理时间。
- `--debug-frontier`：在下发移动命令前打印当前一轮 Frontier 候选及评分组成；
  用于区分本轮候选和 Rerun 中累积显示的历史候选。
- `--min-localization-quality`：仅定位模式使用的最低质量，默认 1。
- `--camera-serial`：连接多台 RealSense 时选择 L515。
- `--camera-calibration`：指定另一份外参 JSON。
- `--camera-height-m`、`--camera-forward-m`、`--camera-left-m`、
  `--camera-yaw-deg`、`--camera-pitch-down-deg`、`--camera-roll-deg`：
  逐项覆盖标定文件，主要用于排错。
- `--base-only`：仅可与 `--preflight-only` 一起使用。

该链路使用 Hermes 自带激光避障与规划，但仍不是独立的功能安全系统。首次标定
和导航必须有人能立即急停。
