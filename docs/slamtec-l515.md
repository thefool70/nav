# Hermes + L515 真机 Adapter

该 Adapter 保持职责分离：SLAMTEC Hermes 提供地图位姿、激光栅格图、自主规划和
运动控制；外接 RealSense L515 只提供对齐后的 RGB 与米制深度。导航算法仍只面向
统一的 `ChassisInterface`，没有思岚或 RealSense 分支。

## 数据链路

```text
Hermes 位姿 + 激光栅格图 ─────────────┐
L515 RGB + 对齐深度 + 内参 ────────────┼─► NavigationFrame ─► core
core 相对位姿 ─► 地图系目标点/目标朝向 ─┴─► Hermes 自主规划 Action
```

`rest_client.py` 只负责 Robot Agent HTTP 协议；`adapter.py` 负责坐标转换和设备
组合；通用 L515 采集放在 `adapters/realsense/`。算法核心没有改动。

当前接入已按 Hermes 固件 6.3.2 的实机返回确认：Robot Agent 从
`http://192.168.11.1:1448` 访问；栅格图按小端协议解析，`0` 转为未知、
`1..127` 转为自由、`128..255` 转为占用。协议给出的地图原点是首格边界，
Adapter 会转换为项目约定的首格中心。

## L515 未连接时预检

先只检查底盘，不会创建任何运动 Action，也不需要视觉模型 Key：

```bash
micromamba activate robot-nav
python -m pip install -e .
python -m robot_nav slamtec-l515 --preflight-only --base-only
```

输出包含型号、固件、位姿、定位质量和地图尺寸。定位质量为 `0` 时仍可完成读取
预检，但 Adapter 会拒绝运动；应先用 RoboStudio 完成建图或重定位。

## 接入 L515

L515 连入 WSL 后安装可选依赖：

```bash
python -m pip install -e '.[slamtec-l515,visualization]'
```

固定相机后填写光心相对底盘中心的二维外参：前方为 `forward` 正方向，左侧为
`left` 正方向，yaw 逆时针为正。先读取一帧检查完整链路：

```bash
python -m robot_nav slamtec-l515 \
  --preflight-only \
  --camera-forward-m 0.0 \
  --camera-left-m 0.0 \
  --camera-yaw-deg 0.0
```

示例中的零值只是尚未测量时的占位，不应作为正式外参。重新安装相机后必须重新
测量或标定。

## 导航

首次运动应清空底盘周围并确保能立即急停。可先用随机评分检查算法和运动链路；
该模式不调用大模型，也不能识别语义目标：

```bash
python -m robot_nav slamtec-l515 \
  --target "门口" \
  --debug-random-score \
  --enable-motion \
  --camera-forward-m 0.0 \
  --camera-left-m 0.0 \
  --camera-yaw-deg 0.0
```

完整语义搜索需设置 `ROBOT_NAV_VLM_API_KEY`，再移除 `--debug-random-score`。
Rerun 默认启用，可用 `--no-rerun` 关闭。

运动命令不会直接发轮速。Adapter 把局部相对平移转换成地图坐标，交给
`MoveToAction` 使用底盘自身规划与避障；随后用 `RotateToAction` 达到目标朝向。
每个 Action 都会轮询到成功、失败或超时，中断和超时时请求终止当前 Action。
固件实际公布的 Action 前缀会在启动时自动发现，不依赖固定命名。

常用参数：

- `--base-url`：Robot Agent 地址，默认 `http://192.168.11.1:1448`。
- `--action-timeout-s`：单个运动 Action 超时，默认 120 秒。
- `--min-localization-quality`：允许运动的最低定位质量，默认 1。
- `--camera-serial`：连接多台 RealSense 时选择 L515。
- `--base-only`：仅可与 `--preflight-only` 一起使用。
