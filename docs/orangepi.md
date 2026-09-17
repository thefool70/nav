# 香橙派无线转发：Hermes + D435i

香橙派固定在 Hermes 上，通过网口连接底盘、USB 连接 D435i，无线地址为
`10.113.48.49`，SSH 用户为 `orangepi`。香橙派只采集、对齐并传输 RGB-D 及标定所需 IMU，
通过 SSH 转发底盘 TCP 通信；不运行导航、地图处理、模型推理或动作控制。
Frontier、FOV 地图缓存、完整导航图、路径检查、动作监控及 Rerun 都在开发机。

```text
开发机                                 香橙派                         设备
OrangePiAdapter ── SSH TCP 转发 ───────► SSH ── 网口 ───────────────► Hermes REST
      │                                │
      └─ RemoteD435iCamera ── SSH ─────► RGB-D 服务 ── USB ──────────► D435i
      │
      └─ NavigationFrame → 共用导航 / VLM / YOLO / SAM2 / Rerun
```

`orangepi` 是无线 CLI 入口；本机直连使用 `slamtec-d435i`（旧名 `slamtec-l515` 为别名）。无线入口复用
Hermes 的同步执行与恢复规则，包括正式导航启动前移 1 m、到位稳定后结束 Action、
探索图未知路径检查和物体停靠使用完整导航图，详见 [Hermes 文档](slamtec-l515.md)。

## 当前设备的服务管理

当前香橙派为 Orange Pi Zero 3、Ubuntu 24.04 / ARM64，网口为
`192.168.11.103/24`，Hermes 地址为 `192.168.11.1:1448`。
相机服务使用已有 `/home/orangepi/miniconda3/envs/h2g/bin/python`（Python 3.10、
手工安装的 RealSense SDK 2.56.5）。D435i 当前连接为 USB 2.0，服务配置为
RGB/深度各 640×480、15 FPS；原默认 30 FPS 组合在此连接下启动时解析失败。

两端已启用用户级服务，不必再手动启动同端口的进程。在开发机仓库根目录查看：

```bash
systemctl --user status robot-nav-tunnel.service
ssh -F data/orangepi/ssh_config orangepi-nav \
  'systemctl --user status robot-nav-camera.service'
```

设备端口由 SSH 服务独占。项目 `.vscode/settings.json` 禁止 VS Code 自动转发
`11448`、`18765`，并关闭本项目恢复历史端口转发；Windows 用户设置也忽略这两个
端口。若仍有旧转发，先在 VS Code 的 Ports 面板停止对应项，再重启隧道服务。
不要为这两个端口另外创建 VS Code 转发，否则会与 WSL 中的监听冲突。

服务常驻监听，但启动时不打开相机。首次取帧前自动唤醒 D435i；最后一次取帧或
唤醒请求结束后空闲 15 分钟（900 秒）即关闭 USB 数据流，下次使用自动重开。连续请求期间
保持相机开启，避免每帧重复启动。这里关闭的是 SDK 数据流，不是切断 USB 供电。

重启和查看相机日志：

```bash
systemctl --user restart robot-nav-tunnel.service
ssh -F data/orangepi/ssh_config orangepi-nav \
  'systemctl --user restart robot-nav-camera.service'
ssh -F data/orangepi/ssh_config orangepi-nav \
  'journalctl -b _SYSTEMD_USER_UNIT=robot-nav-camera.service -n 50 --no-pager'
```

相机配置位于香橙派 `~/.config/robot-nav/orangepi-camera.env`，当前设置
`ROBOT_NAV_CAMERA_PYTHON` 和 `ROBOT_NAV_CAMERA_FPS=15`。启动脚本也支持
`ROBOT_NAV_COLOR_WIDTH/HEIGHT`、`ROBOT_NAV_DEPTH_WIDTH/HEIGHT`；修改后重启相机服务。
`ROBOT_NAV_CAMERA_IDLE_S` 可调整空闲关闭时间，默认 900 秒；直接运行服务时对应
`--idle-timeout-s`。服务显示 running 仅说明监听进程在线，不表示相机流正在开启。
停止服务分别使用 `systemctl --user stop robot-nav-tunnel.service` 和香橙派上的
`systemctl --user stop robot-nav-camera.service`；停止数据通道不等于取消底盘动作。

专用 SSH 密钥、已登记主机公钥和连接配置位于开发机 `data/orangepi/`，该目录
不进入 Git。`tunnel.sh` 自动使用这里的密钥；不保存登录密码。SSH 配置默认独立于
系统配置，必要时可用 `ROBOT_NAV_SSH_CONFIG`、`ROBOT_NAV_SSH_IDENTITY`、
`ROBOT_NAV_SSH_KNOWN_HOSTS` 覆盖。

本机 WSL 另外启用了 `robot-nav-orangepi-route.service`，恢复单条主机路由
`10.113.48.49/32 via 219.223.192.1 dev eth0`。它不改变默认路由；若开发机换网，
需核对 `/etc/systemd/system/robot-nav-orangepi-route.service` 中的网关和接口。
服务模板在 `hardware/orangepi/`；以下首次部署步骤适用于其他设备或重新部署。

## 准备香橙派

香橙派需要 Python 3.9 或更新版本、NumPy、能使用 D435i 的 `pyrealsense2`，
以及当前用户的 USB 访问权限。这些是项目已有 RealSense 采集依赖；不需要
PyTorch、CUDA、OpenCV、Rerun 或模型权重。ARM 上的 SDK 安装取决于操作系统与
Python 版本，请按 [RealSense Linux 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)
及 [Python 绑定说明](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md)
准备已有运行环境。仓库 `hardware/realsense/setup_usb_permissions.sh` 可安装 L515
和 D435i 的 USB 权限规则；相机直接转发到 WSL 时，也需要在 WSL 安装这份规则。
Windows 显示 Attached 只说明转发完成，不代表当前 Linux 用户有 USB 读写权限。

在开发机同步代码（不必上传 `data/`、模型或运行日志）：

```bash
ssh orangepi@10.113.48.49 'mkdir -p ~/nav/hardware'
scp -r src orangepi@10.113.48.49:~/nav/
scp -r hardware/orangepi orangepi@10.113.48.49:~/nav/hardware/
```

SSH 按提示输入已有密码，或使用已配置的公钥；脚本不保存密码。随后登录香橙派：

```bash
ssh orangepi@10.113.48.49
cd ~/nav
bash hardware/orangepi/run_camera.sh
```

保持该终端运行。若 SDK 位于特定环境，先激活该环境，或将
`ROBOT_NAV_CAMERA_PYTHON` 设为其 Python 解释器路径。服务默认采集 640×480 的
RGB 与深度、30 FPS，输出深度对齐到彩色图；普通导航不启用 IMU。可以通过
`--camera-serial`、`--color-width/height`、`--depth-width/height` 和 `--fps` 选择设备
实际支持的配置。服务只监听香橙派 `127.0.0.1:8765`，提供 `POST /prepare` 唤醒
和 `GET /frame` 取帧接口，标定时用 `POST /imu` 短时采集原始 IMU，没有底盘指令接口。

## 开发机建立数据通道

另开开发机终端，在仓库根目录运行并保持：

```bash
bash hardware/orangepi/tunnel.sh
```

两个端口都只绑定开发机回环地址：

| 开发机地址 | 转发目标 | 用途 |
| --- | --- | --- |
| `127.0.0.1:11448` | 从香橙派访问 `192.168.11.1:1448` | Hermes 原始 REST 通信 |
| `127.0.0.1:18765` | 香橙派 `127.0.0.1:8765` | D435i 对齐 RGB-D |

`192.168.11.1:1448` 沿用原 Hermes 默认地址，尚需与香橙派网口的实际网络配置
相符；脚本不修改路由或网卡。若底盘地址不同，启动隧道时设置
`ROBOT_NAV_HERMES_HOST`、`ROBOT_NAV_HERMES_PORT`。香橙派 SSH 目标可通过
`ROBOT_NAV_ORANGEPI_HOST` 修改。端口冲突时可设置 `ROBOT_NAV_LOCAL_HERMES_PORT`、
`ROBOT_NAV_LOCAL_CAMERA_PORT`；同时修改导航的 `--base-url`、`--camera-url`。
远端相机端口用 `ROBOT_NAV_REMOTE_CAMERA_PORT` 与服务的 `--port` 配套修改。

## 外参与运行

D435i 的安装外参保存在开发机，默认路径为
`data/orangepi_d435i/extrinsics.json`，不能沿用 L515 的安装值。
优先使用自动标定，不需要先填写外参。在开发机既有 `robot-nav` 环境执行：

```bash
python -m robot_nav calibrate-orangepi --enable-motion
```

运行前停止其他导航进程，保持底盘静止，并让相机下半幅能看到平坦地面。
运动区域留空，但画面中应有约 1–3 m 的静止纹理物体，例如带图案的纸箱、书架
或海报；左右转动时仍需保持足够重叠。仅有开阔地面、远墙或强逆光窗户并不适合
视觉标定。当前运动求解使用 0.25–4 m 的深度，起点须至少有 80 个带可用深度的
RGB 特征点；仅在 RGB 中可见但深度缺失或超范围的特征不计入。
准备独立急停并清空运动范围。程序默认先采集 IMU 和起点 RGB-D，
检查静止状态、地面与纹理；通过后左转 30°、回正、右转 30°、回正，最后前移
0.20 m。这是独立标定流程，不执行普通导航的启动前移 1 m，也不自动退回起点。
可用 `--turn-angle-deg`（10–45°）、`--drive-distance-m`（0.10–0.50 m）调整幅度。

香橙派按设备公布的配置分别选择加速度计和陀螺仪采样率，短时暂停 RGB-D 后
传回原始 IMU、单位和深度到彩色的旋转，采完立即释放 IMU；下一次取帧重新开启
RGB-D。静止判定、重力转换、地面拟合、视觉运动估计和外参求解都在开发机完成。
每张标定图在停车后采集，取帧前后位姿变化超过 1 cm 或 1° 就停止，不用无线
传输时间假装硬件同步。普通导航不持续传输 IMU。

标定成功后自动保存六个外参，后续 `orangepi` 导航会读取该文件。
`--output` 可指定其他路径。静止、地面、视觉匹配或运动残差不满足条件时停止，
保留旧外参；采集数据保存在结果文件旁的 `calibration-*` 目录，包含原始 IMU、
RGB-D、拍摄位姿和可计算的运动对，供定位失败原因。计算复用 L515 的地面拟合
和运动外参求解，不依赖 VLM 或 YOLO/SAM2；开发机需已有 NumPy 和 OpenCV。

如果选择手工填写，可复制模板并把全部 `null` 替换成实测数值：

```bash
mkdir -p data/orangepi_d435i
cp hardware/orangepi/extrinsics.example.json data/orangepi_d435i/extrinsics.json
```

平移以 Hermes 机器人坐标系前、左、上为正，单位米；角度单位度，yaw 向左、
pitch 向下、roll 为图像顺时针。`height_m` 必须大于零。也可用
`--camera-height-m` 等六个参数逐项覆盖。自动标定得到的也是这一坐标约定。

在开发机既有 `robot-nav` 环境中，先只读预检：

```bash
python -m robot_nav orangepi --preflight-only
```

预检读取 Hermes 状态、地图和一帧 RGB-D，不创建运动 Action。正式导航：

```bash
python -m robot_nav orangepi \
  --target "门口" \
  --enable-motion \
  --max-cycles 100
```

如需场景搜索，增加 `--search-mode scene`。模型环境、视觉凭据、快照与 Rerun
均使用开发机已有配置；JSONL 默认保存为 `data/run_logs/orangepi-*.jsonl`。
开发机需要 NumPy，但这个入口不会加载 `pyrealsense2` 或打开本机 USB 相机。
不要使用原 L515 的 WSL USB 转发启动脚本。

## 数据与断连行为

客户端先请求唤醒，再请求一帧数据，不发送历史队列。冷启动独立等待至少 15 秒的
超时额度（若配置的请求超时更长，则沿用更长值），不计入取帧往返时间；帧传输
仍保留原来的超时检查。空闲时关闭相机，图像编码与传输也仅在取帧时进行。
线上的 RGB 为无损 RGB8，深度为
小端 float32 米数，内参来自同一彩色流，整体使用 gzip 压缩；不压缩成 JPEG，
不把无效深度替换成假定距离。请求、帧大小及协议版本均有检查。

香橙派单调时钟仅随包记录；开发机将接收时间用于本地采集对象，导航帧时间继续
使用开发机时钟。RGB-D 同步来自相机 frameset，但相机与 Hermes 位姿不是硬件
同步采集，仍有无线传输与顺序读取误差。`--camera-request-timeout-s` 默认 5 秒，
`--camera-max-roundtrip-s` 默认 3 秒，超过往返上限则拒绝该帧；该门槛不是同步
精度保证。服务端 FPS 只表示相机开启期间的采集配置，实际传回帧率由请求、带宽和处理时间决定。

相机或 Hermes 读取失败会停止导航，并由开发机尝试取消活跃动作。香橙派不运行
失联接管或控制看门狗；如果 SSH／无线链路完全断开，取消请求可能无法送达，
Hermes 已接收的自主动作可能继续执行。软件会报告取消失败，不把它当作已停车。
恢复连接后不会自动恢复上一导航进程。需用底盘独立急停保证失联时能够停车。

排查顺序：确认香橙派相机终端的错误、SSH 隧道是否仍在运行，再对照开发机的
相机读取错误或 Hermes REST 错误。两条通道独立，能读相机不代表底盘地址可达。
