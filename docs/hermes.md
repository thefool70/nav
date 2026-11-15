# Hermes + D435i 使用说明

Hermes 提供建图、定位、规划和运动执行，D435i 提供 RGB-D。
导航与模型可以运行在随车笔记本，也可以在开发机运行并转发设备数据。
两种部署使用同一 Adapter 和搜索算法。

## 选择运行方式

| 方式 | 导航在哪里运行 | 相机配置 | 底盘地址 |
| --- | --- | --- | --- |
| 随车运行 | 随车笔记本 | `remote`，`ipc:///tmp/rgbd_pose.ipc` | `http://192.168.11.1:1448` |
| 开发机运行、转发设备 | 开发机 | `remote`，`ipc:///tmp/robot-nav-camera.sock` | `http://127.0.0.1:11448` |
| 开发机 USB 直连 | 开发机 WSL | `local`，启动脚本准备 USB | 可访问的底盘 REST 地址 |

这里 `remote` 指相机发布协议，同机 IPC 也使用该选项。仓库默认配置为 USB 直连；
随车部署配置与仓库默认值不同，选择对应地址后再预检。所有命令从项目根目录执行。
配置合并规则见 [README](../README.md#统一运行配置)。

正式导航会取消旧动作、清空底盘地图并启用建图，然后按配置启动前移，默认 1 m。
`--startup-forward-m 0` 仅跳过前移，不跳过清图，也不禁止后续搜索运动。
`--preflight-only` 只读检查，不清图、不创建运动动作；正式运动必须显式传 `--enable-motion`。
停止导航使用 Ctrl+C 并等待收尾，关闭显示窗口不能代替停止；现场须能立即急停。

## 准备环境和模型

基础 Python 环境安装见 [README](../README.md#安装入口)。按相机来源安装依赖：

```bash
# 随车 IPC 或转发相机
python -m pip install -e '.[remote-camera,visualization]'
# 本地 USB 相机
python -m pip install -e '.[hermes,visualization]'
```

模型依赖需要在 `--object-python` 指定的解释器环境中准备；未指定时使用当前解释器。
YOLOE 在导航启动时加载并预热，随导航退出释放；默认使用当前解释器环境。`--object-python` 可指定
模型环境解释器，`--object-class` 可提供简短英文 YOLOE 类别，`--object-device` 默认
为 `cuda`；`--object-yolo-model` 指定权重，`--yolo-frequency-hz` 设置视频检测上限。

YOLOE 使用官方 [YOLOE-26s 文本提示模型](https://docs.ultralytics.com/models/yoloe/)，
模型环境要求 `ultralytics>=8.4.0`，以及 CLIP 分词器。权重放在：

```text
data/models/yoloe/yoloe-26s-seg.pt
data/models/yoloe/mobileclip2_b.ts
```

两个文件都来自 `ultralytics/assets` 的 `v8.4.0` 发布；不要使用不支持自定义文本类别的 `-pf.pt`。
模型子进程固定在该目录读取编码器，类别在启动时编码一次，CUDA 使用 FP16 推理。
已收录的中文和英文别名自动映射到固定提示词，例如 `垃圾桶`、`trash can` 和
`waste bin` 均使用 `waste container`；检测结果仍使用原任务类别与 Qwen 融合。
未收录的中文或带限定的长描述，建议用 `--object-class "door"` 等提供简短英文类别；
VLM 仍使用原始目标描述。预设保存在 `src/robot_nav/perception/models/yoloe_targets.json`，
只编码本轮目标，不加载 PF 模型或全词表检测。
实际检测频率和延迟通过运行日志查看。

导航状态机、Frontier、地图处理与快照编码在 CPU 上执行；`--object-device`
只控制本地视觉模型，不改变导航计算的设备。VLM 通过配置的本地或远程接口调用。
`--debug-random-score` 不启动 YOLOE，不能用该模式评估 GPU 推理性能。

### 随车本地 Qwen3.5-4B

随车笔记本为 RTX 4070 Laptop（8 GB 显存）、16 GB 内存。Ollama 安装在
`/home/hri/.local/opt/ollama`，模型保存在 `/home/hri/.local/share/ollama/models`。
`qwen3.5:4b` 使用 Q4_K_M 量化并包含视觉能力，下载大小约 3.4 GB。
用户服务 `nav-vlm.service` 随 hri 登录启动，只监听 `127.0.0.1:11434`，
禁用 Ollama 云端功能；上下文为 4096 token，单模型、单请求，设置
`OLLAMA_KEEP_ALIVE=5m`，连续 5 分钟无请求后自动卸载模型并释放其显存。
服务配置位于 `/home/hri/.config/systemd/user/nav-vlm.service`；服务启动不加载模型。

导航选择本机 `qwen3.5:4b` 的 `/v1/chat/completions` 接口时，`pipeline/launch.py` 在创建
设备和执行启动动作之前调用 `perception/models/ollama_warmup.py`，用 848×480 固定黑图
完成视觉预热；它不读取相机，也不产生导航线索。预热失败会终止本次启动。
随机评分、只读预检和云端模型不触发预热。
后续模型请求续期，导航退出或异常结束后无需额外清理；不发送保活心跳，长时间
没有模型请求时，即使导航进程仍在运行也会卸载，下次请求重新加载。
服务可保持待命；停止服务可立即释放模型资源。其他客户端显式指定的
`keep_alive` 可以覆盖服务默认值，应避免自行设置永久驻留。

登录随车笔记本后管理服务：

```bash
systemctl --user status nav-vlm.service
systemctl --user restart nav-vlm.service
journalctl --user -u nav-vlm.service -n 50 --no-pager
```

随车部署使用本机回环地址访问模型。

若导航运行在开发机，另开一个终端保持模型端口转发：

```bash
ssh -N -L 127.0.0.1:11434:127.0.0.1:11434 hri@10.113.45.27
```

本地与随车配置使用相同的回环接口地址；启动层自动跳过本机 Ollama 的密钥读取，
无需设置占位凭据。该模型请求使用 `reasoning_effort=none`
关闭思考、`temperature=0` 和 JSON Schema 输出约束，输出上限为
`min(vlm_max_output_tokens, 128)`。Schema 限制字段结构，程序仍独立校验分数范围、
框范围与顺序；格式合法不代表目标识别正确。保留原图分辨率及标准 `bbox_2d` 字段。停止服务可用
`systemctl --user stop nav-vlm.service`。

## 相机安装外参

本地 USB 未配齐六项固定外参时读取
`data/hermes_d435i/extrinsics.json`，也可用 `--camera-calibration` 指定已有外参文件。
安装位置改变后，应通过外部测量或标定更新外参，不能沿用旧安装参数。

没有外参文件时，至少用 `--camera-height-m` 提供手测高度；其余字段可用
`--camera-forward-m`、`--camera-left-m`、`--camera-yaw-deg`、
`--camera-pitch-down-deg` 和 `--camera-roll-deg` 提供或覆盖。


远程模式使用 `configs/navigation.json` 的 `camera_height_m`、`camera_forward_m`、`camera_left_m`、
`camera_yaw_deg`、`camera_pitch_down_deg`、`camera_roll_deg` 六项固定外参。
启动随车发布器及 SSH 隧道后，在开发机获取一次：

```bash
python deployment/hermes/fetch_camera_extrinsics.py --config configs/navigation.json
```

脚本从一包有效数据计算 `inverse(T_map_base) @ T_map_camera`，只更新这六个配置值。
它不重新标定、不控制底盘；以后导航启动时读取配置，运行中不再计算安装外参。
安装位置或发布器标定改变后重新获取。远程模式要求六项齐全；本地 USB 配齐六项时
也直接使用它们，不再读取外参文件。单套配置应对应同一套相机安装。

## 在随车笔记本运行导航

部署目录为 `/home/hri/nav`，导航解释器为 `.venv/bin/python`。
该 venv 依赖 `/home/hri/miniconda3/envs/robot-nav` 的基础解释器，不能删掉基础环境。
相机发布器独占 D435i，使用其独立环境；已有发布器运行时不要重复启动，也不要另开 USB 相机。
发布器路径为 `/home/hri/data_publisher/rgbd_pose_publisher.py`，由该目录的配置决定相机序列号和底盘地址。
本仓库的 `bash deployment/hermes/run_camera.sh` 只调用外部发布器，不维护副本。

随车配置使用上表的 IPC 与底盘地址，六项固定外参对应实际安装，模型文件位于随车 `data/models/yoloe/`。
已部署随车配置关闭 Rerun；需要显示时显式加 `--rerun`。

```bash
cd /home/hri/nav
source .venv/bin/activate
python -m robot_nav hermes --preflight-only
```

预检通过并确认现场可运动后，选择一种任务：

```bash
# 物体搜索
python -m robot_nav hermes --target "沙发" --object-class sofa \
  --startup-forward-m 0 --enable-motion --max-cycles 100
# 场景搜索
python -m robot_nav hermes --search-mode scene --target "洗手间" \
  --startup-forward-m 0 --enable-motion --max-cycles 100
```

中文常用目标自动映射为文本提示词；未收录或有额外限定时可用 `--object-class` 给出简短英文类别，
VLM 仍使用原始目标描述。场景预设与成功条件见 [算法说明](algorithm.md#两种搜索模式)。

## 在开发机运行并转发设备

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

如果随车端有本仓库，也可运行 `bash deployment/hermes/run_camera.sh`，脚本仅调用上述文件。
两种启动方式选其一。新发布器不解析命令行参数，序列号通过其 `CAMERA_SERIAL`
配置；导航要求保持 `ALIGN_DEPTH_TO_COLOR=True`，v2 元数据显式声明对齐方式，
接收端同时检查对齐声明与图像尺寸。发布器的 `ROBOT_IP` 应与底盘转发指向同一台 Hermes。

当前彩色和原始深度均为 848×480、30 FPS；同分辨率仍需按两相机内外参将深度
对齐到彩色图，对齐后同样输出 848×480。分辨率由发布器顶部的 `RGB_WIDTH/RGB_HEIGHT` 和
`DEPTH_WIDTH/DEPTH_HEIGHT` 指定，修改后重启发布器生效。本地 USB 采集使用相同
默认配置，见 `platforms/realsense/d435i_camera.py::D435iConfig`。
导航根据每帧内参自动计算视场角，安装外参不因分辨率变化而修改。

开发机另开终端建立 SSH 转发并保持运行：

```bash
bash deployment/hermes/tunnel.sh
```

默认随车主机 `hri@10.113.45.27`，底盘 `192.168.11.1:1448`。
可用 `ROBOT_NAV_ONBOARD_HOST`、`ROBOT_NAV_HERMES_HOST` 覆盖。
相机使用 SSH Unix 套接字转发：开发机 `/tmp/robot-nav-camera.sock` →
随车端 `/tmp/rgbd_pose.ipc`；底盘仍以 TCP 转发到 `11448`。
可用 `ROBOT_NAV_LOCAL_CAMERA_SOCKET`、`ROBOT_NAV_REMOTE_CAMERA_SOCKET` 改两端路径；
修改本地路径时同步修改导航的 `--camera-endpoint`。SSH 服务端须允许 StreamLocal 转发。
隧道不自动覆盖已有本地 socket；退出后若残留该文件，确认旧隧道已关闭再移除它后重建。
SSH 交互认证或使用现有密钥，仓库不保存密码。
相机主题使用 `rgbd.pose`；重启发布器后重新建立隧道。
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

设备隧道不包含模型端口；使用随车 Qwen 时还需按上文单独转发 11434。协议与时序排错见 [Agent 资料](agent/hermes.md#相机协议与时序)。

## 开发机 USB 直连

```bash
micromamba activate robot-nav
python -m pip install -e '.[hermes,visualization]'
deployment/realsense/build_rsusb.sh
deployment/realsense/setup_usb_permissions.sh
```

D435i 固件 5.17.0.10 使用配套的 librealsense 2.56.5；本机 RSUSB 构建默认安装该版本。
该后端直接访问 USB，不依赖 WSL 内核的相机 HID/IIO 驱动。构建
脚本需要系统已有 `git`、`cmake` 和 C++ 编译器；权限脚本会请求一次 WSL
`sudo`。

以后通过 `deployment/hermes/run.sh` 启动带相机的命令。脚本会识别 Windows
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

### 预检

只检查 Hermes，不连接 D435i：

```bash
python -m robot_nav hermes \
  --preflight-only \
  --base-only
```

检查 Hermes、D435i、位姿和地图的完整链路：

```bash
deployment/hermes/run.sh \
  python -m robot_nav hermes \
  --preflight-only
```

预检不会创建运动 Action。输出中的 SLAM 状态含义如下：

- `mode=mapping`：底盘正在建图，可以使用地图位姿探索；此时 `quality=0` 不作为
  禁止运动条件。
- `mode=localization`：底盘在已有地图中定位，质量必须达到
  `--min-localization-quality`，默认 1。
- `mode=odometry`、`health=error` 或 `health=fatal`：拒绝运动。

搜索命令与随车运行相同，在前面加 `deployment/hermes/run.sh`，并使用本地相机配置。例如：

```bash
deployment/hermes/run.sh python -m robot_nav hermes \
  --camera-source local --target chair --startup-forward-m 0 --enable-motion
```

USB/IMU 和 SDK 版本问题见 [设备排错](agent/hermes.md#usb-与环境)。

## Rerun 与日志

仓库默认开启 Rerun，已部署的随车配置默认关闭。常规导航与延迟对照使用 `--no-rerun`，需要可视化排错时
通过 `--rerun` 按需开启，允许开启时有更高延迟。
`logging.rerun_viewer` 选择 `web`、`native` 或 `record`，可用 `--rerun-viewer` 临时覆盖。
随车配置为 `native`：在笔记本桌面终端运行时自动启动 Rerun App，
通过本机 9878 端口传输实时数据，无需浏览器。通过 SSH 启动并在随车屏幕显示时，
使用下节的桌面启动脚本；在开发机查看时使用网页转发或离线回放。
`record` 只写 RRD，不启动界面或实时服务。三种方式都保存 RRD。关闭 Rerun 会同时关闭界面与 RRD
录制，但保留 JSONL 决策与计时日志；终端 Frontier 调试输出
由 `--debug-frontier` 单独控制。比较延迟时应保持这些开关一致。

### SSH 启动，在随车屏幕查看

随车笔记本先登录 `hri` 的桌面；SSH 也使用同一用户。登录 SSH 后执行：

```bash
cd /home/hri/nav
bash deployment/hermes/run_desktop.sh \
  --target "椅子" --object-class chair \
  --startup-forward-m 0 --enable-motion --max-cycles 100
```

脚本从当前用户的 systemd 环境读取桌面的显示地址和认证路径，自动开启
`--rerun --rerun-viewer native`；无需手填重登后会变化的 `XAUTHORITY`。
同时将项目 `.venv/bin` 加入本次进程的 `PATH`，供 Rerun SDK 查找已安装的查看器。
这只为本次进程设置环境，不修改桌面权限或全局配置。Rerun 窗口显示在随车屏幕，
终端输出保留在 SSH 中，RRD 和 JSONL 仍保存在随车 `data/run_logs/`。
保持 SSH 终端开启；停止导航用 Ctrl+C，关闭 Rerun 窗口不代表停止导航。
未登录桌面时脚本报错，不启动导航。该入口固定开启桌面显示，不能用于关闭可视化。

也可以在开发机项目目录直接执行，无需先手动登录 SSH：

```bash
bash deployment/hermes/tunnel.sh --desktop \
  --target "椅子" --object-class chair \
  --startup-forward-m 0 --enable-motion --max-cycles 100
```

此方式只建立命令终端，不转发或下载可视化数据，也不需要 `ssh -X`。
如果直接坐在随车笔记本前，在桌面终端执行
同一条 `run_desktop.sh` 命令即可。若直接使用 Python 命令，先运行
`source .venv/bin/activate`，再执行
`python -m robot_nav hermes --target chair --object-class chair --startup-forward-m 0 --enable-motion --rerun --rerun-viewer native`。
相机发布器仍按原方式启动，勿重复启动。

### 从开发机启动并自动转发

在开发机项目目录执行下面一条命令，SSH 会同时启动随车导航、转发 Rerun 网页
（9090）与实时数据（9877），不需要另开转发终端：

```bash
bash deployment/hermes/tunnel.sh --run \
  --target "沙发" --object-class sofa \
  --startup-forward-m 0 --enable-motion --max-cycles 100
```

服务启动后，在开发机浏览器打开
<http://127.0.0.1:9090/?url=ws://127.0.0.1:9877>。导航、模型与相机均在随车端运行，
日志也保存在随车端。脚本使用随车 `.venv/bin/python`，默认开启 Rerun 并强制网页模式，
不改写随车配置。`--enable-motion` 仍须显式提供；开始正式运行前可用
`bash deployment/hermes/tunnel.sh --run --preflight-only` 预检。

保持该终端开启；Ctrl+C 交给导航按原有流程停止，SSH 会话结束后关闭转发。
如果本机 9090 或 9877 已被占用，SSH 会报错退出，远端导航不会启动。
`--no-rerun` 可关闭界面与录制，SSH 会话中的两个转发端口仍保留到退出。
主机与项目目录可分别用 `ROBOT_NAV_ONBOARD_HOST`、`ROBOT_NAV_ONBOARD_REPO` 覆盖，
默认是 `hri@10.113.45.27` 和 `/home/hri/nav`。

此自动转发入口在开发机执行；直接在随车笔记本终端运行 Python 不会建立到开发机的转发。
无参数的 `tunnel.sh` 仍用于将相机和底盘转给开发机运行导航，见
[开发机转发设备](#在开发机运行并转发设备)。

### 随车录制，结束后在本机回放

在开发机项目目录运行：

```bash
bash deployment/hermes/tunnel.sh --record \
  --target "椅子" --object-class chair \
  --startup-forward-m 0 --yolo-frequency-hz 10 \
  --enable-motion --max-cycles 100
```

导航和模型在随车端运行，Rerun 只写 RRD，不启动网页、桌面 App 或实时转发。
正常结束或 Ctrl+C 完成导航收尾后，脚本下载本次确定的 RRD 与 JSONL 到
`data/run_logs/onboard/run-*/`，再用本机 Rerun App 打开。关闭回放窗口后脚本返回
本次导航退出码；Ctrl+C 中断通常返回非零。录制仍有编码和写盘开销，不等于 `--no-rerun`。

脚本复用已有 SSH 认证；本机 Rerun 从 PATH、项目 `.venv` 或
`~/micromamba/envs/robot-nav/bin/rerun` 查找，也可用 `ROBOT_NAV_RERUN_BIN` 指定可执行文件。
本机须有可用的桌面显示环境，当前 WSL 使用 WSLg。没有安装 Rerun 时先停止，不启动导航。
下载与回放不删除随车原文件；后续可直接执行 `rerun <本机路径.rrd>` 再次打开。
SSH 断线时无法确认远端已退出，因此不自动下载，需确认导航结束后手动取回；下载失败
保留 `.part`，不把不完整文件作为成功录制打开。

`--record` 自动指定录制和日志路径，不能与 `--no-rerun`、`--preflight-only`、
`--rerun-save` 或 `--run-log` 混用。直接在随车端启动时使用
`--rerun --rerun-viewer record` 可以只录制，但不会自动下载。

### 查看结果

JSONL 保存决策、动作反馈和计时；启用 Rerun 时另存 RRD，默认在 `data/run_logs/`。
`--no-rerun` 同时关闭界面和 RRD，保留 JSONL。`--rerun-save` 可指定新的录制路径。
Rerun 回放使用 0.22.1；World 看地图和路径，Live 看阶段和位姿，
Inference/Observations 看识别画面，VLM full 看模型实际输入输出，Motion details 看动作反馈。
场景确认应查看环视合图，不能只看普通扫描请求。模型子进程日志在 `semantic-*/models/`。
退出码 0 表示成功，3 表示保底返回后停止、未完成搜索，1 表示失败或周期上限。

## 地图与动作行为

每次正式 `hermes` 导航启动时，先确认实时激光接口可用，取消并确认底盘当前动作结束，
再通过 `DELETE /api/navigation/slam/v1/maps` 清空底盘全部地图并启用建图。项目地图、视野缓存、
人工墙和搜索状态从空状态重新建立；不删除日志、模型或磁盘中的地图备份。
只读 `--preflight-only` 不清图、不改变建图开关。清图或启用建图失败时停止启动。

项目占用值来自 `GET /api/navigation/system/v1/laserscan`（最低固件 4.2.2）；底盘概率图只提供格网几何。使用扫描随附的机器人位姿，把有效点的米制距离和相对前方角度投到地图系。
射线穿过的格标为自由 `0.0`，末端命中格立即标为占用 `1.0`，同帧命中优先于自由证据。
未扫描格为未知 `None`；无效点不清除障碍。后续射线可清除历史激光障碍，未重扫区域
保留历史值，不靠时间自动清空。格网分辨率与坐标来自底盘地图接口，但不读取其占用值；
数组随激光历史扩展，不等待概率地图公开障碍。底盘自身规划仍由厂商实现负责。

探索图 `obstacle_map` 独立缓存占用值，不会写回或修改 Hermes 内部地图：

1. 只读取并更新 D435i **当前**理论水平 FOV 内、最远 5 m 的实时激光历史格，不按相机深度额外截断。激光射线本身止于命中障碍，不公开障碍后的未知格。
2. 启动位置周围 0.50 m 仅初始化一次；之后也只有进入当前 FOV 才更新。
3. 视场外保留上次记录的占用值，不跟随全方向激光历史图刷新；从未公开的区域保持未知。
4. 障碍按 0.36 m 半径膨胀，膨胀结果也只写回本次允许更新的区域。
   计算时只搜索更新区域包围框及其外扩膨胀半径内的历史障碍，避免每帧遍历全图；
   视场外的邻近障碍仍参与净空计算。

因此，机器人转开后，即使激光改变了背后的障碍格，探索图仍保留旧值；
再次转回并覆盖该格时才刷新。地图扩展、原点平移时按世界位置保留缓存；坐标系、
分辨率或地图方向变化时重新初始化。缓存仅存在于当前进程，重启后重新建立。

同一 FOV 缓存还输出未膨胀的 `visibility_map`，用于观测方向筛选和视觉覆盖记录。
视觉射线不使用 0.36 m 净空带判断遮挡；Frontier 与探索移动继续使用膨胀的
`obstacle_map`。两图的未知范围和刷新规则一致，视觉图不会提前读取视场外的新地图。

目标定位与停靠使用全方向激光历史的未膨胀 `navigation_map`，并按 0.36 m 净空选点。
它不会反向公开探索图的视场外区域。所有地图仅属于本轮进程。

平移由 `MoveToAction` 执行，最终朝向由 `RotateToAction` 完成。
到达容差并稳定后可交接下一动作，日志区分到位交接与固件终态。
无下一动作、定位等待、异常和退出时取消残留动作并确认终态。

平移动作从首帧累计 3 秒窗口，净位移不足 0.25 m 视为停滞，原地转向不算脱困。
首次停滞取消确认后重发原目标；重试仍无有效平移满 3 秒才取消并建人工墙。
未知路径超限则取消确认后直接建墙，不等待停滞复查；扫描转向不使用建墙检测。
取消确认失败时停止，不重发、不建墙。

近深度用于细化墙位置，无可靠近深度时放在目标路线前方 0.60 m。
墙保留停止位置的车体净空，原始墙写入完整图，膨胀墙写入探索图，激光自由证据不能清除它。
后续平移仍检查实际规划路径是否穿墙。未知路径检查发生在 Action 创建之后，取消前可能已有位移。
详细路径与异常契约见 [底盘接口](chassis-interface.md#执行契约)，日志定位见 [Agent 资料](agent/hermes.md)。

## 底盘操作面板

开发机运行（保持底盘 SSH 隧道，不需要相机）：

```bash
python -m robot_nav.platforms.hermes.chassis_gui
```

浏览器打开 `http://127.0.0.1:8088`。直接连接底盘可改为
`python -m robot_nav.platforms.hermes.chassis_gui --base-url http://192.168.11.1:1448`；
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

GUI 入口与操作编排在 `platforms/hermes/chassis_gui.py`，页面在 `chassis_gui.html`，协议调用仍在
`platforms/hermes/rest_client.py`。回桩采用
[官方 REST 回桩示例](https://wiki.slamtec.com/pages/viewpage.action?pageId=122618038)，
状态接口参照 [官方 REST 文档](https://docs.slamtec.com/)。本面板尚未进行运行与真机验收。

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--base-url` | Robot Agent 地址，默认 `http://192.168.11.1:1448` |
| `--search-mode` | `object` 或 `scene`，默认 `object` |
| `--yolo-frequency-hz` | 视频检测上限，默认 10 Hz，取最新帧；扫描图额外检测 |
| `--action-timeout-s` | 单 Action 总超时，默认 120 秒 |
| `--action-stall-timeout-s` | 原地转向无有效进展的上限，当前根配置 1 秒 |
| `--front-blockage-distance-m` | 用于细化墙位置的最远近深度，当前 0.5 m |
| `--blocked-pose-radius-m` | 判定底盘没有脱困的位置半径，当前 0.25 米 |
| `--blocked-pose-duration-s` | 首次停滞的净位移窗口，根配置 3 秒；重试复查 3 秒，无额外起步等待 |
| `--max-unknown-path-m` | 当前剩余路径允许的累计未知长度，默认 1.5 m；超过才取消，0 表示不允许正长度未知段 |
| `--min-localization-quality` | 定位模式最低质量，默认 1 |
| `--camera-source` | `local`（默认）或 `remote`，不改变导航实现 |
| `--camera-endpoint` / `--camera-topic` | ZMQ 地址和主题，默认 `ipc:///tmp/robot-nav-camera.sock` / `rgbd.pose` |
| `--camera-timeout-s` | 等待新消息的上限，默认 3 秒 |
| `--camera-serial` | 本地 USB 模式的 D435i 序列号；远程模式在服务端指定 |
| `--camera-calibration` | 本地 USB 的外参文件；六项固定外参配齐时不再读取，远程使用六项配置 |
| `--debug-frontier` | 打印本轮 Frontier 评分明细 |
| `--rerun-save` | 指定 RRD 录制路径，默认自动创建 |
| `--rerun-viewer record` | 仅录制，不启动实时界面；需开启 Rerun |
| `--no-rerun` | 关闭 Rerun 界面及录制 |

完整参数以 `python -m robot_nav hermes --help` 为准。

该链路使用 Hermes 自带规划和激光避障，但不是独立的功能安全系统。导航
期间必须有人能够立即急停。
