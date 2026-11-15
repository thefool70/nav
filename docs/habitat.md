# Habitat 仿真

`HabitatChassisAdapter` 把 Habitat-Sim 转换为与真底盘相同的
`ChassisInterface`。导航算法不包含 Habitat 分支；`deployment/habitat/` 只保存环境、
渲染包装和简单的 Adapter 验证脚本。

## 安装

Habitat 使用独立的 Python 3.9 环境，避免图形依赖污染核心环境：

```bash
micromamba env create -f deployment/habitat/environment.yml
micromamba activate robot-nav-habitat
python -m pip install -e .
python -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes \
  --data-path data/habitat \
  --no-replace
```

环境已存在时可按项目配置同步：

```bash
micromamba env update -n robot-nav-habitat \
  -f deployment/habitat/environment.yml \
  --prune
```

当前环境固定 Habitat-Sim 0.3.3、Python 3.9、NumPy 1.26.4、Pillow 10.4.0 和
Rerun 0.22.1。

## 渲染方式

`run.sh` 默认使用 `HABITAT_RENDERER=auto`：先验证 WSL D3D12，失败才回退到
conda `llvmpipe`。当前机器建议显式使用 `gpu`，这样 GPU 链路异常会直接报错，
不会静默使用 CPU。GPU 模式统一加载 Arch 的 Mesa、GLVND 与 DRM 库，避免
conda 中同名库覆盖 D3D12 所需版本。

## 运行导航

在项目根目录运行时，默认读取 `configs/navigation.json` 中的运行参数，命令行参数优先。场景路径可在
`habitat.scene` 中设置；配置规则见 [统一运行配置](../README.md#统一运行配置)。


只检查探索与运动流程时，可以运行随机评分调试模式：

```bash
env HABITAT_RENDERER=gpu deployment/habitat/run.sh \
  python -m robot_nav habitat \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --target "门口" \
  --seed 1 \
  --debug-random-score \
  --max-cycles 100
```

该模式不调用模型，只能检查扫描、Frontier、路径规划、运动和重新选点，不能识别
目标。相同场景和 `--seed` 会使用相同随机起点。

正式语义搜索去掉 `--debug-random-score`。默认连接本机回环地址的 Ollama
`qwen3.5:4b`，无需云端密钥；在开发机运行仿真时，先按
[本地模型说明](hermes.md#随车本地-qwen35-4b) 转发 11434 端口。
提示词使用英文并关闭思考。显式切换云端模型时再配置对应接口、模型和
`ROBOT_NAV_VLM_API_KEY`；请求失败原因会直接打印在终端。

完整物体搜索示例（开发机另一个终端保持上述模型端口转发）：

```bash
micromamba activate robot-nav-habitat
env HABITAT_RENDERER=gpu deployment/habitat/run.sh \
  python -m robot_nav habitat \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --target "沙发" --object-class sofa --seed 1 \
  --object-python /你的模型环境/bin/python \
  --max-cycles 100 --rerun
```

YOLOE 在开发机 `robot-nav` 环境运行，Qwen 通过 SSH 隧道在随车端运行。
运行前需准备 `data/models/yoloe/yoloe-26s-seg.pt`、
`data/models/yoloe/mobileclip2_b.ts`。
固定 seed 可复用相同起点。仿真当前为 320×240 图像，
用于检查算法流程；其耗时不能直接代表真机 848×480、30 FPS 的持续视频负载。

场景搜索将上述命令的目标参数替换为以下参数，并去掉 `--object-class`：

```bash
--search-mode scene --target "洗手间"
```

场景模式必须使用 VLM，不能与 `--debug-random-score` 同时使用。

物体与场景搜索都会加载 YOLOE，只有随机评分模式不加载。模型环境与权重准备见
[Hermes 的模型说明](hermes.md#准备环境和模型)。`--object-python` 指向已有的模型解释器，
仿真仍在 Python 3.9 环境运行。完整搜索规则见 [算法说明](algorithm.md)。

## Rerun 与日志

默认启动 Web Viewer，手动打开终端打印的地址：
<http://127.0.0.1:9090/?url=ws://127.0.0.1:9877>。
Viewer 初始化早于 Habitat；地址尚未打印时，不要直接认定渲染器卡住。

World 显示地图、Frontier、算法目标、Adapter 路径和机器人轨迹；
Inference/Observations 显示实际分析画面，VLM full 显示提示词与模型输出，
Live/Motion details 显示阶段和执行反馈。可视化编号不属于普通 VLM 输入图。

JSONL 默认保存到 `data/run_logs/habitat-*.jsonl`；启用 Rerun 时持续保存 `rerun-*.rrd`。
`--run-log` 和 `--rerun-save` 指定输出路径，RRD 不覆盖已有文件。
`--rerun --rerun-viewer record` 只录制，`--no-rerun` 关闭显示和 RRD、保留 JSONL。
回放使用 Rerun 0.22.1。界面内存淘汰不影响独立磁盘录制，但强制结束可能丢失末尾数据。

## Adapter 行为

- 导航帧显式携带相机离地高度 `sensor_height_m`（默认 1 m），与仿真传感器安装
  位置一致；地面锚点投影使用该外参。相机默认采集分辨率仍为 320×240。
- 内部二维坐标为 `x = Habitat x`、`y = -Habitat z`；top-down map 的行方向在
  Adapter 内完成转换。
- 地图只公开机器人附近和相机视野中具有 navmesh 视线的区域，其余格为未知，
  供 Frontier 算法逐步探索。
- 相对位姿目标先投影到 navmesh，再使用 `GreedyGeodesicFollower` 按 0.25 m
  前进和 10° 转向的离散动作执行。
- 无路径或 follower 无法生成动作时返回明确的可恢复失败，由核心按动作用途处理。
- 动作期间交付帧给记录与感知，不额外推进搜索状态机，也不因普通模型结果中断动作。

## 常见输出

测试场景可能提示缺少 `.scn` 或 `info_semantic.json`。这表示场景没有 Habitat
语义标注，不影响本项目使用 RGB、深度、navmesh、位姿和占用图。

如果输出停在渲染后端之前，先确认已经激活 `robot-nav-habitat`；如果强制 GPU
时 EGL 自检失败，检查 `/dev/dxg`、WSLg、Arch Mesa 和 Windows NVIDIA 驱动。

## 公共导航与路径约束

Habitat 与 Hermes 共用 `pipeline/launch.py` 的感知、日志、回调和导航循环装配；
`platforms/setup.py` 只负责创建各自 Adapter，真机启动前移不用于仿真。
仿真导航也自动保存 `data/run_logs/habitat-*.jsonl`，可用 `--run-log` 指定路径。

受约束动作在开始执行和每个离散动作前，按选点时的地图检查当前位置到目标的
navmesh 剩余路径。未知长度超过 `navigation.max_unknown_path_m`（默认 1.5 米）
就停止下发动作，并通过与 Hermes 相同的失败结果交给核心恢复；
`--max-unknown-path-m` 可临时覆盖。不使用运动中新扩展的可见地图放宽本次约束。
检查的是 navmesh 规划折线；GreedyGeodesicFollower 的离散轨迹可能存在偏差，
不代表两种环境的实际运动轨迹完全相同。
