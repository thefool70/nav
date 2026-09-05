# Habitat 仿真

`HabitatChassisAdapter` 把 Habitat-Sim 转换为与真底盘相同的
`ChassisInterface`。导航算法不包含 Habitat 分支；`sim/habitat/` 只保存环境、
渲染包装和简单的 Adapter 验证脚本。

## 安装

Habitat 使用独立的 Python 3.9 环境，避免图形依赖污染核心环境：

```bash
micromamba env create -f sim/habitat/environment.yml
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
  -f sim/habitat/environment.yml \
  --prune
```

当前环境固定 Habitat-Sim 0.3.3、Python 3.9、NumPy 1.26.4、Pillow 10.4.0 和
Rerun 0.22.1。

## 先检查 Adapter

下面的命令读取一帧并打印位姿、地图、RGB 和深度尺寸，不调用 VLM：

```bash
micromamba activate robot-nav-habitat
env HABITAT_RENDERER=gpu sim/habitat/run.sh \
  python sim/habitat/adapter_demo.py \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --seed 1
```

成功时应看到：

```text
Habitat 渲染后端：WSL Mesa D3D12（GPU）。
Renderer: D3D12 (NVIDIA GeForce RTX 3060)
```

`run.sh` 默认使用 `HABITAT_RENDERER=auto`：先验证 WSL D3D12，失败才回退到
conda `llvmpipe`。当前机器建议显式使用 `gpu`，这样 GPU 链路异常会直接报错，
不会静默使用 CPU。GPU 模式统一加载 Arch 的 Mesa、GLVND 与 DRM 库，避免
conda 中同名库覆盖 D3D12 所需版本。

## 运行导航

没有 VLM 凭据时，先运行随机评分调试模式：

```bash
env HABITAT_RENDERER=gpu sim/habitat/run.sh \
  python -m robot_nav habitat \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --target "门口" \
  --seed 1 \
  --debug-random-score \
  --max-cycles 100
```

该模式不调用模型，只能检查扫描、Frontier、路径规划、运动和重新选点，不能识别
目标。相同场景和 `--seed` 会使用相同随机起点。

正式语义搜索先执行 `opencode auth login`，再去掉
`--debug-random-score`。也可以用 `ROBOT_NAV_VLM_API_KEY` 显式覆盖凭据。默认
模型为 OpenCode Go 的 `qwen3.7-plus`，提示词使用英文并关闭 thinking。

场景搜索额外增加：

```bash
--search-mode scene --target "洗手间"
```

场景模式必须使用 VLM，不能与 `--debug-random-score` 同时使用。

## Rerun

导航启动 Rerun Web 服务但不自动打开浏览器，请手动访问：

[http://127.0.0.1:9090/?url=ws://127.0.0.1:9877](http://127.0.0.1:9090/?url=ws://127.0.0.1:9877)

启动时依次显示服务启动、界面布局发送和 Viewer 地址，之后才初始化 Habitat。
若尚未显示 Viewer 地址就停住，应先检查 Rerun 启动阶段。

主要视图含义：

- RGB、深度和当前占用图。
- 绿色 Frontier、黄色选中点、橙色算法命令。
- 红色 Adapter 实际目标、紫色 navmesh 路径、蓝色机器人轨迹。
- VLM 标签页中的实际输入图、提示词、原始输出和解析结果。

World 中不自动显示 Frontier 文字标签，避免遮挡轨迹。侧栏 `Live` 显示随运动帧
更新的位姿和最近决策；`Frontiers` 表格显示候选编号、状态、暂存顺序、路径距离
与分数，点击编号可选中对应点。完整决策详情位于 `Status` 标签页。

开启界面时自动从启动开始持续录制到 `data/run_logs/rerun-*.rrd`，终端显示
完整路径。`--rerun-save <PATH>` 可指定新文件，不覆盖已有文件；`--no-rerun`
同时关闭界面和录制。录制包含图像、地图、状态及默认布局，可在 Rerun 0.22.1
中打开回放。

Web Viewer 默认内存上限为 2.5 GB（约 2.33 GiB）；WebSocket 服务端缓存另有
系统总内存 25% 的上限。内存淘汰旧数据不影响独立写入的 RRD，但界面不会自动
从磁盘补回已淘汰帧。正常退出时 SDK 刷新并关闭录制，强制杀进程或断电仍可能
丢失最后尚未写出的数据；磁盘文件会随运行持续增长。

前往 Frontier 时，黄色点与橙色命令指向同一个最终位置，等待本次移动结束后再决策。
返回父节点时，橙色命令指向本次返回节点，`Live` 显示节点编号、分支深度与
该节点暂存方向数。
区域用 `region:N` 标识，状态面板显示局部 Frontier 待检查点与复用点数；首次
环扫后，只补查含局部可见 Frontier 且覆盖不可复用的方向。`scan basis` 显示
观察依据，具体规则见 [算法说明](algorithm.md)。
`frontier choice` 的 `source=new` 表示优先探索新方向，`source=deferred` 表示
新候选耗尽后逐个返回父节点，遇到仍有有效方向的节点再继续探索，对应 `backtrack.return` 与
`backtrack.resume`；`--debug-frontier` 同时显示返回目标、新旧数量和暂存顺序。
达到 `--max-cycles` 不代表搜索已完成。

## Adapter 行为

- 内部二维坐标为 `x = Habitat x`、`y = -Habitat z`；top-down map 的行方向在
  Adapter 内完成转换。
- 地图只公开机器人附近和相机视野中具有 navmesh 视线的区域，其余格为未知，
  供 Frontier 算法逐步探索。
- 相对位姿目标先投影到 navmesh，再使用 `GreedyGeodesicFollower` 按 0.25 m
  前进和 10° 转向的离散动作执行。
- 目标无法投影、没有路径或 follower 无法生成动作时，Adapter 报告可恢复运动
  失败；Frontier 移动会淘汰当前目标并继续，返回父节点失败则跳过该返回节点，
  从实际位置重新观察，其有效 Frontier 继续作为候选。
- 返回动作结束后距父节点须不超过 0.25 m；超出容差时以 `motion.backtrack_recovered`
  记录原因并继续搜索，不重复发送同一失败节点的返回命令。
- 动作中间帧会送入 Rerun，但不会额外推进算法状态或调用 VLM。

## 常见输出

测试场景可能提示缺少 `.scn` 或 `info_semantic.json`。这表示场景没有 Habitat
语义标注，不影响本项目使用 RGB、深度、navmesh、位姿和占用图。

如果输出停在渲染后端之前，先确认已经激活 `robot-nav-habitat`；如果强制 GPU
时 EGL 自检失败，检查 `/dev/dxg`、WSLg、Arch Mesa 和 Windows NVIDIA 驱动。
