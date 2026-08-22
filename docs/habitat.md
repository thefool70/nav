# Habitat 仿真

Habitat-Sim 通过 `HabitatChassisAdapter` 接入，与核心算法运行在同一 Python
进程中。Habitat 依赖只存在于独立的 `robot-nav-habitat` 环境，核心环境不受
影响。

## 环境与场景

```bash
micromamba env create -f sim/habitat/environment.yml
micromamba activate robot-nav-habitat
python -m pip install -e .
python -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes --data-path data/habitat --no-replace
```

环境已存在时可用下面的命令同步：

```bash
micromamba env update -n robot-nav-habitat \
  -f sim/habitat/environment.yml --prune
```

`environment.yml` 固定使用 `rerun-sdk==0.22.1`、NumPy 1.26.4 和 Pillow
10.4.0，以兼容 Habitat-Sim 0.3.3 的 Python 3.9 环境。

## 启动 Adapter

先读取一帧，确认 Habitat 能向统一接口提供位姿、RGB、深度和障碍图：

```bash
micromamba activate robot-nav-habitat
sim/habitat/run.sh python sim/habitat/adapter_demo.py \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb
```

`run.sh` 默认自动选择 WSL GPU Mesa 或 CPU `llvmpipe`，并配置无窗口 EGL。
可通过 `HABITAT_RENDERER=gpu` 或 `HABITAT_RENDERER=cpu` 强制选择。原生
NVIDIA EGL 环境需要时，可给 demo 增加 `--gpu-device-id 0`；当前 WSL Mesa
路径默认使用 `-1`。

## 运行完整导航

先把现有 Key 放入 `ROBOT_NAV_VLM_API_KEY` 环境变量，不要写进代码或提交到
Git。然后通过通用项目入口选择 Habitat Adapter：

```bash
sim/habitat/run.sh python -m robot_nav habitat \
  --scene data/habitat/versioned_data/habitat_test_scenes/apartment_1.glb \
  --target "门口" \
  --max-cycles 200
```

入口默认使用 OpenCode Zen Responses API 和
`muse-spark-1.2-contributor-free`。它负责组装 Adapter、目标观察器和周期循环；
`run_navigation_cycle` 仍是环境无关的单周期入口。`sim/habitat/` 只保存 Habitat
环境、渲染包装和 Adapter 验证脚本，不包含导航算法。

完整导航默认启动 Rerun Web Viewer 实时可视化（每个周期记录 RGB、米制深度、
三色占用图、机器人位姿与轨迹、相对控制命令箭头、历史候选点和状态文本），
在 Windows 浏览器打开
[Rerun Web Viewer](http://127.0.0.1:9090/?url=ws://127.0.0.1:9877) 查看；该地址
显式连接 9877 数据端口，同时避免 Rerun 继承 Habitat 的无窗口 EGL 图形环境。
不需要时加 `--no-rerun` 关闭。

当前 adapter demo 仍只验证仿真输入边界，不调用外部模型，也不会假造语义
目标结果。

## Adapter 约定

- 内部二维坐标固定为 `x = Habitat x`、`y = -Habitat z`；Habitat top-down
  map 的行方向在 Adapter 内翻转，core 不感知 Habitat 坐标。
- 障碍图只公开机器人附近和相机视野内、未被 navmesh 障碍遮挡的已知区域；
  其他区域为 `None`，供 Frontier 算法探索。
- 相对位姿命令先用 navmesh 检查路径和终点，再瞬时放置到路径终点。这适合
  第一阶段算法迭代，但不是电机、惯性或碰撞过程的物理仿真。
