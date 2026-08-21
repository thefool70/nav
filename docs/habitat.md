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

## 接入算法

Adapter 已实现 `ChassisInterface`，可直接传入 `run_navigation_cycle`：

```python
state = None
with HabitatChassisAdapter(config) as chassis:
    result = run_navigation_cycle(chassis, goal, state, observer)
    state = result.state
```

这里的 `observer` 必须是具体的 `TargetObserver`。项目尚未绑定某个视觉/VLM，
所以 adapter demo 只验证仿真输入边界，不会假造语义目标结果。

## Adapter 约定

- 内部二维坐标固定为 `x = Habitat x`、`y = -Habitat z`；Habitat top-down
  map 的行方向在 Adapter 内翻转，core 不感知 Habitat 坐标。
- 障碍图只公开机器人附近和相机视野内、未被 navmesh 障碍遮挡的已知区域；
  其他区域为 `None`，供 Frontier 算法探索。
- 相对位姿命令先用 navmesh 检查路径和终点，再瞬时放置到路径终点。这适合
  第一阶段算法迭代，但不是电机、惯性或碰撞过程的物理仿真。
