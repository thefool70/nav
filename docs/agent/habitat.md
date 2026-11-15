# Habitat 启动排错

面向 Agent；安装和运行见 [Habitat 使用说明](../habitat.md)。未获测试授权时只读代码与已有日志。

- 启动顺序为 JSONL → Rerun → 感知/模型 → Adapter。包装脚本打印渲染后端不表示已经创建 Habitat；
  尚无 Viewer 地址时先定位 Rerun，见 [可视化排错](visualization.md)。
- `deployment/habitat/run.sh` 的 `auto` 先检查 WSL D3D12，失败才回退 llvmpipe；
  强制 `HABITAT_RENDERER=gpu` 时失败直接退出。Arch Mesa/GLVND/DRM 与 conda 同名库的混用会影响 EGL。
- Codex 沙箱可能隐藏 `/dev/dxg`，不能单凭沙箱不可见断定宿主无 GPU。
- Habitat 环境 Python 3.9，基础环境 Python 3.11。RealSense 二进制扩展必须匹配解释器；
  `deployment/hermes/run.sh` 只准备 USB 和库路径，不切换解释器。
- `.scn`、`info_semantic.json` 缺失表示场景没有语义标注，不妨碍本项目使用 RGB-D/navmesh。
- Habitat 按视场与 navmesh 视线公开地图；Hermes 由激光历史生成地图并按相机 FOV 更新探索缓存。
  比较 Frontier 差异先检查实际 `NavigationFrame.obstacle_map`，不能假设两边输入等价。
- 两种环境共用 `pipeline/launch.py::_assemble_and_run`、感知与搜索；Hermes 另有启动前移。
  路径约束均使用决策地图，Habitat 在每个离散动作前检查 navmesh 剩余路径。

## 原生 Linux NVIDIA 服务器

- SSH 别名 `PKU-2-4090-48G` 使用 `zmj`。独立仿真环境为
  `/data0/zmj/steepgs/downloads/nav-sim-20261115/env`，源码副本在同级 `repo/`。
  本地源码修改不会自动同步到该副本。
- 原生 NVIDIA 使用 Habitat-Sim 0.3.3 的 headless + withbullet 构建；直接运行环境内 Python，
  不套用依赖 Arch/WSL D3D12 的 `deployment/habitat/run.sh`。
  `CUDA_VISIBLE_DEVICES=1` 时 Habitat 的 `--gpu-device-id 0` 指向所暴露的 GPU。
- 开启 Rerun 后若 llvmlite 报系统 `libstdc++.so.6` 缺少 `GLIBCXX_3.4.29`，
  在该次命令设置 `LD_PRELOAD=<仿真环境>/lib/libstdc++.so.6` 可解决库加载冲突；不替换系统库。
- 官方测试场景亦可从 `https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip` 获取；
  包内场景路径为 `data/scene_datasets/habitat-test-scenes/apartment_1.glb`。
