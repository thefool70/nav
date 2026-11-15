# robot-nav

零样本目标导航研究项目：给定一个物体或场景名称，让机器人通过观察和探索寻找目标。
使用 YOLOE 和视觉语言模型理解画面，结合地图选择探索方向、定位目标并接近。

目前支持 **Habitat 仿真**和**思岚 Hermes + D435i 真机**，两者共用导航算法。
新底盘通过适配层接入；V550 尚未实现。

## 项目目录

```text
nav/
├── src/robot_nav/   导航算法、感知、设备适配与运行流程
├── configs/         运行配置
├── deployment/      环境安装与启动脚本
├── docs/            架构、算法和使用说明
├── data/            场景、模型、标定与运行记录
└── archive/         归档材料
```

## 阅读入口

| 想了解什么 | 从这里开始 |
| --- | --- |
| 代码分为哪些部分、如何协作 | [架构说明](docs/architecture.md) |
| 没找到目标时往哪走、找到线索后怎么办 | [算法说明](docs/algorithm.md) |
| 在思岚底盘运行 | [Hermes 使用说明](docs/hermes.md) |
| 在仿真中运行 | [Habitat 使用说明](docs/habitat.md) |
| 接入新的底盘 | [底盘接口](docs/chassis-interface.md) |

## 安装入口

Hermes 的基础 Python 环境：

```bash
micromamba env create -f deployment/environment.yml
micromamba activate robot-nav
python -m pip install -e .
```

随后按使用说明准备设备依赖、模型和相机标定。Habitat 使用独立环境，按其使用说明安装。
`deployment/` 的环境文件管理 Python 环境，`pyproject.toml` 定义项目包与依赖。

## 统一运行配置

在项目根目录运行，默认读取 `configs/navigation.json`；命令行参数覆盖配置文件。
配置文件中的相对路径以该文件所在目录为基准，命令行路径以当前目录为基准。

场景、模型和标定需自行准备，运行记录默认保存在 `data/run_logs/`。
具体启动、停止和查看结果的方法见对应环境的使用说明。
