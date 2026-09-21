# 本文档已失效

原香橙派（Orange Pi）无线转发链路已由随车笔记本取代，`orangepi` CLI 入口、
`adapters/orangepi/` 与 `hardware/orangepi/` 都已移除。

Hermes 目前保留两种连接方式，共用同一套地图处理、运动监控与导航实现：

- 本地直连：`python -m robot_nav hermes`，USB 直连 D435i。
- 经随车笔记本转发：只把 `--base-url` 指向随车端暴露的 Hermes 地址；转发端
  仅负责设备采集与通讯转发，导航、本地模型与动作监控仍跑在开发机。

两种方式都不需要车载导航服务，也没有失联取消服务。安装、标定与运行见
[Hermes + D435i](hermes.md)。「随车笔记本」的定义见 `CONTEXT.md`。
