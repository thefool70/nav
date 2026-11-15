# Rerun 排错

面向 Agent；运行命令须先遵守 AGENTS.md 的测试授权规则。

## 快速隔离 Rerun

- 给原命令增加 `--no-rerun`，其余参数保持不变。
- 如果随后出现 `Renderer:`、导航周期和 `stage=`，说明 Habitat、GPU 与导航循环
  已经启动，继续检查 Rerun 初始化。
- 当前 Rerun Web 服务使用端口 `9090` 和 `9877`，且不自动调用 WSL 浏览器；手动
  打开程序打印的 Viewer 地址。
- `recording/visualization/rerun_view.py::RerunVisualizer.__init__` 用 `rr.new_recording` 建立磁盘流并
  `rr.save`；web/native 再创建独立实时流并连接输出。record 模式只有磁盘流。
  所有输出建立后才给 `_recordings` 发送 blueprint 并 `flush(blocking=True)`，避免切换 sink 时争用。
- Rerun 0.22.1 没有顶层 `rr.flush`；使用所持有的 `RecordingStream.flush`。
  当前实现不依赖默认全局记录流。SDK 的 Python 导出和方法
  分别位于 `rerun/__init__.py` 与 `rerun/recording_stream.py`，不要把 Rust 绑定
  函数直接当成 Python 顶层 API。
- Rerun 0.22.1 的 Python 绑定 `serve_web` 持有 GIL 调用 `set_sink`，后者同步等待
  后台线程转发缓存；Python 所有的 Arrow 数据释放又可能需要 GIL。`flush` 的
  绑定会释放 GIL。源码见 [python_bridge.rs](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/src/python_bridge.rs)
  和 [recording_stream.rs](https://github.com/rerun-io/rerun/blob/0.22.1/crates/top/re_sdk/src/recording_stream.rs)。
- 0.22.1 的 `save` 和 `serve_web` 都替换指定流的 sink，不能对同一记录流连调来
  实现双写。磁盘流用独立 UUID；`_log`、`_set_frame_time` 和 blueprint 显式
  写入所有已启用的流，后台 VLM/检测回调也必须为这些流设置时间。SDK 的 `atexit`
  会刷新并关闭所有记录流。
- RRD 默认位于 `data/run_logs/rerun-*.rrd`，`--rerun-save` 指定新文件；禁止
  覆盖已有文件。`--no-rerun` 与预检模式均不创建录制。排错优先使用磁盘 RRD，
  浏览器手动导出的录制可能已经缺少被淘汰的数据。
- Web Viewer 0.22.1 的启动内存上限为 `2_500_000_000` 字节，见
  [web.rs](https://github.com/rerun-io/rerun/blob/0.22.1/crates/viewer/re_viewer/src/web.rs)
  的 `create_app`。服务端上限由 `RERUN_SERVER_MEMORY_LIMIT = "25%"` 控制；
  两者独立，均不限制磁盘录制大小，也不支持在本项目界面里自动从磁盘补帧。
- 如果输出停在“正在启动 Rerun Web 服务”，用
  `ss -ltnp '( sport = :9090 or sport = :9877 )'` 核对监听者。两个端口都由当前
  Python 进程监听，只说明绑定成功，不代表 `serve_web` 已返回；检查上述初始化
  顺序。输出“服务已启动；正在发送界面布局”后停住则检查布局构造和发送。

`habitat_test_scenes/apartment_1.glb` 没有语义描述文件时会打印 `.scn` 或
`info_semantic.json` 警告；只要之后仍出现导航周期，该警告不影响随机感知调试。

## 离线读取 RRD

- `recording/visualization/vlm_trace.py` 只保存轻量摘要，`panels.py` 生成完整卡片，
  `rerun_view.py` 记录卡片和 World 叠加；显示坐标转换在 `view_geometry.py`。`model/vlm/summary` 是简表，`model/interaction` 是最新完整事件；
  `model/vlm/requests/R000001` 保留单次会话，`/text` 保留不依赖字体的完整文本，
  `model/vlm/jobs/J000001` 保留任务事件与接收／排序周期。
- VLM 请求、返回及队列事件均调用 `_begin_sample`，不能将返回事件写回请求帧，否则会在回放中提前出现未来结果。`frame` 现在包含模型和
  队列事件，不等于导航周期；简表的 C 是决策回调序号。
- `recording/visualization/semantic_world.py` 保存每个固定节点：
  `world/observations/J000001/V1` 是拍摄视角，`J000001/F1` 是该请求的评分点；
  停靠成功后直接把对应历史 V 节点标为完成，不新增到达确认节点。V 实体预览单图评分卡，
  `V1/rgb` 保留原图。J/V 点在拍摄朝向前方 0.30m，箭头起点才是真实拍摄位置；
  F 点在冻结目标位置。这些点只在默认布局的 `World history` 页展开。
  `scores` 同时标明 VLM 分与当时 Frontier 分，`navigation` 区分已接收与已用于排序。
- `world/jobs/J000001` 是主图任务点，附带整组评分卡。`_render_job_markers` 在
  真实拍摄位置聚合 0.45m 内的任务，优先当前处理任务，随后活动线索和排队任务；
  普通完成任务只显示最近三个，其余只清空 Position2D，卡片保留。完整 J/V/raw
  链接在 `observations/index`；`observations/focus` 自动放大当前处理视角，独立于
  Viewer Selection。`observation_card.py` 缓存缩略图并画原始像素坐标的 F 锚点。
- `rerun_view.py::_log_world_map` 将翻转的占用图放在 `world/occupancy`，通过
  Transform3D 表达分辨率、原点 yaw、格中心到图像边缘的半格偏移及 World 的 y 取反。
  底图不含 Frontier 轮廓；详细栅格标记仍在 `Map details`。
- 节点只附加 `[Image.buffer, Image.format]`，不附加 ImageIndicator，避免把图片
  直接铺到 World；Points2D 与 AnyValues 提供悬浮字段，Selection 的实体预览显示 RGB。
  Rerun 0.22.1 的点实例悬浮不提供完整图片；如选中了实例，可双击选择整个实体。
- 物体 `object_localization_started/object_progress` 直接更新 Live 与状态面板；
  不等待 `pipeline/navigation_loop.py` 的最终周期回调。`_log_motion_status` 在推理期间优先显示进度，
  不将旧 object.approach 命令显示为仍在执行。物体事件不覆盖普通 J 任务的状态。
- 使用现有环境的 `rerun.dataframe.load_recording(path)`，按 `frame` 时间线查询。
  `navigation/motion:Text` 包含最近决策和实时位姿；
  `world/robot:LineStrip2D` 可恢复更精确位姿。
  world 图层的 Y 已取反，分析时须转回世界坐标。
  `chassis/status:Text` 保存健康与 Action 原始反馈，JSON 正文中的各项 `received_at`
  才是对应采集时刻；同一健康告警会在后续 Action 更新时重复展示，分析时需去重。
  碰撞等 `baseError` 告警目前不展开到 JSONL，排查接近受阻时应同时读取该 RRD 实体。
- `rerun_view.py::_send_default_blueprint` 将 `navigation/live` 放到主图下方 `Live`，
  `navigation/motion` 放到 `Motion details`；侧栏默认是推理单图评分卡与 Observations
  索引，`navigation/frontiers` 在 `Frontiers` 标签页。World 不记录 HUD 点，
  `world/current_frontiers` 保留 ID 作为数据但显式 `show_labels=False`。
  表格编号链接的实例序号必须与该 Points2D 的顺序一致，候选与路径距离是最近一次
  Frontier 刷新的快照；运动中仅更新实时摘要，不重算候选。
- 优先读 `navigation/status_text:Text`；它保留局部待查数、复用数、完整目标和
  执行异常。状态图层帧号对应周期回调，运动帧不重复写状态面板。
- Arrow 表转 Python 对象前省略或转成整数的 `log_time`，避免没有 pandas 时
  纳秒时间戳转换失败。读取既有日志无需启动 Habitat 或请求 VLM。
