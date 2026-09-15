# Habitat 仿真排错

## 启动顺序

`robot_nav habitat` 会先在 `src/robot_nav/__main__.py::_build_visualization`
构造 Rerun，再创建观察器和 `HabitatChassisAdapter`。因此，仅看到启动脚本打印的
“Habitat 渲染后端”时，还不能判断 Habitat 或 GPU 卡住。

从 Habitat 切换到 Hermes 时，先切换到 `robot-nav` 环境：

- `robot-nav-habitat` 使用 Python 3.9，`robot-nav` 使用 Python 3.11。
  当前 `hardware/realsense/.rsusb/python/pyrealsense2.cpython-311-*.so` 是
  Python 3.11 扩展；3.9 无法导入，即使 USB 已接入也会报模块不存在。
- `hardware/slamtec_l515/run.sh` 只注入 RSUSB 库路径，不切换 Python 环境。
  遇到此错误先核对 traceback 的解释器路径和扩展文件名，再选择匹配的环境；
  不要直接在 Habitat 环境安装另一套 RealSense 库。

## 仿真与真机差异的回归定位

- 当前三个 CLI 入口都使用 `QueuedSemanticObserver`；仅
  `__main__.py::_build_slamtec_observer` 的正常物体模式传入 YOLO/SAM2
  `local_observer`。队列本身满足 `ContinuousTargetObserver` 协议，不能仅凭
  `isinstance` 判断本地检测已启用。Hermes 本地检测走接近与确认；后台线索
  在三个入口都走返回拍摄位姿。直接使用同步观察器的 API 仍保留同步状态分支。
- S100 的沿途帧来自 `adapters/s100_l515/adapter.py::_publish_stopped_frame`：
  每个最多 0.20 m 的平移小段停止后取帧；Habitat 在每个离散动作后取帧，Hermes
  另有连续采集线程。共用预采样器不表示三个 Adapter 的采集时机相同。
- S100 `adapters/s100_l515/planner.py::plan_known_free_path` 禁止未知格，
  无路时抛普通 `RuntimeError`；Adapter 未转换为 `RecoverableMotionError`，
  因而不会进入 `app.py::_execute_command` 的普通运动失败恢复。比较返回失败
  行为时先查异常类型，不能仅凭共用状态机推断 S100 会自动尝试下一条线索。
- 检测链先检查 `__main__.py::_run_habitat` 与 `_build_slamtec_observer`，不能仅凭
  共用 `core/navigator.py` 判断算法一致。`a99543a`（2026-08-26）首次只在 Hermes
  入口包装 SAM2；其父版本两边均调用 `_build_observer`。`c16e919` 仅提供 SAM2
  实现，入口实际启用点是 `a99543a`。
- `89129dd`（2026-08-26）只给 Hermes 接入 YOLO+SAM2 持续检测、运动中断与
  启动前移 1 m；Habitat 的运动循环没有对应目标中断入口。
- 地图输入从 Hermes 接入的 `33d8cd5`（2026-08-25）起就不等价：当时直接转换
  厂商地图，Habitat 已按局部视场和视线公开地图。`e42900e` 加入 Hermes 的
  L515 地图筛选与膨胀，`c7a27ed`（2026-09-04）又移除深度截断，改为理论 FOV。
  排查 Frontier 数量差异时先比较两边实际 `NavigationFrame.obstacle_map`。
- `9ab3402` / `c7a27ed`（2026-09-04）增加共用的 object/scene 状态与入口；
  场景模式本身不需要 YOLO+SAM2，不能把模式差异误判为环境差异。
- `87a1f24`（2026-09-05）在 `app.py::_execute_command` 引入已知区路径扩展，
  仅 Hermes 实现；未知路径取消与区域屏蔽没有在 Habitat 中同等触发。
  历史日期为 Git 提交日期；可用 `git show <commit> -- <path>` 核对实际接入位置。

## 快速隔离 Rerun

- 给原命令增加 `--no-rerun`，其余参数保持不变。
- 如果随后出现 `Renderer:`、导航周期和 `stage=`，说明 Habitat、GPU 与导航循环
  已经启动，继续检查 Rerun 初始化。
- 当前 Rerun Web 服务使用端口 `9090` 和 `9877`，且不自动调用 WSL 浏览器；手动
  打开程序打印的 Viewer 地址。
- `visualization/rerun_view.py::RerunVisualizer.__init__` 先 `rr.init` 创建实时流、
  独立创建磁盘流并 `rr.save`，再 `rr.serve_web`，之后才给两个流发送 blueprint
  并 `flush(blocking=True)`。服务建立前不要发送布局或其他 Arrow 数据。
- Rerun 0.22.1 没有顶层 `rr.flush`；使用 `rr.get_data_recording()` 取得当前
  `RecordingStream`，非 `None` 时调用其 `flush` 方法。SDK 的 Python 导出和方法
  分别位于 `rerun/__init__.py` 与 `rerun/recording_stream.py`，不要把 Rust 绑定
  函数直接当成 Python 顶层 API。
- Rerun 0.22.1 的 Python 绑定 `serve_web` 持有 GIL 调用 `set_sink`，后者同步等待
  后台线程转发缓存；Python 所有的 Arrow 数据释放又可能需要 GIL。`flush` 的
  绑定会释放 GIL。源码见 [python_bridge.rs](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/src/python_bridge.rs)
  和 [recording_stream.rs](https://github.com/rerun-io/rerun/blob/0.22.1/crates/top/re_sdk/src/recording_stream.rs)。
- 0.22.1 的 `save` 和 `serve_web` 都替换指定流的 sink，不能对同一记录流连调来
  实现双写。磁盘流用独立 UUID；`_log`、`_set_frame_time` 和 blueprint 显式
  写入两个流，后台 VLM/检测回调也必须为两个流设置时间。SDK 的 `atexit`
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

## Frontier 与观察记录排错

- 地图上的绿色轮廓来自 `rerun_view.py::_update_frontier_markers` 遍历每个
  候选的全部 `frontier_cells`，绿色格数不等于导航目标数。`Frontiers` 表格每行
  才是一个候选；JSONL 的 `frontier_candidates` 长度与各项
  `frontier_cell_count` 可分别核对候选数和边界格数，截图须关联周期后才能比较。
- `frontier.py::find_frontier_candidates` 按八邻接保留完整连续边界，不按长度
  或跨度拆分。`_merge_frontier_fragments` 仅按 0.30 m 自由区短路径和未知侧
  朝向判断断段是否合并，不限制总跨度；合并后统一过滤跨度小于 0.50 m 的区域。
  每个有效区域只产生一个移动代表点，`frontier_cells` 保留完整边界供扫描与匹配。
  排查连续边界被分成多个候选时，先核对边界是否实际八邻接连通和候选刷新周期。
- `core/navigator.py::_refresh_frontier_regions` 重提有效边界；
  `core/history.py::match_frontier_regions` 用世界坐标边界关联 ID。历史节点只保留
  实际尝试的 Frontier 移动，节点的出发位置也是本轮未选方向的父节点。
- `FrontierRegion.deferred_order` 保存暂存时的（观测节点序号，候选排名），与
  `FrontierCandidate` 同步；序号索引 `observation_history`，同节点排名小的先恢复。
  新候选耗尽后，`_begin_backtracking` 只取 `branch_node_ids[-1]` 对应节点的
  `position_world_xy`，用 `backtrack_node_id` 固定返回目标；不能按全局暂存队列
  找更早节点并直接跳过去。`backtrack.return` 不新增历史、不消耗暂存队列。
  `history.py::defer_unselected_frontiers` 仅在真正选定命令后暂存本轮其他新方向，
  等待 VLM 评分时不得提前暂存。选中区域解除暂存，沿该方向继续产生的边界可优先探索。
- `SearchState.branch_node_ids` 与完整 `observation_history` 分开：提交 Frontier
  命令时压入出发节点，正常返回到达且没有有效暂存方向后出栈；返回失败的节点
  通过 `_recover_backtrack_issue` 单独跳过，并释放它的暂存方向。新分支沿
  剩余栈继续压入；不能简单倒序遍历全部历史，否则会重新走进已经退完的旧分支。
  恢复方向时新的移动记录可能与原节点同位置；容差内的节点直接检查，无额外返回动作。
- `_continue_backtracking` 只在同步返回动作结束后的新周期运行；与父节点距离
  不超过 `BACKTRACK_ARRIVAL_M = 0.25` 才刷新边界并检查本节点方向，否则
  `_recover_backtrack_issue(issue_kind="not_arrived")` 返回 `OK + SCANNING`，
  不得直接置为 `FAILED`，也不重复下发同一失败节点。起点已在容差内则直接恢复。
  没有有效暂存方向则出栈；若到达位置显露新候选则重新选点，否则只返回上一层。
  即使全局候选为空，也先逐级退完当前分支，再等待视觉队列，才 `explore.exhausted`。返回开始时清空本轮
  `scan_evidence` 与扫描计划，避免在父节点沿用分支末端图像评分；`observed_views` 仍保留。
- 地图更新造成旧区域自然分裂时，子区域继承暂存顺序；合并时保留最先恢复的顺序。
  实际边界重叠优先于一格邻域匹配，避免相邻旧区域把当前分支一起暂存。
  区域消失或没有可达代表点时删除，恢复目标取当前代表点而非历史坐标。
- Rerun 在 `explore.select` / `backtrack.resume` 使用相同坐标显示黄色 Frontier
  与橙色命令；`backtrack.return` 的 `destination_world_xy` 是父节点位置，
  不记录选中 `candidate_id`。`Live` 和终端显示本次返回节点、`branch_depth` 与
  `pending_direction_count`，数量为发出命令时的快照，到达后须刷新。
  JSONL 状态保存 `backtrack_node_id` 和完整 `branch_node_ids`，可核对返回顺序。
  `app.py::_execute_command` 等待同步动作结束后才允许下一周期决策；
  排查途中改目标时先核对 Adapter 完成反馈。
- Hermes 的 `KnownSpaceChassisInterface` 扩展在 `explore.select`、
  `backtrack.return`、`backtrack.resume` 和 `target.revisit` 使用。
  `app.py` 将本周期 `frame.obstacle_map` 和 `reference_pose=frame.pose` 显式传给
  `send_relative_pose_in_known_space`；启动、标定和目标接近仍走普通接口，
  Habitat/S100 当前未实现该扩展。
- Hermes `_send_relative_pose` 用 `reference_pose` 还原世界目标与目标 yaw，
  `start_pose` 仅计算剩余距离和反馈。排查“Action 完成但未到达父节点”时对照
  `cycle_decision.command.target_world_xy`、`Hermes command target` 和实际终点。
  已确认日志 `slamtec-l515-20260905-214649-157351.jsonl` 第 38 周期两次朝向相差
  约 8.7°，旧实现把目标偏移约 0.36 m，最终距原父节点 0.308 m，触发旧终止分支。
  当时仍有有效暂存候选 `region:4`（顺序 `1:1`）和 12 个回退节点，不能归因于
  `explore.exhausted`；已屏蔽的 3 个区域不计入有效待探索候选。
  这不能仅靠放宽 0.25 m 容差掩盖；平移完成的 `target_error` 仅相对底盘收到的目标。
- `core/path_validation.py::measure_unknown_path_length` 按格边交点切分每段路径，
  累加未知部分的实际长度；`None` 和地图外算未知。只触碰格角长度为零，沿格边
  行走时任一侧未知即计入一次。不能用格数×分辨率代替斜线/部分穿格的长度。
  Hermes `_check_known_space_path` 将当前位置
  加在剩余路径前，使用选点地图快照；不能改用原始 Hermes 地图、最新运动帧地图，
  或仅检查离散路径点，否则会漏掉未知绕行。
- `SlamtecL515Config.max_unknown_path_m` 与 CLI `--max-unknown-path-m` 默认均为
  1.5 m，非负有限数。只在未知长度 > 上限时取消（比较保留 1e-9 m 浮点容差）；
  多段未知区累加，不取最长连续段，不累加不同轮询或已走过的路。0 禁止正长度
  未知段；地图快照仍固定到选点时刻，不能改成运动中扩展的可见地图。
- Action 进度 `unknown_path=长度/上限` 为当前轮询测量；取消异常保留
  `unknown_length_m`、`limit_m`、`total_path_length_m`，`app.py` 将其写入
  `unknown_path_length_m`、`unknown_path_limit_m`、`checked_path_length_m`。
  这些诊断不改变超限后的区域屏蔽、返回恢复和线索放弃流程。
- Hermes 路径检查在 `_monitor_action` 的图像采集和进度节流之前执行，不能受
  `on_motion_plan` 或 `path_error_reported` 控制。空路径继续轮询，路径读取错误
  则取消并停止运行，不能用可视化路径读取的容错分支掩盖检查失败。
- 未知路径长度超限、Action 执行失败、停滞或目标检测中断触发取消后，均用
  `wait_for_action(require_success=False)` 确认 Action 进入终态；确认超时使用
  `request_timeout_s`（默认 5 秒）。取消或
  确认失败不得转换为可恢复异常，否则下一目标可能覆盖仍在执行的动作。成功后
  将异常及其完整 `path_world_xy` 送回 `app.py`；必须先于父类
  `RecoverableMotionError` 捕获，不能只转成字符串丢失路径。
- `recover_from_motion_failure(rejected_path_world_xy=...)` 将尝试标为
  `INVALIDATED`，并把当时完整世界边界存入
  `SearchState.blocked_frontier_regions`。取消结果记录 `rejection_scope=region`、
  `rejected_path_world_xy`；首个未知格仍保存在 Action 日志和历史 `execution_reason`。
  这适用于 `explore.select` / `backtrack.resume`；`backtrack.return` 的执行失败、
  停滞或未知路径长度超限均调用 `_recover_backtrack_issue`，不能据返回失败屏蔽未尝试的区域。
  恢复从 `branch_node_ids` 弹出失败节点，将其所属区域的 `deferred_order` 清为
  `None`，保留边界和整片屏蔽记录，再由下一周期按实际帧重新检查。日志
  `motion.backtrack_recovered` 记录失败种类、跳过节点、释放区域及可用的位置误差。
  返回被目标检测中断时，`_release_interrupted_direction` 清除回退状态并转扫描，
  保留所有暂存方向供后续恢复。
- `_refresh_frontier_regions` 在区域 ID 关联之前调用
  `history.filter_blocked_frontier_regions`。被屏蔽边界独立于候选保存，按世界格心
  与自由区一格邻域匹配，并保留所有匹配过的完整边界。不能只屏蔽失败坐标附近
  `max(0.4 m, 2 格)`，也不能只靠旧 `region_id`，否则换代表点或分裂会反复下发。
- 区域屏蔽在本次运行中持续有效，不复查旧路径、不因地图更新自动解除。
  `BlockedFrontierRegion` 只保存区域 ID 与边界；被拒绝路径仅写入取消日志。
  新动作仍用自己的选点地图检查实时路径；普通失败、停滞、目标检测中断不额外
  屏蔽整片区域。Rerun `Frontiers` 与 JSONL `blocked_frontier_regions` 可查屏蔽记录。
- `SearchState.scan_observation_points` 来自本轮有效 Frontier 的全部边界格，
  新候选和暂存候选均参与；观察点与朝向在整轮扫描期间冻结，不因途中边界消失
  而重排。`observed_views` 只在有效本地观测或成功后台目标判断后增加；相机采集、
  地图公开和 Rerun 运动帧均不代表 VLM 已检查。
- 扫描未指向 Frontier 代表点时，先检查 `observation_coverage.py` 的
  `frontier_observation_points`：只保留光心距离大于 0.10 m、至多 4 m 且地图
  视线可达的边界格。`scan.py::build_unobserved_scan_headings` 合并视场后，
  镜头中心可在多个边界点之间。首次仍环扫；之后无局部待查点时物体模式跳过扫描，
  场景模式只采集当前朝向。日志 `scan_mode` 分别为 `initial`、`frontier`、
  `scene_current_view`，Rerun `scan basis` 显示对应原因。
- `core/observation_coverage.py::_local_coverage_points` 的 0.25 m 固定世界
  网格和首层未知格仅用于记录已检查图像，不触发扫描。`capture_observation_view`
  必须同时接收冻结的 Frontier 观察点：覆盖复用按世界坐标匹配，边界格心未必落在
  固定网格上；省略这些点会导致重复检查。未知格后方不计为已观察。
- `scan_local_point_count` / `local_observation_point_count` 表示覆盖过滤前的
  局部可见 Frontier 点数，`observation_point_count` 为本轮待查数；不代表普通
  已知区域的采样数。Rerun 显示为 `frontier coverage`。
- `ObservationView.map_visible_world_xy` 是当时地图可见的相机视锥，仅在光心
  相距 0.10 m 内复用；`visible_world_xy` 还经过 RGB-D 投影与遮挡检查，才能
  跨位置复用。新暴露区域、超过 30° 的观察角度变化或明显接近会触发补查。
  `depth_coverage_available=False` 时不得扩大原地复用范围来消除重复扫描。
- JSONL 的 `scan_views` 与 `last_checked_view` 记录真实光心位置、时间戳和
  覆盖点数。异步的 `PENDING` 证据不代表已检查；要结合全局 `observed_views`
  和 `pending_observation_view_count` 判断。同步场景仍须等 `assess_scene` 成功。
- `--seed` 只传给 Habitat；随机观察器使用独立、未设种子的 `random.Random()`。
  随机评分运行可检查状态流，不能据单轮轨迹归因 VLM 效果或衡量算法改进。
- 排查“没有优先向前”时检查 `navigator.py::_select_exploration_target`：当前排序
  先分新候选与暂存候选，只有新候选进入 `FrontierScoreRequest`；没有新候选时
  才逐个回到父节点，遇到该节点的有效暂存方向按原顺序恢复，不再使用区域切换扣分。
  扫描结束时的车头 yaw 不参与分层。`frontier_selection_source` 为 `new` 或
  `deferred`；提交 Frontier 移动时，新旧数量是本次选择前的数量；
  JSONL 状态的 `deferred_frontiers` 是选择后的暂存队列，Rerun 同样显示来源与队列数量。
- 同步场景判断 `uncertain` 时保留本轮采集；异步目标返回只核对位置和朝向。物体模式
  全部覆盖可复用时跳过扫描，异步选点仍可查询严格匹配的评分缓存；场景模式仍采集
  当前画面入队。局部扫描未拍到的 Frontier 方位不投影到图片边缘。
- `INVALIDATED` 表示执行失败，`STALLED` 表示停滞；二者均不证明 navmesh
  不连通。有紫色路径但无运动帧时，继续检查 Habitat `_follow_path()` 的动作生成。
  周期回调发生在执行前，异常原因通过历史方向的 `execution_reason` 保留，
  下一周期 Rerun 的 `last exploration issue` 显示最近一次异常及其节点编号。

## 异步视觉队列

- 连接故障先读 `semantic-*/job-*/result.json` 的 `detection_error`，区分 HTTP
  状态码与收到响应前断开。已确认 2026-09-07 的 `semantic-cri8lcr5` 共完成
  34 批：7 批成功、27 批连接失败（25 次 RemoteDisconnected、1 次 Broken pipe、
  1 次 SSL EOF）；job-000004～000010 成功，job-000011 起连续失败。
  此类日志不能直接证明密钥失效、额度耗尽或 Qwen 全局宕机。
  `_post_json` 使用默认 urllib opener，会继承启动进程的 HTTP(S) 代理设置；
  排查 OpenCode Go 链路时核对该进程的代理环境，不用当前工具进程环境替代历史证据。
  不输出凭据值，也不要把连接失败当作“未检测到目标”。
- `VlmInteraction.context` 与 `SemanticAnalyzer.analyze_views(trace_context=...)`
  只增加记录来源，不参与提示词或 HTTP payload；不能为了显示关联关系串用
  `_scan_images` 或共享可变的当前任务 ID。后台目标返回不另发视觉请求；本地或
  同步物体定位请求记录其前置可见性请求 `parent_request_id`。
- `QueuedSemanticObserver._score_origins` 随评分缓存保留 J/R 来源；
  `semantic_received_jobs` 是该周期接收结果，`semantic_score_sources` 是传给
  排序的有效缓存输入。两者不是同一件事，返回有效分数不等于已用于导航。
- `adapters/queued_semantics.py::QueuedSemanticObserver` 是 CLI 的默认观察器。
  扫描线程冻结画面；运动回调只提交最新原始帧，由独立采样线程处理；单个分析
  线程按 FIFO 调用 `OpenAICompatibleTargetObserver.analyze_views`，不使用
  `_scan_images`。设备只由 Adapter 读取，核心状态只由主循环更新。
- `data/run_logs/semantic-*/job-*` 保存 `view-N.rgb.gz`、`snapshot.json` 和
  `result.json`。快照中的 `snapshot:批次:序号` 与 `source_region_id` 分别是
  请求内候选 ID 和原区域 ID；分数缓存必须同时匹配地图 frame、原区域 ID、
  世界目标坐标，不能只按新分配的预览 ID 套用。预览不提交核心区域编号。
- 扫描中出现连续单图批次时，先检查 `snapshot.json.source`。`_capture_scan`
  正常只在 `context.index + 1 >= context.count` 时提交整轮，不按新 Frontier
  提前拆批。被打断或重建计划时，`sync_state` / 下一轮开始处仍提交未送出的
  部分画面；后续局部扫描本身可能只有一图，不能只凭图数判断是否重复请求。
- `app.py::_execute_command` 单独调用 `set_motion_prefetch_enabled`：仅允许
  explore.select / backtrack.return / backtrack.resume，且命令
  必须有非零平移；在 finally 关闭。`set_motion_interrupt_enabled` 只转发本地
  检测中断开关，不能再控制 VLM 预采样，否则 scan.turn 会额外产生单图 motion。
  target.revisit 不启用预采样。动作期间已接收的帧保留其原始状态快照，允许
  采样线程稍后完成处理。
- `frontier_overlay.py::build_semantic_analysis_sheet` 保留所有输入图像，包含
  没有 Frontier 的视角；V/F 编号在图外，F 用单像素引线连到图内 3×3 锚点。
  评分失效不能删除未检测任务。RGB 最大宽度 320；V 蓝底白字用连续笔画抗锯齿，
  不依赖系统字体。页眉 28px，页脚按标签分配 0–3 行、每行 30px；每排按最高
  tile 对齐，间距 2px。无 F 时不得恢复固定 96px 空页脚。
  `core/vision.py::parse_semantic_analysis_response` 分别校验线索列表和评分。
- `core/frontier_projection.py::project_frontier_ground_points` 在采集时将候选
  `world_xy` 按局部地面 z=0 投影；变换为 `grounding._camera_point_to_robot` 的
  逆变换（撤销 yaw/pitch/roll）。中心有效、3×3 至少三点有效，中位值及中心
  光轴深度误差 ≤0.15 m，近处遮挡会过滤。不要用欧氏距离比较深度。
- `BufferedScanImage.frontier_projections` 与完整相机外参写入 `snapshot.json`；
  `_read_snapshot` 还原为元组。锚点按同一世界坐标匹配，不依赖队列重命名的
  candidate ID。拼图选择已通过深度检查的视角；没有可靠锚点或页脚容量不足
  就省略该候选评分，不能恢复旧的画面边缘兜底。
- `has_frontier_direction_in_view` 只决定运动帧采样触发；可见方位没有可靠地面
  锚点时仍提交画面检测目标，不能因评分不可用而删除这次检测。入队候选只含
  有锚点者，避免为未请求的评分占用在途缓存键。
- 已确认 `semantic-5zgxc5zj` 的前 16 个完成任务均为 `candidates=[]`、
  `frontier_projections=[]`、`frontier_scores={}`。对应
  `slamtec-l515-20260906-182020-217885.jsonl` 前 33 个决策的 70 条候选记录均无
  语义分；不能归因于界面将请求拆开。前两批 source=motion 说明采样前有可见
  Frontier 方位，评分入口随后被地面投影/深度检查过滤。现有快照不记录淘汰
  原因，尚不能区分投影出界、无效深度、地面不符或外参问题；不要直接断言阈值过严。
- `build_frontier_score_sheet` 保留旧同步 API 的数字方向标注；默认 FIFO 联合
  分析走 `build_semantic_analysis_sheet` 的地面锚点。排查时先确认实际请求 task。
- Habitat `_build_navigation_frame` 显式填写 `CameraExtrinsics(height_m=sensor_height_m)`；
  不得再使用默认零高度，否则地面锚点无法生成。采集分辨率没有提升。
- Rerun 的 F 记录新增 `view_id`、`source_pixel_xy`，后者是缩放前 RGB 像素；
  用快照 `camera_depth_m` / `observed_depth_m` 追查过滤，不能拿返回时的新深度核对。
- 提示词 `_frontier_scoring_rules` 为联合分析和旧批量评分共用；0.5 表示缺少线索，
  `target.view_ids=[]` 不自动压低评分，模型不计算导航距离/可达性。模型输出
  `target.view_ids` 有序整数列表，不再输出 `target.found` / `target.view_id`。
  `SemanticAnalysis.target_view_ids` 空元组是有效无线索，None 才是检测失败；
  越界、布尔值、重复编号、非列表均判为检测失败，仍保留有效分数。
  列表只包含检测匹配的画面；到位后没有二次视觉确认，不能再提示模型将模糊猜测
  留给现场复查。Rerun 的 `Return order` 是模型返回顺序，非新的世界候选编号。
- `prepare_cycle` 接收结果：检测有效才登记覆盖，评分失败仍可保留目标线索。
  `_pending_coverage` 在结果被主循环接收前一直保留；失败后移除，不冒充已检查。
  `pending_semantic_jobs` 还含采样、待接收结果、部分扫描和排队线索，不是 HTTP 数。
- `TargetClue.pose` 是拍摄时机器人位姿。`target.revisit` 返回后恢复拍摄朝向，
  `_continue_target_clue(frame, state)` 在位置误差 ≤0.25m、朝向误差 ≤5° 时直接
  返回 COMPLETE / target.revisit_complete，细节保留拍摄位姿目标与剩余误差。
  不进入 VERIFYING_SCENE 或 LOCALIZING_TARGET，也不再调用新帧复查接口。
  `prepare_cycle` 按列表原顺序入队，
  位姿去重不重新排序；批次仍按 FIFO。`_discard_target_clue` 返回无命令状态，
  返回失败时以 target.revisit_failed 让下一周期优先取下一条，COMPLETE 不再取线索。
  `navigate` 的 active_target_clue 分支优先于视觉处理；`app.py` 跳过返回期间
  的目标观测，同时禁用 target.revisit / target.revisit_turn 的本地检测中断。
- `QueuedSemanticObserver.sync_state` 在提交残余扫描前更新 `_background_paused`：
  有待查线索、active_target_clue、目标返回/本地定位/确认阶段或导航终态时暂停。
  `_worker_loop` 与 `_capture_loop` 等待暂停解除；`submit_motion_frame` 不接收
  新预采样帧。`prepare_cycle` 根据传入状态与待查线索决定是否接收完成结果，
  暂停期间保留 `_completed` 与 `_pending_coverage`，不抢占当前线索。
  两条线索之间不恢复后台工作；现有线索都无法返回、回到普通搜索时 notify_all 恢复。
  已开始的请求/采样可以完成并保存结果；暂停不是取消已发送的 HTTP 请求。
  `semantic_background_paused` 进入周期诊断，Rerun 简表显示普通队列暂停提示。
- `QueuedSemanticObserver.assess_scene` 仅满足通用观察器协议，正常异步导航不调用；
  场景判断由联合检测完成。同步观察器仍可用 `assess_scene(goal)` 判断扫描拼图；
  本地物体检测保留接近与最终确认，后台线索不进入这条路径。
- 分数只在下一次决策读取，不修改执行中的目标；本地 YOLO 中断逻辑仍保留。
  `_wait_for_semantics_or_finish` 必须等待任务与线索耗尽；失败检测计数单独报告。
  `close` 停止继续出队，已落盘数据保留，当前没有恢复队列或磁盘限额机制。

## 导航退出路径排查

- Hermes 静止检测默认 8 秒，`__main__.py` 的 `--action-stall-timeout-s` 与
  `SlamtecL515Config.action_stall_timeout_s` 必须同步。`_monitor_action` 仍按
  默认 2 秒进度采样检查，另受帧采集与 REST 延迟影响，不能理解为精确第 8 秒取消。
  MoveToAction 的有效进展仍是平移至少 2 cm，原地转向不重置其平移停滞计时。
- `__main__.py::_run_navigation` 在目标完成、`FAILED`、非 `OK` 且非等待感知状态，
  或达到 `max_cycles` 时退出；`MISSING_DATA` 当前也立即退出。周期上限不能当作
  Frontier 耗尽；`OK + WAITING_FOR_SEMANTICS` 每次最多等待结果 1 秒，不消耗决策
  额度。普通模型失败记为失败批次、评分降级；本地物体最终确认失败沿用 `uncertain` 重试。
- `core/navigator.py::recover_from_motion_failure` / `continue_after_motion_stall`
  对 `scan.turn` 使用 `_recover_scan_turn`，清除扫描计划后按真实朝向重新规划，
  不登记失败方向的覆盖。首次扫描尚未完成时仍按首次环扫规则重建。
- `__main__.py::_move_slamtec_forward_on_start` 在 `_run_navigation` 之前直接调用
  底盘；前移 1 m 的可恢复失败/停滞在确认动作结束后继续启动，其他异常仍停止。
- `slamtec_l515/adapter.py::_execute_action` 将 Action 创建、起始位姿读取和监控
  放在同一异常范围；`_cancel_active_action` 与 `close` 负责取消遗留动作。已获得
  ID 时还要确认终态；创建请求失败而没有 ID 时只能尝试取消，原异常仍终止运行。
- CLI 的 `_optional_callback` 停用失败的 Rerun 周期/运动回调；Hermes 进度回调
  与 JSONL `_write` 也隔离记录错误。直接调用 `run_navigation_cycle(on_cycle=...)`
  的自定义回调仍由调用方负责。采集、路径检查、健康错误仍停止并收尾动作。
- 正式扫描的快照落盘失败仍传播并停止；运动预采样失败记录 `prefetch_skipped`，
  不登记已检查。`snapshot_failed` 与普通日志丢失不同；结果落盘失败时内存结果
  仍交给主循环。队列停止后运行中的 HTTP 可能完成，退出不自动续跑剩余任务。

## 离线读取 RRD

- `visualization/vlm_trace.py` 只保存轻量摘要，`rerun_view.py` 写完整卡片和 World
  叠加。`model/vlm/summary` 是简表，`model/interaction` 是最新完整事件；
  `model/vlm/requests/R000001` 保留单次会话，`/text` 保留不依赖字体的完整文本，
  `model/vlm/jobs/J000001` 保留任务事件与接收／排序周期。
- VLM 请求、返回及队列事件均调用 `_begin_sample`，不能恢复旧 `_vlm_samples`
  的请求帧回写方式，否则会在回放中提前出现未来结果。`frame` 现在包含模型和
  队列事件，不等于导航周期；简表的 C 是决策回调序号。旧录制没有新增的来源信息。
- `world/vlm/captures` / `headings` 是最新返回请求的拍摄机器人位置与相机 yaw；
  `frontiers` / `frontier_outlines` 是该请求的候选快照，不做当前有效性声明。
  `capture_link` 是位置对照线，不得作为底盘规划路径解释。图层按最近返回请求
  更新，地图 frame 不匹配时递归清空；V/F 表格实例顺序须与点序列一致。
- 使用现有环境的 `rerun.dataframe.load_recording(path)`，按 `frame` 时间线查询。
  `navigation/motion:Text` 包含最近决策和实时位姿（旧记录为 `world/hud:Text`）；
  `world/robot:LineStrip2D` 可恢复更精确位姿。
  world 图层的 Y 已取反，分析时须转回世界坐标。
- `rerun_view.py::_send_default_blueprint` 将 `navigation/motion` 放到侧栏 `Live`，
  `navigation/frontiers` 放到默认的 `Frontiers` 标签页。World 不再记录 HUD 点，
  `world/current_frontiers` 保留 ID 作为数据但显式 `show_labels=False`。
  表格编号链接的实例序号必须与该 Points2D 的顺序一致，候选与路径距离是最近一次
  Frontier 刷新的快照；运动中仅更新实时摘要，不重算候选。
- 优先读 `navigation/status_text:Text`；它保留局部待查数、复用数、完整目标和
  执行异常。旧记录可能只有中文 `navigation/status` 的 `ImageBuffer` 与
  `ImageFormat`；状态图层帧号对应周期回调，运动帧不重复写状态面板。
- Arrow 表转 Python 对象前省略或转成整数的 `log_time`，避免没有 pandas 时
  纳秒时间戳转换失败。读取既有日志无需启动 Habitat 或请求 VLM。
