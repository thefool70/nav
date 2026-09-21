# Habitat 仿真排错

## 启动顺序

`robot_nav habitat` 先创建 JSONL 运行日志，再在 `src/robot_nav/launch.py::_build_visualization`
构造 Rerun，再创建 `SemanticPerception` 和 `HabitatChassisAdapter`。因此，仅看到启动脚本打印的
“Habitat 渲染后端”时，还不能判断 Habitat 或 GPU 卡住。

从 Habitat 切换到 Hermes 时，先切换到 `robot-nav` 环境：

- `robot-nav-habitat` 使用 Python 3.9，`robot-nav` 使用 Python 3.11。
  当前 `hardware/realsense/.rsusb/python/pyrealsense2.cpython-311-*.so` 是
  Python 3.11 扩展；3.9 无法导入，即使 USB 已接入也会报模块不存在。
- `hardware/realsense/run.sh` 只注入 RSUSB 库路径，不切换 Python 环境。
  遇到此错误先核对 traceback 的解释器路径和扩展文件名，再选择匹配的环境；
  不要直接在 Habitat 环境安装另一套 RealSense 库。

## 仿真与真机差异的回归定位

- `habitat` 与 `hermes` 两个 CLI 入口都在 `launch.py` 创建 `SemanticPerception`
  （`_build_perception`），两种模式共用同一套感知与搜索核心。探索队列不接受
  同步观察器；物体接近通过 `--object-*` 接入独立模型进程；运动帧预采样由
  `set_motion_prefetch_enabled` 在动作期间开关。
- 两种入口共用 `launch.py::_run_navigation`；`hermes` 由 `environment.py::create_chassis` 创建 `HermesAdapter`（`adapters/hermes/`）。
  地图缓存、REST、Action 监控与路径检查全部在开发机执行；本地直连与经随车笔记本
  转发共用同一实现；底盘改 `--base-url`，远程相机另设 `--camera-source remote` 与 `--camera-endpoint`。
- 检测链先检查 `launch.py::_build_perception` 与 `_build_analyzer`，不能仅凭
  共用 `core/navigator.py` 判断算法一致。两种入口不再接入持续检测与模型运动中断；
  Hermes 仍保留首次扫描前的启动动作，见 `environment.py`；距离由
  `hermes.startup_forward_m` 指定，默认 1 m。
- 地图输入天然不等价：Hermes 按理论水平 FOV 筛选并膨胀厂商地图，Habitat 按局部
  视场和视线公开 navmesh 可见区域。排查 Frontier 数量差异时先比较两边实际
  `NavigationFrame.obstacle_map`。
- object/scene 两种状态与入口共用；场景模式本身不需要 YOLO+SAM2，不能把模式差异
  误判为环境差异。
- 已知区路径检查（`send_relative_pose_in_known_space`）两种 Adapter 均实现；
  Habitat 在离散动作前检查 navmesh 剩余路径，超限也会触发区域屏蔽。
  上限来自 `navigation.max_unknown_path_m`；不支持该接口的 Adapter 会报错停止。

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

- Hermes `adapters/hermes/observed_map.py::HermesObservedMap` 分别缓存世界格网上的
  `_raw_occupancy` 与 `_inflated_occupancy`。只从当前理论水平 FOV（最远 5m）
  读取底盘占用值，膨胀结果也只写回该区域；不能恢复成累计 seen 掩码后每帧重读
  所有已见格，否则视场外地图会变化。FOV 不读取深度、不检查遮挡。
  启动点 0.50m 区域只初始化一次，扩图不补读该圆内的视场外新格。原点平移时
  按首次格网锚点投回当前数组；frame_id、分辨率或 yaw 变化才清空并重新初始化。
  该规则控制算法使用的 FOV 缓存，不改变 Hermes 内部 SLAM 与避障地图。
  `adapters/hermes/adapter.py::_read_frame_locked` 还将同次读取的完整未膨胀图保留为
  `navigation_map`，供物体障碍定位和停靠；`navigation_clearance_m=0.36` 只在
  停靠规划时排除障碍邻域。完整图不得回写到 `HermesObservedMap` 的视场外缓存。
  `HermesObservedMap.update` 返回 `(obstacle_map, visibility_map)`；后者在膨胀前冻结，
  与探索图使用相同 FOV 缓存。`observation_coverage.py` 的方向筛选、固定网格覆盖及
  首层未知边界均使用视觉图，Frontier 格子仍从探索图转换为世界点。
  JSONL 的 camera 字段记录 `visibility_map_source`、`origin_in_obstacle_map` 和
  `origin_in_visibility_map`。底盘格为自由不代表前置相机光心也在自由格；若本地待查
  数为 0，先检查光心是否落在导航膨胀带，勿直接当作旧视角复用。
- 地图上的绿色轮廓来自 `rerun_view.py::_update_frontier_markers` 遍历每个
  候选的全部 `frontier_cells`，绿色格数不等于导航目标数。`Frontiers` 表格每行
  才是一个候选；JSONL 的 `frontier_candidates` 长度与各项
  `frontier_cell_count` 可分别核对候选数和边界格数，截图须关联周期后才能比较。
- `frontier.py::extract_frontiers` 返回候选及过滤统计，按八邻接保留完整连续边界，不按长度
  或跨度拆分。`_merge_frontier_fragments` 仅按 0.30 m 自由区短路径和未知侧
  朝向判断断段是否合并，不限制总跨度；合并后统一过滤跨度小于 0.50 m 的区域。
  每个有效区域只产生一个移动代表点，`frontier_cells` 保留完整边界供扫描与匹配。
  排查连续边界被分成多个候选时，先核对边界是否实际八邻接连通和候选刷新周期。
  聚类前 `_small_unknown_holes` 仅从候选邻接的未知格开始，在同格网 `visibility_map`
  上搜索面积不超过 `MAX_UNKNOWN_HOLE_AREA_M2=0.05` 的封闭八连通块；触图边、
  超面积或连接到已确认保留的区域就停止。不得改用膨胀图判断连通性。
  `_find_frontier_cells` 与 `_unknown_side_normal` 使用同一排除集，避免合并时重新
  引入孔洞方向。排除集仅在本次提取有效；JSONL `state.frontier_hole_filter` 的
  applied=false 表示未执行过滤，不能与执行后未发现孔洞混淆。
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
  不记录选中 `candidate_id`。`Motion details` 和终端显示本次返回节点、`branch_depth` 与
  `pending_direction_count`，数量为发出命令时的快照，到达后须刷新。
  JSONL 状态保存 `backtrack_node_id` 和完整 `branch_node_ids`，可核对返回顺序。
  `app.py::_execute_action` 等待同步动作结束后才允许下一周期决策；
  排查途中改目标时先核对 Adapter 完成反馈。
- Hermes 的 `KnownSpaceChassisInterface` 扩展在 `explore.select`、
  `backtrack.return`、`backtrack.resume`、`target.revisit`、`object.fallback_return`
  和 `object.approach` 使用。`app.py::_execute_action` 对物体停靠优先传入
  `frame.navigation_map`，其余动作传入 `frame.obstacle_map`，同时显式传入
  `reference_pose=frame.pose`。不能用 FOV 图检查完整导航图选出的物体停靠路径。
  启动前移、标定与扫描转向仍走普通接口；
  Habitat 当前未实现该扩展。
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
- `HermesConfig.max_unknown_path_m` 与 CLI `--max-unknown-path-m` 默认均为
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
  镜头中心可在多个边界点之间。首次仍环扫；之后无局部待查点时两种模式均采集
  当前朝向。日志 `scan_mode` 分别为 `initial`、`frontier`、`current_view`，
  Rerun `scan basis` 显示对应原因；旧录制的 `scene_current_view` 仅对应当时的场景采集。
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
  和 `pending_observation_view_count` 判断，只有成功分析的覆盖才进入已观察集合。
- `--seed` 只传给 Habitat；随机观察器使用独立、未设种子的 `random.Random()`。
  随机评分运行可检查状态流，不能据单轮轨迹归因 VLM 效果或衡量算法改进。
- 排查“没有优先向前”时检查 `core/exploration.py::select_exploration_target`：当前排序
  先分新候选与暂存候选，只有新候选进入 `FrontierScoreRequest`；没有新候选时
  才逐个回到父节点，遇到该节点的有效暂存方向按原顺序恢复，不再使用区域切换扣分。
  扫描结束时的车头 yaw 不参与分层。`frontier_selection_source` 为 `new` 或
  `deferred`；提交 Frontier 移动时，新旧数量是本次选择前的数量；
  JSONL 状态的 `deferred_frontiers` 是选择后的暂存队列，Rerun 同样显示来源与队列数量。
- 同步场景判断 `uncertain` 时保留本轮采集；异步场景返回只核对位置和朝向，
  队列物体模式以停靠命令成功执行为完成依据，不再请求到达感知。
  两种模式无局部待查 Frontier 时均用当前 yaw 采集一张图入队，不再让物体跳过
  当前画面。异步选点仍查严格匹配的评分缓存；未拍到的方位不投影到图片边缘。
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
  `_scan_images` 或共享可变的当前任务 ID。场景返回不另发视觉请求；物体定位请求通过 job/view 编号关联历史画面。
- `SemanticPerception._score_origins` 随评分缓存保留 J/R 来源；
  `semantic_received_jobs` 是该周期接收结果，`semantic_score_sources` 是传给
  排序的有效缓存输入。两者不是同一件事，返回有效分数不等于已用于导航。
- `perception/semantic_queue.py::SemanticPerception` 是 CLI 的默认感知实现。
  扫描线程冻结画面；运动回调只提交最新原始帧，由独立采样线程处理；单个分析
  线程按 FIFO 调用 `OpenAICompatibleTargetObserver.analyze_views`，不使用
  `_scan_images`。设备只由 Adapter 读取，核心状态只由核心函数更新。
- `data/run_logs/semantic-*/job-*` 保存 `view-N.rgb.gz`、`snapshot.json` 和
  `result.json`。快照中的 `snapshot:批次:序号` 与 `source_region_id` 分别是
  请求内候选 ID 和原区域 ID；分数缓存必须同时匹配地图 frame、原区域 ID、
  世界目标坐标，不能只按新分配的预览 ID 套用。预览不提交核心区域编号。
- `CapturedView.depth_gzip` 在 `_capture` 时复制并压缩同帧对齐深度；不保留
  原始帧引用。`snapshot_store.write_snapshot` 先写 `view-N.depth.pending.f64.gz`，VLM 返回后
  `snapshot_store.retain_clue_depth` 仅把命中 V 编号重命名为 `view-N.depth.f64.gz`，删除其余
  临时深度。检测列表为 `[]` 时全删临时深度，为 `None` 时保留待判定数据。
  队列退出或分析未完成时不清理这些临时文件，也不自动恢复任务。
- 深度由 `adapters/snapshot_depth.py` 编解码；gzip 内为 `<8sII` 头
  （`RNDEPTH1`、宽、高）及按行排列的小端 float64 米制数据，None 编为零，
  非正或非有限值解码为 None。`snapshot_store.read_clue_frame` 仍能读取旧 `.depth.json.gz`。
  `snapshot.json` 的 `depth.encoding=gzip-float64-le-v1` 标明新格式；尺寸须与 RGB 相同。
  每张图的 `depth.captured` 只表示采集时是否有深度，不表示模型命中或测距可靠。
  实际保留情况见 `result.json.depth_retention.retained_view_ids`，命中但无文件见
  `missing_view_ids`；`pending_detection` 表示检测失败，`selection_failed` 的
  `errors` 与 `depth_retention_failed` 事件记录文件处理失败。文件失败不改写目标
  判断，深度也不发送给 VLM。旧快照不会补出历史深度。
- 扫描中出现连续单图批次时，先检查 `snapshot.json.source`。`capture_scan_view`
  正常只在 `context.index + 1 >= context.count` 时提交整轮，不按新 Frontier
  提前拆批。被打断或重建计划时，`app.py::_sync_perception` / 下一轮开始处仍提交未送出的
  部分画面；后续局部扫描本身可能只有一图，不能只凭图数判断是否重复请求。
- `app.py::_execute_action` 用 `set_motion_prefetch_enabled` 开关运动采样，
  `_prefetch_allowed` 按 `ActionKind` 与 `ActionPurpose` 选择探索及回退移动；
  scan.turn、目标接近和目标返回不采样，finally 中关闭开关。
  动作期间已接收的帧保留冻结的采集上下文，允许采样线程稍后完成处理。
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
  `snapshot_store.read_snapshot` 还原为元组。锚点按同一世界坐标匹配，不依赖队列重命名的
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
- `build_frontier_score_sheet` 保留数字方向标注的旧拼图形式；默认 FIFO 联合
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
  列表只包含检测匹配的画面；场景和队列物体模式到位后均不进行二次视觉确认。
  Rerun 的 `Clue order` 是模型返回顺序，非新的世界候选编号。
- `SemanticPerception.begin_cycle` 接收结果：检测有效才登记覆盖，评分失败仍可保留目标线索。
  `_pending_coverage` 在结果被主循环接收前一直保留；失败后移除，不冒充已检查。
  `pending_semantic_jobs` 还含采样、待接收结果、部分扫描和排队线索，不是 HTTP 数。
- `TargetClue.pose` 是拍摄时机器人位姿，`job_id/view_id` 关联同帧 RGB-D。
  场景由 `core/scene_target.py::continue_scene_target` 返回并恢复朝向；位置误差 ≤0.25m、
  朝向误差 ≤5° 时返回 COMPLETE / target.revisit_complete。
  物体由 `core/object_approach.py` 处理 LOCALIZING_OBJECT → APPROACHING_OBJECT
  → COMPLETE。`_complete_approach` 以停靠命令成功执行为依据，记录
  `completion_basis=standoff_command_completed`，不记录到达后重新测量的目标距离。
  `destination` 在停靠发出时写入，运动失败由 `recover_object_motion` 清空；下一周期
  接近状态仍有目的地即表示命令已成功执行，直接完成，不再调用模型。
  单条历史失败保留 `ObjectApproachState.fallback_clue/history_localized`，继续
  其他历史线索。`continue_object_history` 在 WAITING_FOR_SEMANTICS 原地排空
  已采集队列；全部不能定位才经 REVISITING_TARGET 保底返回，最终 STOPPED。
  STOPPED 以 CLI 退出码 3 结束，不再重采或调用确认。
  `app.py::_supply_perception` 每周期最多消费一次定位结果，队列
  `localize_object` 始终调用 `snapshot_store.read_clue_frame`，没有到达后当前帧定位分支。
  历史帧的位姿与标定用于定位，停靠始终使用当前帧的位姿和地图。
  `snapshot_store.read_clue_frame` 保留当前帧地图；历史深度文件缺失时返回 `depth=None`，
  已有深度尺寸或单位错误仍使该快照失败。障碍射线由历史相机位姿发出，查询当前
  同坐标系地图；`frame.json` 的 `timestamp_s` 是拍摄时间，`map_timestamp_s`
  是当前导航帧采集时间，不能把地图理解成拍摄时保存的地图。
  `SemanticPerception.begin_cycle` 按列表原顺序入队，物体按 job/view 保留每张不同快照；
  场景仍按位姿去重且不重新排序，批次按 FIFO。`core/scene_target.py::discard_target_clue` 返回无命令状态，
  场景返回失败以 target.revisit_failed、物体线索失败以 object.clue_failed
  让下一周期优先取下一条，COMPLETE 与 STOPPED 不再取线索。
  `navigate` 的 active_target_clue 分支优先于普通视觉处理；`app.py` 跳过线索
  处理期间的普通目标观测，物体定位只响应 NEEDS_OBJECT_LOCALIZATION。
- `perception/object_localizer.py` 的两个检测线程独立运行，首个有效框进入 SAM2；
  不要求 YOLO/VLM 同时成功，也不做框重叠否决。SAM2 失败或掩码无法测距时用框。
  `object_model_process.py` 为 YOLO、SAM2 分别维护常驻进程和单请求锁；旧 YOLO
  尚未返回时不积压新帧，VLM 路径仍可继续。每次本地请求有超时，退出时关闭进程。
  同一 `rgb_file` 的多个框复用 SAM2 图像编码；输入路径每次定位独立，不覆盖旧图。
  子进程移除 LD_PRELOAD/LD_LIBRARY_PATH/PYTHONHOME，默认通过
  `cli.py::_default_object_python` 使用已有 robot-nav 解释器。
- `object-localization/observation-*/frame.json` 记录尺寸、标定与关联编号；
  `input.rgb.gz`、`input.depth.json.gz` 保留同帧输入，缺深度时不写深度文件并以
  `depth_available=false` 标记。`input.obstacle_map.json.gz` 保存当次定位查询地图；
  `localization_map` 区分 full_navigation/exploration。`localization.json` 保存结果、
  bbox、detector_source、localization_method、拍摄机器人坐标系下的距离与角度。
  `yolo.request.json/result.json/progress.jsonl` 位于该目录；SAM2 对应记录及
  `mask.json.gz` 位于候选来源子目录 `yolo/` 或 `vlm/`。模型完整日志位于
  `object-localization/models/yolo.log` 与 `sam2.log`；20 秒未完成时输出调用栈。
  结合 importing/loading_yolo/encoding_class_text/encoding_image/segmenting 阶段
  定位耗时；进程报错不等于未检出，另一检测器或框定位仍可成功。
  `np.frombuffer(bytes)` 返回只读数组；worker 复制成可写 RGB 再交给模型，避免
  SAM2 的 ToTensor 在 `torchvision/transforms/functional.py` 发出只读输入警告。
- `core/object_grounding.py` 直接调用定位计算：只要求存在可计算的正深度，
  不设 8 点、有效率、中位深度集中度、连通块或占用格支持门槛。
  两路检测都不能提供 RGB-D 位置时，`localize_obstacle_on_image_ray` 有框沿框
  中心、无框沿相机光轴，用格边遍历返回首个占用格的进入点。未知格不提供距离；
  不截断到 5m，也不设置假定距离。使用完整未膨胀导航图，缺图时沿用探索图。
  `source=bbox_obstacle/front_obstacle`、`localization_method=obstacle_assumption`
  和 `sample_count=0` 表示位置假设；无框的 visibility 仍为 uncertain，VLM 否定
  仍保留 rejected。核心依据是否有有效坐标决定接近，不把这些值改成模型确认。
- `core/object_standoff.py::plan_object_standoff` 一次计算四邻接可达区，枚举目标
  周围 0.60–2.0m 全圆域自由格；优先最接近机器人侧 0.75m 理想位置，再按路径距离
  排序。Hermes 用完整图加 0.36m 净空，Habitat 沿用已有图，不再额外膨胀。
  `standoff_map`、`standoff_clearance_m`、`standoff_search_radius_m` 与
  `standoff_candidate_count` 记录选点依据；unknown/occupied/unreachable/excluded
  格数均带 `standoff_` 前缀与 `_cells` 后缀，unreachable 包括净空膨胀排除。
  排查“无停靠点”先看这些字段。
- `app.py::_sync_perception` 发布冻结的 `CaptureContext` 并同步暂停标志。
  有待查线索、active_target_clue、目标返回/接近/定位阶段或导航终态时暂停。
  `_worker_loop` 与 `_capture_loop` 等待暂停解除，`observe_motion_frame` 不接收新帧。
  `SemanticPerception.begin_cycle` 在暂停期间保留 `_completed` 与 `_pending_coverage`；
  接收后由 `core/perception_flow.py::receive_perception` 更新搜索状态。
  两条已有线索之间保持暂停；物体线索用完时原地恢复已采集队列，继续接收线索。
  `fallback_clue` 本身不使队列暂停，否则会在等待历史结果时自锁。
  已开始的请求/采样可以完成并保存结果；暂停不是取消已发送的 HTTP 请求。
  `semantic_background_paused` 进入周期诊断，Rerun 简表显示普通队列暂停提示。
- 场景判断由联合分析完成；模型协议仅包含 `analyze_views` 与 `locate_object`，
  没有同步观察器、到达后模型确认或检测触发的运动中断分支。
- 分数只在下一次决策读取，不修改执行中的目标；默认 CLI 不使用本地 YOLO 中断。
  `_wait_for_semantics_or_finish` 必须等待任务与线索耗尽；失败检测计数单独报告。
  `close` 停止继续出队，已落盘数据保留，当前没有恢复队列或磁盘限额机制。

## 导航退出路径排查

- OpenCode Go 的 `HTTP 400 / MissingSessionID` 表示请求已到服务端，但缺少
  `x-opencode-session`；不能归因为 DNS 失败或直接认定凭据错误。要求见
  [官方接入说明](https://opencode.ai/docs/go/#where-can-i-use-it)。
  `launch.py::_build_analyzer` 为每次导航生成 UUID，填入
  `OpenAICompatibleConfig.opencode_session_id`，`_post_json` 在各类请求中复用。
  不要每次 HTTP 请求重新生成；User-Agent 保持 robot-nav 的真实标识。
  查看 `job-*/result.json` 的 detection_error/scoring_error 或终端请求错误；
  仅有导航周期继续输出不代表模型成功，普通模型失败会继续几何探索。
- Hermes 静止检测默认 1 秒，直连及无线 CLI 的 `--action-stall-timeout-s` 与
  `HermesConfig.action_stall_timeout_s` 必须同步。`_monitor_action` 每次
  Action 轮询读取位姿并检查，默认间隔 0.2s；2s 仅控制日志输出，不能用于节流到位判断。
  实际间隔另受帧采集与 REST 延迟影响，不能理解为精确第 1 秒取消。
  MoveToAction 的有效进展仍是平移至少 2 cm，原地转向不重置其平移停滞计时。
- `adapters/hermes/adapter.py::_check_stable_arrival` 对 working Action 检查目标
  位置误差 ≤`action_arrival_position_m`（0.30m）或转向误差 ≤`yaw_tolerance_rad`（5°）。
  在容差内连续 `action_arrival_hold_s`（0.001s，实际至少等到后续轮询）相对采样锚点移动 <2cm、转动 <1° 才
  抛出内部到达信号；MoveTo 还要求先有有效平移。`_execute_action` 捕获后调用
  `_cancel_active_action`，用 `require_success=False` 确认终态，再读位姿复查。
  取消或终态确认失败仍报错；复查超出容差作为可恢复运动失败。该容差不同于
  `position_tolerance_m`（0.03m），后者仅决定是否下发微小平移命令。
- `app.py::run_navigation` 在目标完成、`FAILED`、非 `OK` 且非等待感知状态，
  或达到 `max_cycles` 时退出；`MISSING_DATA` 当前也立即退出。周期上限不能当作
  Frontier 耗尽；`OK + WAITING_FOR_SEMANTICS` 每次最多等待结果 1 秒，不消耗决策
  额度。普通模型失败记为失败批次、评分降级。
- 物体线索处理由状态机持续推进，NEEDS_OBJECT_LOCALIZATION 属于等待感知状态。
  `core/object_approach.py::recover_object_motion` 对 object.approach 清空目的地，
  下一周期用最新地图换停靠点；object.fallback_return/object.fallback_turn 失败则 STOPPED。
  接近阶段不沿用 Frontier 整区域屏蔽，三次停靠仍失败才放弃线索。
- `core/navigator.py::recover_from_motion_failure` / `continue_after_motion_stall`
  对 `scan.turn` 使用 `_recover_scan_turn`，清除扫描计划后按真实朝向重新规划，
  不登记失败方向的覆盖。首次扫描尚未完成时仍按首次环扫规则重建。
- `environment.py::_move_hermes_forward_on_start` 在 `run_navigation` 之前直接调用
  底盘；前移 1 m 的可恢复失败/停滞在确认动作结束后继续启动，其他异常仍停止。
- `adapters/hermes/adapter.py::_execute_action` 将 Action 创建、起始位姿读取和监控
  放在同一异常范围；`_cancel_active_action` 与 `close` 负责取消遗留动作。已获得
  ID 时还要确认终态；创建请求失败而没有 ID 时只能尝试取消，原异常仍终止运行。
- `runtime_reporting._optional_callback` 集中处理可视化的 I/O 与运行库故障；
  类型或字段错误继续传播。Adapter 不再重复捕获回调异常；JSONL `_write` 只隔离
  文件 I/O 错误，序列化错误直接暴露。直接传入 Adapter 或周期函数的自定义回调
  由调用方负责。采集、路径检查、健康错误仍停止并收尾动作。
- 正式扫描的快照落盘失败仍传播并停止；运动预采样的文件 I/O 失败记录 `prefetch_skipped`，
  不登记已检查。`snapshot_failed` 与普通日志丢失不同；结果落盘失败时内存结果
  仍交给主循环。队列停止后运行中的 HTTP 可能完成，退出不自动续跑剩余任务。

- `SemanticPerception._run_worker` 只负责把后台异常传到 `begin_cycle` / `wait_for_result`，
  不转成普通检测失败。排查队列停止先看原始异常堆栈；不要通过增加宽泛捕获恢复等待。
- `core/navigation_io.py` 只检查进入决策的物理数据，不遍历检查 `SearchState` 的字段类型。
  状态错误应回到创建或更新该状态的行为模块修复。

## 离线读取 RRD

- `visualization/vlm_trace.py` 只保存轻量摘要，`rerun_view.py` 写完整卡片和 World
  叠加。`model/vlm/summary` 是简表，`model/interaction` 是最新完整事件；
  `model/vlm/requests/R000001` 保留单次会话，`/text` 保留不依赖字体的完整文本，
  `model/vlm/jobs/J000001` 保留任务事件与接收／排序周期。
- VLM 请求、返回及队列事件均调用 `_begin_sample`，不能恢复旧 `_vlm_samples`
  的请求帧回写方式，否则会在回放中提前出现未来结果。`frame` 现在包含模型和
  队列事件，不等于导航周期；简表的 C 是决策回调序号。旧录制没有新增的来源信息。
- `visualization/semantic_world.py` 保存每个固定节点：
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
  不等待 `app.py` 的最终周期回调。`_log_motion_status` 在推理期间优先显示进度，
  不将旧 object.approach 命令显示为仍在执行。物体事件不覆盖普通 J 任务的状态。
- 使用现有环境的 `rerun.dataframe.load_recording(path)`，按 `frame` 时间线查询。
  `navigation/motion:Text` 包含最近决策和实时位姿（旧记录为 `world/hud:Text`）；
  `world/robot:LineStrip2D` 可恢复更精确位姿。
  world 图层的 Y 已取反，分析时须转回世界坐标。
- `rerun_view.py::_send_default_blueprint` 将 `navigation/live` 放到主图下方 `Live`，
  `navigation/motion` 放到 `Motion details`；侧栏默认是推理单图评分卡与 Observations
  索引，`navigation/frontiers` 在 `Frontiers` 标签页。World 不再记录 HUD 点，
  `world/current_frontiers` 保留 ID 作为数据但显式 `show_labels=False`。
  表格编号链接的实例序号必须与该 Points2D 的顺序一致，候选与路径距离是最近一次
  Frontier 刷新的快照；运动中仅更新实时摘要，不重算候选。
- 优先读 `navigation/status_text:Text`；它保留局部待查数、复用数、完整目标和
  执行异常。旧记录可能只有中文 `navigation/status` 的 `ImageBuffer` 与
  `ImageFormat`；状态图层帧号对应周期回调，运动帧不重复写状态面板。
- Arrow 表转 Python 对象前省略或转成整数的 `log_time`，避免没有 pandas 时
  纳秒时间戳转换失败。读取既有日志无需启动 Habitat 或请求 VLM。
