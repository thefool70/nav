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
- 两种入口共用 `launch.py::_assemble_and_run`；`hermes` 由 `environment.py::create_chassis` 创建 `HermesAdapter`（`adapters/hermes/`）。
  地图缓存、REST、Action 监控与路径检查全部在开发机执行；本地直连与经随车笔记本
  转发共用同一实现；底盘改 `--base-url`，远程相机另设 `--camera-source remote` 与 `--camera-endpoint`。
- 检测链先检查 `launch.py::_build_perception` 与 `_build_analyzer`，不能仅凭
  共用 `core/navigator.py` 判断算法一致。物体模式由 Adapter 持续交付最新帧，由 YOLOE 限频检测；命中不触发运动中断；
  Hermes 仍保留首次扫描前的启动动作，见 `environment.py`；距离由
  `hermes.startup_forward_m` 指定，默认 1 m。
- 地图输入天然不等价：Hermes 按理论水平 FOV 筛选并膨胀厂商地图，Habitat 按局部
  视场和视线公开 navmesh 可见区域。排查 Frontier 数量差异时先比较两边实际
  `NavigationFrame.obstacle_map`。
- object/scene 两种状态与入口共用；场景模式本身不需要 YOLOE，不能把模式差异
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
- `visualization/rerun_view.py::RerunVisualizer.__init__` 用 `rr.new_recording` 建立磁盘流并
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

## Frontier 与观察记录排错

- Hermes `adapters/hermes/observed_map.py::HermesObservedMap` 分别缓存世界格网上的
  `_raw_occupancy` 与 `_inflated_occupancy`。只从当前理论水平 FOV（最远 5m）
  读取底盘占用值，膨胀结果也只写回该区域；不能恢复成累计 seen 掩码后每帧重读
  所有已见格，否则视场外地图会变化。FOV 不读取深度、不检查遮挡。
  启动点 0.50m 区域只初始化一次，扩图不补读该圆内的视场外新格。原点平移时
  按首次格网锚点投回当前数组；frame_id、分辨率或 yaw 变化才清空并重新初始化。
  该规则控制算法使用的 FOV 缓存，不改变 Hermes 内部 SLAM 与避障地图。
  `adapters/hermes/adapter.py::_capture_frame` 还将同次读取的完整未膨胀图保留为
  `navigation_map`，供物体障碍定位和停靠；`navigation_clearance_m=0.36` 只在
  停靠规划时排除障碍邻域。完整图不得回写到 `HermesObservedMap` 的视场外缓存。
  `HermesObservedMap.update` 返回 `(obstacle_map, visibility_map)`；后者在膨胀前冻结，
  与探索图使用相同 FOV 缓存。`scan.py` 的方向筛选、固定网格覆盖及
  首层未知边界均使用视觉图，Frontier 格子仍从探索图转换为世界点。
  JSONL 的 camera 字段记录 `visibility_map_source`、`origin_in_obstacle_map` 和
  `origin_in_visibility_map`。底盘格为自由不代表前置相机光心也在自由格；若本地待查
  数为 0，先检查光心是否落在导航膨胀带，勿直接当作旧视角复用。
- 地图上的绿色轮廓来自 `rerun_view.py::_update_frontier_markers` 遍历每个
  候选的全部 `frontier_cells`，绿色格数不等于导航目标数。`Frontiers` 表格每行
  才是一个候选；JSONL 的 `frontier_candidates` 长度与各项
  `frontier_cell_count` 可分别核对候选数和边界格数，截图须关联周期后才能比较。
- `exploration.py::extract_frontiers` 返回完整边界、移动候选及过滤统计，按八邻接保留完整连续边界，不按长度
  或跨度拆分。`_merge_frontier_fragments` 仅按 0.30 m 自由区短路径和未知侧
  朝向判断断段是否合并，不限制总跨度；合并后统一过滤跨度小于 0.50 m 的区域。
  每个有效区域只产生一个移动代表点，候选的 `frontier_cells` 保留完整边界供区域匹配。
  `FrontierExtraction.boundary_cells` 在小孔洞过滤后、移动候选筛选前保存，供扫描使用。
  排查连续边界被分成多个候选时，先核对边界是否实际八邻接连通和候选刷新周期。
  聚类前 `_small_unknown_holes` 仅从候选邻接的未知格开始，在同格网 `visibility_map`
  上搜索面积不超过 `MAX_UNKNOWN_HOLE_AREA_M2=0.05` 的封闭八连通块；触图边、
  超面积或连接到已确认保留的区域就停止。不得改用膨胀图判断连通性。
  `_find_frontier_cells` 与 `_unknown_side_normal` 使用同一排除集，避免合并时重新
  引入孔洞方向。排除集仅在本次提取有效；JSONL `state.frontier_hole_filter` 的
  applied=false 表示未执行过滤，不能与执行后未发现孔洞混淆。
- `core/exploration.py::refresh_frontier_regions` 返回更新状态与 `FrontierExtraction`；
  `core/exploration.py::match_frontier_regions` 用世界坐标边界关联 ID。历史节点只保留
  实际尝试的 Frontier 移动，节点的出发位置也是本轮未选方向的父节点。
  JSONL `state.latest_node` 直接记录 `candidate_id`、`destination_world_xy`、
  `state` 与 `execution_reason`，不再嵌套 `directions`；`COMMITTED` 尚不参与
  目标点排除，其余三种状态参与排除。未选方向只由区域暂存顺序表达。
- `FrontierRegion.deferred_order` 保存暂存时的（观测节点序号，候选排名），与
  `FrontierCandidate` 同步；序号索引 `observation_history`，同节点排名小的先恢复。
  新候选耗尽后，`begin_backtracking` 只取 `branch_node_ids[-1]` 对应节点的
  `position_world_xy`，用 `backtrack_node_id` 固定返回目标；不能按全局暂存队列
  找更早节点并直接跳过去。`backtrack.return` 不新增历史、不消耗暂存队列。
  `exploration.py::defer_unselected_frontiers` 仅在真正选定命令后暂存本轮其他新方向，
  等待 VLM 评分时不得提前暂存。选中区域解除暂存，沿该方向继续产生的边界可优先探索。
- `SearchState.branch_node_ids` 与完整 `observation_history` 分开：提交 Frontier
  命令时压入出发节点，正常返回到达且没有有效暂存方向后出栈；返回失败的节点
  通过 `recover_backtrack_issue` 单独跳过，并释放它的暂存方向。新分支沿
  剩余栈继续压入；不能简单倒序遍历全部历史，否则会重新走进已经退完的旧分支。
  恢复方向时新的移动记录可能与原节点同位置；容差内的节点直接检查，无额外返回动作。
- `continue_backtracking` 只在同步返回动作结束后的新周期运行；与父节点距离
  不超过 `BACKTRACK_ARRIVAL_M = 0.25` 才刷新边界并检查本节点方向，否则
  `recover_backtrack_issue(issue_kind="not_arrived")` 返回 `OK + SCANNING`，
  不得直接置为 `FAILED`，也不重复下发同一失败节点。起点已在容差内则直接恢复。
  没有有效暂存方向则出栈；若到达位置显露新候选则重新选点，否则只返回上一层。
  即使全局候选为空，也先逐级退完当前分支，再等待视觉队列，才 `explore.exhausted`。返回开始时清空本轮
  `scan_views` 与扫描计划；`observed_views` 和感知侧按地图、区域、目标匹配的评分缓存仍保留。
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
  启动前移与扫描转向仍走普通接口；
  Habitat 当前未实现该扩展。
- Hermes `_send_relative_pose` 用 `reference_pose` 还原世界目标与目标 yaw，
  `start_pose` 仅计算剩余距离和反馈。排查“Action 完成但未到达父节点”时对照
  `cycle_decision.command.target_world_xy`、`Hermes command target` 和实际终点。
  已确认日志 `slamtec-l515-20260905-214649-157351.jsonl` 第 38 周期两次朝向相差
  约 8.7°，旧实现把目标偏移约 0.36 m，最终距原父节点 0.308 m，触发旧终止分支。
  当时仍有有效暂存候选 `region:4`（顺序 `1:1`）和 12 个回退节点，不能归因于
  `explore.exhausted`；已屏蔽的 3 个区域不计入有效待探索候选。
  这不能仅靠放宽 0.25 m 容差掩盖；平移完成的 `target_error` 仅相对底盘收到的目标。
- `core/geometry.py::measure_unknown_path_length` 按格边交点切分每段路径，
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
- `core/navigator.py::apply_execution_result` 接收 `PATH_UNKNOWN` 反馈后将尝试标为
  `INVALIDATED`，并把当时完整世界边界存入
  `SearchState.blocked_frontier_regions`。取消结果记录 `rejection_scope=region`、
  `rejected_path_world_xy`；首个未知格仍保存在 Action 日志和历史 `execution_reason`。
  这适用于 `explore.select` / `backtrack.resume`；`backtrack.return` 的执行失败、
  停滞或未知路径长度超限均调用 `exploration.recover_backtrack_issue`，不能据返回失败屏蔽未尝试的区域。
  恢复从 `branch_node_ids` 弹出失败节点，将其所属区域的 `deferred_order` 清为
  `None`，保留边界和整片屏蔽记录，再由下一周期按实际帧重新检查。日志
  `motion.backtrack_recovered` 记录失败种类、跳过节点、释放区域及可用的位置误差。
  返回被目标检测中断时，`_release_interrupted_direction` 清除回退状态并转扫描，
  保留所有暂存方向供后续恢复。
- `frontier_regions.refresh_frontier_regions` 在区域 ID 关联之前调用
  `history.filter_blocked_frontier_regions`。被屏蔽边界独立于候选保存，按世界格心
  与自由区一格邻域匹配，并保留所有匹配过的完整边界。不能只屏蔽失败坐标附近
  `max(0.4 m, 2 格)`，也不能只靠旧 `region_id`，否则换代表点或分裂会反复下发。
- 区域屏蔽在本次运行中持续有效，不复查旧路径、不因地图更新自动解除。
  `BlockedFrontierRegion` 只保存区域 ID 与边界；被拒绝路径仅写入取消日志。
  新动作仍用自己的选点地图检查实时路径；普通失败、停滞、目标检测中断不额外
  屏蔽整片区域。Rerun `Frontiers` 与 JSONL `blocked_frontier_regions` 可查屏蔽记录。
- `SearchState.scan_observation_points` 来自 `FrontierExtraction.boundary_cells`，
  不受最小移动距离、区域跨度、历史目标排除或未知路径区域屏蔽影响。
  `refresh_frontier_regions` 只替换提取结果中的移动候选，保留扫描边界与帧缓存。
  观察点与朝向在整轮扫描期间冻结，不因途中边界消失
  而重排。`observed_views` 只在有效本地观测或成功后台目标判断后增加；相机采集、
  地图公开和 Rerun 运动帧均不代表 VLM 已检查。
- 扫描未指向 Frontier 代表点时，先检查 `scan.py` 的
  `frontier_observation_points`：只保留光心距离大于 0.10 m、至多 4 m 且地图
  视线可达的边界格。`scan.py::build_unobserved_scan_headings` 合并视场后，
  镜头中心可在多个边界点之间。启动与后续扫描共用此规则；无局部待查点时两种
  模式均采集当前朝向。日志 `scan_mode` 为 `frontier` 或 `current_view`，
  Rerun `scan basis` 显示对应原因；旧录制的 `scene_current_view` 仅对应当时的场景采集。
- 扫描规划日志的 `frontier_scan_cell_count` 是移动筛选前的边界格数，
  `frontier_move_candidate_count` 是筛选后的移动候选数，两者可能一个非零、一个为零。
  `WAITING_FOR_SEMANTICS` 也检查这些边界的未观察覆盖，无移动候选时仍可恢复扫描。
- `core/scan.py::_local_coverage_points` 的 0.25 m 固定世界
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
  覆盖点数；`scan_captured_count` 是本轮已采集数，也是下一个计划方向的下标。
  `scan_views` 直接复用提交队列时的覆盖，不重复采样，不代表已检查；要结合全局 `observed_views`
  和 `pending_observation_view_count` 判断，只有成功分析的覆盖才进入已观察集合。
- `--seed` 只传给 Habitat；随机观察器使用独立、未设种子的 `random.Random()`。
  随机评分运行可检查状态流，不能据单轮轨迹归因 VLM 效果或衡量算法改进。
- 排查“没有优先向前”时检查 `core/exploration.py::select_exploration_target`：当前排序
  先分新候选与暂存候选，`FrontierScoreRequest.candidates` 才查询分数，`deferred` 原序保存。
  运行层查询缓存后直接进入 `complete_frontier_selection`，不重新调用 `navigate` 或刷新区域。
  没有新候选时才逐个回到父节点，遇到该节点的有效暂存方向按原顺序恢复，不再使用区域切换扣分。
  扫描结束时的车头 yaw 不参与分层。`frontier_selection_source` 为 `new` 或
  `deferred`；提交 Frontier 移动时，新旧数量是本次选择前的数量；
  JSONL 状态的 `deferred_frontiers` 是选择后的暂存队列，Rerun 同样显示来源与队列数量。
- 场景返回只核对位置和朝向，
  队列物体模式以停靠命令成功执行为完成依据，不再请求到达感知。
  两种模式无局部待查 Frontier 时均用当前 yaw 采集一张图入队，不再让物体跳过
  当前画面。异步选点仍查严格匹配的评分缓存；未拍到的方位不投影到图片边缘。
- `INVALIDATED` 表示执行失败，`STALLED` 表示停滞；二者均不证明 navmesh
  不连通。有紫色路径但无运动帧时，继续检查 Habitat `_follow_path()` 的动作生成。
  周期回调发生在执行前，异常原因通过历史方向的 `execution_reason` 保留，
  下一周期 Rerun 的 `last exploration issue` 显示最近一次异常及其节点编号。

## 异步视觉队列

- 连接故障先读 JSONL 中 `semantic_queue` 事件的 `detection_error`，区分 HTTP
  状态码与收到响应前断开。已确认 2026-09-07 的 `semantic-cri8lcr5` 共完成
  34 批：7 批成功、27 批连接失败（25 次 RemoteDisconnected、1 次 Broken pipe、
  1 次 SSL EOF）；job-000004～000010 成功，job-000011 起连续失败。
  此类日志不能直接证明密钥失效、额度耗尽或 Qwen 全局宕机。
  `_post_json` 使用默认 urllib opener，会继承启动进程的 HTTP(S) 代理设置；
  排查 OpenCode Go 链路时核对该进程的代理环境，不用当前工具进程环境替代历史证据。
  不输出凭据值，也不要把连接失败当作“未检测到目标”。
- `VlmInteraction.context` 与 `SemanticAnalyzer.analyze_view(trace_context=...)`
  只增加记录来源，不参与提示词或 HTTP payload；不能为了显示关联关系串用
  `_scan_images` 或共享可变的当前任务 ID。场景返回不另发视觉请求；物体定位通过 job/view 编号关联历史画面并复用扫描框。
- `SemanticPerception._scores` 每项保存 `(score, job_id, interaction_id)`，保留 J/R 来源；
  `semantic_received_jobs` 是该周期接收结果，`semantic_score_sources` 是传给
  排序的有效缓存输入。两者不是同一件事，返回有效分数不等于已用于导航。
- `perception/semantic_queue.py::SemanticPerception` 持有 RGB-D 内存快照，准备线程只做扫描
  RGB 打包与深度复制；VLM 线程对扫描图请求同帧 YOLOE，再按 FIFO 调用 `analyze_view`。
  视频高分结果绕过准备线程和 VLM 队列。`begin_cycle` 只返回覆盖，线索另行领取。
  分数缓存匹配地图系、原区域 ID 与世界目标，保留首次有效分数。
- `_snapshots` 最多 256 帧、按最多 10 掩码预留 2 GiB 图像预算；视频最多 64 帧，
  待 VLM 确认最多 32 帧。未打包且持有地图的 `_scan_submissions` 单独限为 32 帧。
  新增视频满额跳过，扫描满额报错；已有任务不被逐出。
  未命中/检测失败后释放，命中物体保留至定位，场景只保留线索位姿；退出清空。
  `semantic_snapshot_count/bytes` 是实际快照图像使用量，不是进程 RSS。
- `snapshot_store.freeze_depth` 保存只读 float32 米制数组，None 为 NaN；必须与 RGB 对齐。
  `clue_frame` 复用历史 RGB-D、拍摄位姿与标定，同时保留当前同坐标系地图。
  不再写 `job-*`、深度文件或 `object-localization/`；日志和可视化不参与算法数据交接。
- `perception/snapshot_store.py` 保留原始 RGB 分辨率和标定；模型输入无 V/F 标注。
  不再使用地面 Frontier 投影过滤评分，扫描覆盖本身的深度支持检查仍保留。
- `perception/analyzer.py` 的物体输出为 `score`、`bbox_2d`（0～1000 坐标或 null）和 `confidence`，
  场景输出为 `score`、`found` 和 `confidence`。解析后 `SemanticAnalysis.bbox_norm` 为 0～1；
  `found=None` 表示检测失败，False 表示有效未命中；评分与检测独立校验。
  无足够线索用 0.5，没看到目标不是方向反证。模型不计算路径距离或可达性。
- `TargetClue.bbox_norm` 只属于该线索的历史 RGB-D，物体定位直接复用，不再次请求 VLM。
  `OpenAICompatibleTargetObserver` 拒绝非 stop 的 chat completions 结束原因；
  本地 `qwen3.5:4b` 使用 JSON Schema、零温度、关闭思考、输出上限 128 token。
  Ollama 0.34.4 实测不能依赖 Schema 的 minimum/maximum 保证数值范围，解析器校验必须保留。
  随车模型服务使用 `OLLAMA_KEEP_ALIVE=5m`，不再使用服务 `ExecStartPost` 预热。
  `launch.py::_build_analyzer` 在随机模式分支之后调用 `adapters/ollama_warmup.py`，
  仅对本机 `qwen3.5:4b` 预热视觉输入，请求显式 `keep_alive=5m`；无保活心跳。
  预热发生在设备创建之前，失败即终止启动。GPU 发现期间 `/api/version` 可能超时，
  预热函数在有界等待中处理该超时。其他客户端请求可覆盖 Ollama 的卸载计时。

- `SemanticPerception.begin_cycle` 接收结果：检测有效才登记覆盖，评分失败仍可保留目标线索。
  `_pending_coverage` 在结果被主循环接收前一直保留；失败后移除，不冒充已检查。
  `pending_semantic_jobs` 还含快照准备、待接收结果和排队线索，不是 HTTP 数。
- `TargetClue.pose` 是拍摄时机器人位姿，`job_id/view_id` 关联同帧 RGB-D。
  场景由 `core/target.py::continue_scene_target` 返回并恢复朝向；位置误差 ≤0.25m、
  朝向误差 ≤5° 时返回 COMPLETE / target.revisit_complete。
  物体由 `core/target.py::continue_target_search` 处理 LOCALIZING_OBJECT → APPROACHING_OBJECT
  → COMPLETE。接近分支以停靠命令成功执行为依据，记录
  `completion_basis=standoff_command_completed`，不记录到达后重新测量的目标距离。
  `destination` 在停靠发出时写入，运动失败由 `recover_object_motion` 清空；下一周期
  接近状态仍有目的地即表示命令已成功执行，直接完成，不再调用模型。
  单条历史失败保留 `ObjectApproachState.fallback_clue/history_localized`，继续
  其他历史线索。`continue_object_history` 在 WAITING_FOR_SEMANTICS 原地排空
  已采集队列；全部不能定位才经 REVISITING_TARGET 保底返回，最终 STOPPED。
  STOPPED 以 CLI 退出码 3 结束，不再重采或调用确认。
  `app.py::_supply_perception` 每周期最多消费一次定位结果，队列
  `localize_object` 始终调用 `snapshot_store.clue_frame`，没有到达后当前帧定位分支。
  历史帧的位姿与标定用于定位，停靠始终使用当前帧的位姿和地图。
  `snapshot_store.clue_frame` 保留当前帧地图；未采集深度时 `depth=None`，不能借用新帧深度。
  `object_localization_started.timestamp_s` 是拍摄时间，`map_timestamp_s` 是当前导航帧时间。
  `SemanticPerception.begin_cycle` 按 FIFO 入队，物体按 job/view 保留每张不同快照；
  场景仍按位姿去重且不重新排序，批次按 FIFO。`core/target.py::discard_target_clue` 返回无命令状态，
  场景返回失败以 target.revisit_failed、物体线索失败以 object.clue_failed
  让下一周期优先取下一条，COMPLETE 与 STOPPED 不再取线索。
  `navigate` 的 active_target_clue 分支优先于普通视觉处理；`app.py` 跳过线索
  处理期间的普通目标观测，物体定位只响应 NEEDS_OBJECT_LOCALIZATION。
- `observe_rgbd_frame` 覆盖最新帧槽，`_video_loop` 默认每 0.1 秒最多取一次，不维护视频 FIFO。
  `yolo_frame.duration_s/queue_wait_s/backlog/received/replaced` 区分推理、等待和覆盖，backlog ≤1。
  视频不登记扫描覆盖；扫描图额外检测，故总 YOLO 请求率可能略高于配置的视频上限。
  同位置（0.5m）、朝向（15°）分别接纳一次中等分确认和一次高分命中。
  JSONL `completed.detection` 为融合依据，`vlm` 为原始 VLM 回答。
- `analyzer.match_target` 统一置信度规则。历史定位复用认可框；同帧 YOLOE 最高 IoU
  达到 `detection_box_iou` 时才使用该实例掩码，避免用另一个物体分割来测距。
  `yoloe.py` 使用 `retina_masks=True`，掩码须与原始 RGB 尺寸一致；框与掩码排序同步。
  掩码经过 packbits/base64 走现有 JSON 管道，RGB 是 JSON 头后的二进制，不产生临时文件。
  `ObjectModelProcess` 只维护 YOLOE，扫描和视频请求共用互斥锁；无 SAM2 入口。
- YOLOE 子进程在权重目录加载 `mobileclip2_b.ts`，移除 LD_PRELOAD/LD_LIBRARY_PATH/PYTHONHOME。
  `data/run_logs/semantic-*/models/yoloe.log` 保存加载阶段和超过 20 秒的调用栈；每请求有超时。
  `object_localized` JSONL 记录定位来源、bbox、距离、支持点数和耗时；Rerun 直接接收同帧图像/掩码。
  日志边界必须排除 `image/observation_frame/observation` 大对象，Rerun 索引也不能缓存完整输入。
- `perception/object_localizer.py::localize_segmented_object` 处理深度定位：只要求存在可计算的正深度，
  不设 8 点、有效率、中位深度集中度、连通块或占用格支持门槛。
  认可框不能提供 RGB-D 位置时，`localize_obstacle_on_image_ray` 沿框中心
  用格边遍历返回首个占用格的进入点。未知格不提供距离；
  不截断到 5m，也不设置假定距离。使用完整未膨胀导航图，缺图时沿用探索图。
  `source=bbox_obstacle`、`localization_method=obstacle_assumption`
  和 `sample_count=0` 表示位置假设。核心依据是否有有效坐标决定接近，不把这些值改成模型确认。
- `core/target.py::plan_object_standoff` 一次计算四邻接可达区，枚举目标
  周围 0.60–2.0m 全圆域自由格；优先最接近机器人侧 0.75m 理想位置，再按路径距离
  排序。Hermes 用完整图加 0.36m 净空，Habitat 沿用已有图，不再额外膨胀。
  `standoff_map`、`standoff_clearance_m`、`standoff_search_radius_m` 与
  `standoff_candidate_count` 记录选点依据；unknown/occupied/unreachable/excluded
  格数均带 `standoff_` 前缀与 `_cells` 后缀，unreachable 包括净空膨胀排除。
  排查“无停靠点”先看这些字段。
- `app.py::_capture_scan_direction` 将帧和 `capture_context(frame, state)` 同次传给
  `capture_scan_view`；评分显式传入 `frame.obstacle_map.frame_id`，队列不缓存当前上下文。
  `launch.py::_assemble_and_run` 创建同一份目标传给感知组件与导航循环。
- `app.py::_sync_perception_pause` 同步暂停标志。
  有待查线索、active_target_clue、目标返回/接近/定位阶段或导航终态时暂停。
  `_worker_loop` 等待暂停解除；YOLOE 线程仍限频检测，但暂停期间不新增视频快照。
  `SemanticPerception.begin_cycle` 在暂停期间保留 `_completed` 与 `_pending_coverage`；
  接收后由 `core/navigator.py::receive_perception` 更新搜索状态。
  两条已有线索之间保持暂停；物体线索用完时原地恢复已采集队列，继续接收线索。
  `fallback_clue` 本身不使队列暂停，否则会在等待历史结果时自锁。
  已开始的请求/采样可以完成并保存结果；暂停不是取消已发送的 HTTP 请求。
  `semantic_background_paused` 进入周期诊断，Rerun 简表显示普通队列暂停提示。
- 场景判断由联合分析完成；模型协议仅包含 `analyze_view`，
  没有同步观察器、到达后模型确认或检测触发的运动中断分支。
- 分数只在下一次决策读取，不修改执行中的目标；默认 CLI 不使用本地 YOLO 中断。
  `wait_for_semantics_or_finish` 必须等待任务与线索耗尽；失败检测计数单独报告。
  `close` 停止出队，关闭 YOLOE、收尾准备线程并释放内存快照；在途 HTTP 只有限等待，不续跑任务。

## 导航退出路径排查

- OpenCode Go 的 `HTTP 400 / MissingSessionID` 表示请求已到服务端，但缺少
  `x-opencode-session`；不能归因为 DNS 失败或直接认定凭据错误。要求见
  [官方接入说明](https://opencode.ai/docs/go/#where-can-i-use-it)。
  `launch.py::_build_analyzer` 为每次导航生成 UUID，填入
  `OpenAICompatibleConfig.opencode_session_id`，`_post_json` 在各类请求中复用。
  不要每次 HTTP 请求重新生成；User-Agent 保持 robot-nav 的真实标识。
  查看 JSONL 中 `semantic_queue` 事件的 detection_error/scoring_error 或终端请求错误；
  仅有导航周期继续输出不代表模型成功，普通模型失败会继续几何探索。
- Hermes 的 `MoveToAction` 在运动帧中检查底盘位姿：朝向剩余路径后连续停留在
  0.25 m 半径内达到 `blocked_pose_duration_s` 时，先取消并确认动作结束，再建人工墙。
  根配置与随车配置均为 5s；扫描转向不计时，MoveTo 朝向路径偏差超过 25° 时重置计时，
  平移达到 0.25 m 也重新开始。近深度只细化墙的位置，不参与触发判断。
  “前向采样区未检出近障碍”指 0.1～`front_blockage_distance_m` 范围内无合格深度簇，
  不能据此判断目标深度缺失；目标定位看 `object_localized` 的 `mask_used/sample_count/reason`。
  `--action-stall-timeout-s` 只控制转向无进展上限。Action 轮询默认 0.2s，
  运动帧间隔默认 0.5s；实际触发时间还受相机采集与 REST 延迟影响。
- `adapters/hermes/adapter.py::_check_stable_arrival` 对 working Action 检查目标
  位置误差 ≤`action_arrival_position_m`（0.30m）或转向误差 ≤`yaw_tolerance_rad`（5°）。
  在容差内连续 `action_arrival_hold_s`（0.001s，实际至少等到后续轮询）相对采样锚点移动 <2cm、转动 <1° 才
  抛出内部到达信号；MoveTo 还要求先有有效平移。`_execute_monitored_action` 捕获后
  保留任务供下一 Action 替换；没有下一动作、进入定位、异常或退出时才取消并确认终态。
  该到位交接不等于固件报告成功。该容差不同于
  `position_tolerance_m`（0.03m），后者仅决定是否下发微小平移命令。
- `app.py::run_navigation` 在目标完成、`FAILED`、非 `OK` 且非等待感知状态，
  或达到 `max_cycles` 时退出；`MISSING_DATA` 当前也立即退出。周期上限不能当作
  Frontier 耗尽；`OK + WAITING_FOR_SEMANTICS` 每次最多等待结果 1 秒，不消耗决策
  额度。普通模型失败记为失败批次、评分降级。
- 物体线索处理由状态机持续推进，NEEDS_OBJECT_LOCALIZATION 属于等待感知状态。
  `core/target.py::recover_object_motion` 对 object.approach 清空目的地，
  下一周期用最新地图换停靠点；object.fallback_return/object.fallback_turn 失败则 STOPPED。
  接近阶段不沿用 Frontier 整区域屏蔽，三次停靠仍失败才放弃线索。
- `core/navigator.py::apply_execution_result`
  对 `scan.turn` 使用 `scan.recover_scan_turn`，清除扫描计划后按真实朝向重新规划，
  不登记失败方向的覆盖；所有扫描都按当前边界与已有覆盖重新规划。
- `environment.py::_move_hermes_forward_on_start` 在 `run_navigation` 之前直接调用
  底盘；前移 1 m 的可恢复失败/停滞在确认动作结束后继续启动，其他异常仍停止。
- `adapters/hermes/adapter.py::_execute_action` 将 Action 创建、起始位姿读取和监控
  放在同一异常范围；`_cancel_active_action` 与 `close` 负责取消遗留动作。已获得
  ID 时还要确认终态；创建请求失败而没有 ID 时只能尝试取消，原异常仍终止运行。
- `run_log.optional_callback` 集中处理可视化的 I/O 与运行库故障；
  类型或字段错误继续传播。Adapter 不再重复捕获回调异常；JSONL `_write` 只隔离
  文件 I/O 错误，序列化错误直接暴露。直接传入 Adapter 或周期函数的自定义回调
  由调用方负责。采集、路径检查、健康错误仍停止并收尾动作。
- 扫描准备失败或内存容量耗尽会传回主循环并停止；视频容量满只跳过新增候选。
  队列停止后在途 HTTP 可能完成；其返回不重新插入已清空的快照或任务。

- `SemanticPerception._run_worker` 只负责把后台异常传到 `begin_cycle` / `wait_for_result`，
  不转成普通检测失败。排查队列停止先看原始异常堆栈；不要通过增加宽泛捕获恢复等待。
- `core/navigator.py::validation_error` 只检查进入决策的物理数据，不遍历检查 `SearchState` 的字段类型。
  状态错误应回到创建或更新该状态的行为模块修复。

## 离线读取 RRD

- `visualization/vlm_trace.py` 只保存轻量摘要，`panels.py` 生成完整卡片，
  `rerun_view.py` 记录卡片和 World 叠加；显示坐标转换在 `view_geometry.py`。`model/vlm/summary` 是简表，`model/interaction` 是最新完整事件；
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
  `chassis/status:Text` 保存健康与 Action 原始反馈，JSON 正文中的各项 `received_at`
  才是对应采集时刻；同一健康告警会在后续 Action 更新时重复展示，分析时需去重。
  碰撞等 `baseError` 告警目前不展开到 JSONL，排查接近受阻时应同时读取该 RRD 实体。
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

- Hermes 人工墙：`front_obstruction.update_path` 用剩余路径首个距机器人 ≥0.20m 的点判断
  行进方向（暂无路径时使用目标）；朝向差 >25° 清空计时窗口。扫描 RotateTo 不启用该检测。
  `_continuous_frame_loop` 仅发布阻塞事实，`_execute_monitored_action` 确认取消终态后调用
  `_commit_front_obstruction`。`HermesObservedMap.navigation_map` 向完整图叠加同一份原始墙，
  `add_permanent_wall` 按停止后 REST 位姿检查净空并拒绝让膨胀墙覆盖当前格。
