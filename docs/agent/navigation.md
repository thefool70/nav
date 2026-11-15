# 搜索与恢复排错

面向 Agent。行为定义见 [算法说明](../algorithm.md)，此处用于按日志定位实现。

## Frontier 与观察记录排错

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
- `navigation/exploration.py::refresh_frontier_regions` 返回更新状态与 `FrontierExtraction`；
  `navigation/exploration.py::match_frontier_regions` 用世界坐标边界关联 ID。历史节点只保留
  实际尝试的 Frontier 移动，节点的出发位置也是本轮未选方向的父节点。
  JSONL `state.latest_node` 直接记录 `candidate_id`、`destination_world_xy`、
  `state` 与 `execution_reason`，不嵌套 `directions`；`COMMITTED` 尚不参与
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
  `pipeline/navigation_loop.py::_execute_action` 等待同步动作结束后才允许下一周期决策；
  排查途中改目标时先核对 Adapter 完成反馈。
- Action 进度 `unknown_path=长度/上限` 为当前轮询测量；取消异常保留
  `unknown_length_m`、`limit_m`、`total_path_length_m`，`pipeline/navigation_loop.py` 将其写入
  `unknown_path_length_m`、`unknown_path_limit_m`、`checked_path_length_m`。
  这些诊断不改变超限后的区域屏蔽、返回恢复和线索放弃流程。
- 未知路径长度超限、Action 执行失败、停滞或目标检测中断触发取消后，均用
  `wait_for_action(require_success=False)` 确认 Action 进入终态；确认超时使用
  `request_timeout_s`（默认 5 秒）。取消或
  确认失败不得转换为可恢复异常，否则下一目标可能覆盖仍在执行的动作。成功后
  将异常及其完整 `path_world_xy` 送回 `pipeline/navigation_loop.py`；必须先于父类
  `RecoverableMotionError` 捕获，不能只转成字符串丢失路径。
- `navigation/navigator.py::apply_execution_result` 接收 `PATH_UNKNOWN` 反馈后将尝试标为
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
  Rerun `scan basis` 显示对应原因。
- 扫描规划日志的 `frontier_scan_cell_count` 是移动筛选前的边界格数，
  `frontier_move_candidate_count` 是筛选后的移动候选数，两者可能一个非零、一个为零。
  `WAITING_FOR_SEMANTICS` 也检查这些边界的未观察覆盖，无移动候选时仍可恢复扫描。
- `navigation/scan.py::_local_coverage_points` 的 0.25 m 固定世界
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
- 排查“没有优先向前”时检查 `navigation/exploration.py::select_exploration_target`：当前排序
  先分新候选与暂存候选，`FrontierScoreRequest.candidates` 才查询分数，`deferred` 原序保存。
  运行层查询缓存后直接进入 `complete_frontier_selection`，不重新调用 `navigate` 或刷新区域。
  没有新候选时才逐个回到父节点，遇到该节点的有效暂存方向按原顺序恢复，不使用区域切换扣分。
  扫描结束时的车头 yaw 不参与分层。`frontier_selection_source` 为 `new` 或
  `deferred`；提交 Frontier 移动时，新旧数量是本次选择前的数量；
  JSONL 状态的 `deferred_frontiers` 是选择后的暂存队列，Rerun 同样显示来源与队列数量。
- 场景任务停靠后由 `pipeline/navigation_loop.py::_confirm_scene_at_target` 执行环视采图，确认通过才完成；
  物体模式以停靠命令成功执行为完成依据，不再请求到达感知。
  两种模式无局部待查 Frontier 时均用当前 yaw 采集一张图入队，没有待查方向也保留
  当前画面。异步选点仍查严格匹配的评分缓存；未拍到的方位不投影到图片边缘。
- `INVALIDATED` 表示执行失败，`STALLED` 表示停滞；二者均不证明 navmesh
  不连通。有紫色路径但无运动帧时，继续检查 Habitat `_follow_path()` 的动作生成。
  周期回调发生在执行前，异常原因通过历史方向的 `execution_reason` 保留，
  下一周期 Rerun 的 `last exploration issue` 显示最近一次异常及其节点编号。



## 导航退出路径排查

- `pipeline/navigation_loop.py::run_navigation` 在目标完成、`FAILED`、非 `OK` 且非等待感知状态，
  或达到 `max_cycles` 时退出；`MISSING_DATA` 当前也立即退出。周期上限不能当作
  Frontier 耗尽；`OK + WAITING_FOR_SEMANTICS` 每次最多等待结果 1 秒，不消耗决策
  额度。普通模型失败记为失败批次、评分降级。
- 物体线索处理由状态机持续推进，NEEDS_OBJECT_LOCALIZATION 属于等待感知状态。
  `navigation/target_search.py::recover_object_motion` 对 object.approach 清空目的地，
  下一周期用最新地图换停靠点；object.fallback_return/object.fallback_turn 失败则 STOPPED。
  接近阶段不沿用 Frontier 整区域屏蔽，三次停靠仍失败才放弃线索。
- `navigation/navigator.py::apply_execution_result`
  对 `scan.turn` 使用 `scan.recover_scan_turn`，清除扫描计划后按真实朝向重新规划，
  不登记失败方向的覆盖；所有扫描都按当前边界与已有覆盖重新规划。
- `run_log.optional_callback` 集中处理可视化的 I/O 与运行库故障；
  类型或字段错误继续传播。Adapter 不重复捕获回调异常；JSONL `_write` 只隔离
  文件 I/O 错误，序列化错误直接暴露。直接传入 Adapter 或周期函数的自定义回调
  由调用方负责。采集、路径检查、健康错误仍停止并收尾动作。
- 扫描准备失败或内存容量耗尽会传回主循环并停止；视频容量满只跳过新增候选。
  队列停止后在途 HTTP 可能完成；其返回不重新插入已清空的快照或任务。

- `SemanticPerception._run_worker` 只负责把后台异常传到 `begin_cycle` / `wait_for_result`，
  不转成普通检测失败。排查队列停止先看原始异常堆栈；不要通过增加宽泛捕获恢复等待。
- `navigation/navigator.py::validation_error` 只检查进入决策的物理数据，不遍历检查 `SearchState` 的字段类型。
  状态错误应回到创建或更新该状态的行为模块修复。
