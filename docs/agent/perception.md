# 感知排错

面向 Agent。入口为 `perception/session.py`；不得把模型错误当作无目标结果。

## 异步视觉队列

- 连接故障先读 JSONL 中 `semantic_queue` 事件的 `detection_error`，区分 HTTP
  状态码与收到响应前断开。连接失败不能直接证明密钥失效、额度耗尽或 Qwen 全局宕机。
  `_post_json` 使用默认 urllib opener，会继承启动进程的 HTTP(S) 代理设置；
  排查 OpenCode Go 链路时核对该进程的代理环境，不用当前工具进程环境替代历史证据。
  不输出凭据值，也不要把连接失败当作“未检测到目标”。
- `VlmInteraction.context` 与 `SemanticAnalyzer.analyze_view(trace_context=...)`
  只增加记录来源，不参与提示词或 HTTP payload；不能为了显示关联关系串用
  `_scan_images` 或共享可变的当前任务 ID。场景停靠后提交环视合图确认；物体定位通过 job/view 编号关联历史画面并复用已认可的框和掩码。
- `SemanticPerception._scores` 每项保存 `(score, job_id, interaction_id)`，保留 J/R 来源；
  `semantic_received_jobs` 是该周期接收结果，`semantic_score_sources` 是传给
  排序的有效缓存输入。两者不是同一件事，返回有效分数不等于已用于导航。
- `perception/session.py::SemanticPerception` 持有 RGB-D 内存快照，准备线程只做扫描
  RGB 打包与深度复制；VLM 线程对扫描图请求同帧 YOLOE，再按 FIFO 调用 `analyze_view`。
  视频高分结果绕过准备线程和 VLM 队列。`begin_cycle` 只返回覆盖，线索另行领取。
  分数缓存匹配地图系、原区域 ID 与世界目标，保留首次有效分数。
- `_snapshots` 最多 256 帧、按最多 10 掩码预留 2 GiB 图像预算；视频最多 64 帧，
  待 VLM 确认最多 32 帧。未打包且持有地图的 `_scan_submissions` 单独限为 32 帧。
  新增视频满额跳过，扫描满额报错；已有任务不被逐出。
  未命中/检测失败后释放，命中线索保留至定位；退出清空。
  `semantic_snapshot_count/bytes` 是实际快照图像使用量，不是进程 RSS。
- `snapshots.freeze_depth` 保存只读 float32 米制数组，None 为 NaN；必须与 RGB 对齐。
  `clue_frame` 复用历史 RGB-D、拍摄位姿与标定，同时保留当前同坐标系地图。
  不写快照目录或深度文件；日志和可视化不参与算法数据交接。
- `perception/snapshots.py` 保留原始 RGB 分辨率和标定；模型输入无 V/F 标注。
  不使用地面 Frontier 投影过滤评分，扫描覆盖本身的深度支持检查仍保留。
- `perception/vlm_analysis.py` 的物体输出为 `score`、`bbox_2d`（0～1000 坐标或 null）和 `confidence`，
  场景输出为 `score`、`found` 和 `confidence`。解析后 `SemanticAnalysis.bbox_norm` 为 0～1；
  `found=None` 表示检测失败，False 表示有效未命中；评分与检测独立校验。
  无足够线索用 0.5，没看到目标不是方向反证。模型不计算路径距离或可达性。
- `TargetClue.bbox_norm` 只属于该线索的历史 RGB-D，物体定位直接复用，不再次请求 VLM。
  `OpenAICompatibleTargetObserver` 拒绝非 stop 的 chat completions 结束原因；
  本地 `qwen3.5:4b` 使用 JSON Schema、零温度、关闭思考、输出上限 128 token。
  Ollama 0.34.4 实测不能依赖 Schema 的 minimum/maximum 保证数值范围，解析器校验必须保留。
  随车模型服务使用 `OLLAMA_KEEP_ALIVE=5m`，服务启动不预热模型。
  `pipeline/launch.py::_build_analyzer` 在随机模式分支之后调用 `perception/models/ollama_warmup.py`，
  仅对本机 `qwen3.5:4b` 预热视觉输入，请求显式 `keep_alive=5m`；无保活心跳。
  预热发生在设备创建之前，失败即终止启动。GPU 发现期间 `/api/version` 可能超时，
  预热函数在有界等待中处理该超时。其他客户端请求可覆盖 Ollama 的卸载计时。

- `SemanticPerception.begin_cycle` 接收结果：检测有效才登记覆盖，评分失败仍可保留目标线索。
  `_pending_coverage` 在结果被主循环接收前一直保留；失败后移除，不冒充已检查。
  `pending_semantic_jobs` 还含快照准备、待接收结果和排队线索，不是 HTTP 数。
- `TargetClue.pose` 是拍摄时机器人位姿，`job_id/view_id` 关联同帧 RGB-D。
  两种模式均由 `navigation/target_search.py::continue_target_search` 执行物体定位与接近。
  场景只在停靠成功后请求 NEEDS_SCENE_CONFIRMATION，进入 CONFIRMING_SCENE；
  `pipeline/navigation_loop.py::_confirm_scene_at_target` 顺序执行原地一圈、停稳采图和一次编号网格合图确认。
  场景尾处理不进入暂停的普通 FIFO；转向失败经 SCENE_TURN → discard_object_clue 恢复。
  物体由 `navigation/target_search.py::continue_target_search` 处理 LOCALIZING_OBJECT → APPROACHING_OBJECT
  → COMPLETE。接近分支以停靠命令成功执行为依据，记录
  `completion_basis=standoff_command_completed`，不记录到达后重新测量的目标距离。
  `destination` 在停靠发出时写入，运动失败由 `recover_object_motion` 清空；下一周期
  接近状态仍有目的地即表示命令已成功执行；object 直接完成，scene 转入环视确认。
  单条历史失败保留 `ObjectApproachState.fallback_clue/history_localized`，继续
  其他历史线索。`continue_object_history` 在 WAITING_FOR_SEMANTICS 原地排空
  已采集队列；全部不能定位才经 REVISITING_TARGET 保底返回，最终 STOPPED。
  STOPPED 以 CLI 退出码 3 结束，不再重采或调用确认。
  `pipeline/navigation_loop.py::_supply_perception` 每周期最多消费一次定位结果，队列
  `localize_object` 始终调用 `snapshots.clue_frame`，没有到达后当前帧定位分支。
  历史帧的位姿与标定用于定位，停靠始终使用当前帧的位姿和地图。
  `snapshots.clue_frame` 保留当前帧地图；未采集深度时 `depth=None`，不能借用新帧深度。
  `object_localization_started.timestamp_s` 是拍摄时间，`map_timestamp_s` 是当前导航帧时间。
  `SemanticPerception.begin_cycle` 按 FIFO 入队，物体按 job/view 保留每张不同快照；
  scene 的普通检测也使用物体语义与固定类别清单，线索同样按单图任务去重、可用顺序交付。
  两种模式的线索失败均通过 `navigation/target_search.py::discard_object_clue` 返回 object.clue_failed，
  让下一周期优先取下一条，COMPLETE 与 STOPPED 不再取线索。
  `navigate` 的 active_target_clue 分支优先于普通视觉处理；`pipeline/navigation_loop.py` 跳过线索
  处理期间的普通目标观测，物体定位只响应 NEEDS_OBJECT_LOCALIZATION。
- `observe_rgbd_frame` 覆盖最新帧槽，`_video_loop` 默认每 0.1 秒最多取一次，不维护视频 FIFO。
  `yolo_frame.duration_s/queue_wait_s/backlog/received/replaced` 区分推理、等待和覆盖，backlog ≤1。
  视频不登记扫描覆盖；扫描图额外检测，故总 YOLO 请求率可能略高于配置的视频上限。
  同位置（0.5m）、朝向（15°）分别接纳一次中等分确认和一次高分命中。
  JSONL `completed.detection` 为融合依据，`vlm` 为原始 VLM 回答。
- `detection_fusion.match_target` 统一置信度规则。历史定位复用认可框；同帧 YOLOE 最高 IoU
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
- `navigation/target_search.py::plan_object_standoff` 一次计算四邻接可达区，枚举目标
  周围 0.60–2.0m 全圆域自由格；优先最接近机器人侧 0.75m 理想位置，再按路径距离
  排序。Hermes 用完整图加 0.36m 净空，Habitat 沿用已有图，不再额外膨胀。
  `standoff_map`、`standoff_clearance_m`、`standoff_search_radius_m` 与
  `standoff_candidate_count` 记录选点依据；unknown/occupied/unreachable/excluded
  格数均带 `standoff_` 前缀与 `_cells` 后缀，unreachable 包括净空膨胀排除。
  排查“无停靠点”先看这些字段。
- `pipeline/navigation_loop.py::_capture_scan_direction` 将帧和 `capture_context(frame, state)` 同次传给
  `capture_scan_view`；评分显式传入 `frame.obstacle_map.frame_id`，队列不缓存当前上下文。
  `pipeline/launch.py::_assemble_and_run` 创建任务目标；场景固定类别在 `_build_perception` 内补充后传给感知，导航循环仍接收原任务目标。
- `pipeline/navigation_loop.py::_sync_perception_pause` 同步暂停标志。
  有待查线索、active_target_clue、目标返回/接近/定位阶段或导航终态时暂停。
  `_worker_loop` 等待暂停解除；YOLOE 线程仍限频检测，但暂停期间不新增视频快照。
  `SemanticPerception.begin_cycle` 在暂停期间保留 `_completed` 与 `_pending_coverage`；
  接收后由 `navigation/navigator.py::receive_perception` 更新搜索状态。
  两条已有线索之间保持暂停；物体线索用完时原地恢复已采集队列，继续接收线索。
  `fallback_clue` 本身不使队列暂停，否则会在等待历史结果时自锁。
  已开始的请求/采样可以完成并保存结果；暂停不是取消已发送的 HTTP 请求。
  `semantic_background_paused` 进入周期诊断，Rerun 简表显示普通队列暂停提示。
- 模型协议包含单图分析和场景线索生成；停靠后的场景确认复用 `analyze_view`，
  由运行层直接调用，不进入暂停的普通队列。后台检测不触发运动中断。
- 分数只在下一次决策读取，不修改执行中的目标；默认 CLI 不使用本地 YOLO 中断。
  `wait_for_semantics_or_finish` 必须等待任务与线索耗尽；失败检测计数单独报告。
  `close` 停止出队，关闭 YOLOE、收尾准备线程并释放内存快照；在途 HTTP 只有限等待，不续跑任务。

## 场景确认

- 场景前处理在 `pipeline/launch.py::_build_perception` 先读 9 个常用场景预设，其余由 Qwen 生成 3–5 类固定清单，生成失败重试一次。预设与文本提示词映射在 `perception/models/yoloe_targets.json`。
  `_goal.object_classes` 传入 YOLOE 和 Qwen；`class_text` 随框贯通，联合融合及掩码匹配要求同类。
  场景否决后的实例记录属于感知队列；同地图、同类且 0.5m 内的世界位置对应被排除线索。
  `scene_objects_rejected.unlocalized_objects` 表示入镜检测未能定位，不能可靠跨图排除。
  `scene_view` 留存各方向图，`scene_confirmation` VLM 请求显示实际合图；只看普通 FIFO 无法复盘尾处理。

## 执行与退出定位

- OpenCode Go 的 `HTTP 400 / MissingSessionID` 表示请求已到服务端，但缺少
  `x-opencode-session`；不能归因为 DNS 失败或直接认定凭据错误。要求见
  [官方接入说明](https://opencode.ai/docs/go/#where-can-i-use-it)。
  `pipeline/launch.py::_build_analyzer` 为每次导航生成 UUID，填入
  `OpenAICompatibleConfig.opencode_session_id`，`_post_json` 在各类请求中复用。
  不要每次 HTTP 请求重新生成；User-Agent 保持 robot-nav 的真实标识。
  查看 JSONL 中 `semantic_queue` 事件的 detection_error/scoring_error 或终端请求错误；
  仅有导航周期继续输出不代表模型成功，普通模型失败会继续几何探索。