"""生成导航状态文本与模型交互卡片；不调用 Rerun、不修改导航状态。"""

from __future__ import annotations

import math
import os
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from ..adapters.perception import VlmInteraction
from ..core.actions import action_command
from ..core.models import (
    MaskImage, NavigationFrame, NavigationResult, SearchDirectionState, TargetObservation,
)
from .view_geometry import command_world_vector
from .vlm_trace import context_text

BBOX_RGB = (0, 255, 80)
STATUS_IMAGE_WIDTH = 560
STATUS_FONT_SIZE = 16
STATUS_LINE_SPACING_PX = 6
STATUS_PADDING_PX = 12
STATUS_BG_RGB = (24, 24, 24)
STATUS_TEXT_RGB = (235, 235, 235)
VLM_CARD_MIN_WIDTH = 960
VLM_CARD_MAX_WIDTH = 1400
VLM_CARD_PADDING_PX = 24
VLM_CARD_GAP_PX = 14
VLM_CARD_BORDER_RGB = (75, 90, 105)
VLM_CARD_TITLE_RGB = (100, 210, 255)
VLM_CARD_SECTION_RGB = (255, 190, 90)
VLM_CARD_IMAGE_BORDER_RGB = (120, 135, 150)
STATUS_FONT_CANDIDATES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
_FONT_NOTICE_PRINTED = False


def vlm_card_text(interaction: VlmInteraction) -> str:
    """构造单卡片的文本回退内容。"""
    parts = [
        (
            f"VLM R{interaction.interaction_id} | "
            f"{interaction.task} | {_vlm_card_status(interaction)}"
        )
    ]
    for title, body in _vlm_card_sections(interaction):
        parts.append(f"[{title}]\n{body}")
        if title == "完整输入提示词":
            parts.append(
                "[实际输入 RGB]\n"
                f"{interaction.image.width_px}x"
                f"{interaction.image.height_px} RGB"
            )
    return "\n\n".join(parts)


def render_vlm_interaction_card(
    font: object,
    interaction: VlmInteraction,
) -> np.ndarray:
    """把完整 VLM 交互和输入图渲染为一张 CJK 卡片。"""
    from PIL import Image, ImageDraw

    input_image = _vlm_input_pil_image(font, interaction)
    max_inner_width = VLM_CARD_MAX_WIDTH - 2 * VLM_CARD_PADDING_PX
    if input_image.width > max_inner_width:
        scale = max_inner_width / input_image.width
        input_image = input_image.resize(
            (
                max_inner_width,
                max(1, round(input_image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    card_width = max(
        VLM_CARD_MIN_WIDTH,
        min(
            VLM_CARD_MAX_WIDTH,
            input_image.width + 2 * VLM_CARD_PADDING_PX,
        ),
    )
    max_text_width = card_width - 2 * VLM_CARD_PADDING_PX
    wrapped_sections = [
        (
            title,
            _wrap_multiline_text(font, body, max_text_width),
        )
        for title, body in _vlm_card_sections(interaction)
    ]
    line_height = STATUS_FONT_SIZE + STATUS_LINE_SPACING_PX
    card_height = 2 * VLM_CARD_PADDING_PX + line_height
    for title, lines in wrapped_sections:
        card_height += VLM_CARD_GAP_PX + line_height
        card_height += len(lines) * line_height
        if title == "完整输入提示词":
            card_height += VLM_CARD_GAP_PX + line_height + input_image.height

    card = Image.new("RGB", (card_width, card_height), STATUS_BG_RGB)
    draw = ImageDraw.Draw(card)
    draw.rectangle(
        (0, 0, card_width - 1, card_height - 1),
        outline=VLM_CARD_BORDER_RGB,
        width=2,
    )
    y = VLM_CARD_PADDING_PX
    draw.text(
        (VLM_CARD_PADDING_PX, y),
        (
            f"VLM R{interaction.interaction_id} | "
            f"{interaction.task} | {_vlm_card_status(interaction)}"
        ),
        font=font,
        fill=VLM_CARD_TITLE_RGB,
    )
    y += line_height

    for title, lines in wrapped_sections:
        y += VLM_CARD_GAP_PX
        draw.line(
            (
                VLM_CARD_PADDING_PX,
                y + line_height - 2,
                card_width - VLM_CARD_PADDING_PX,
                y + line_height - 2,
            ),
            fill=VLM_CARD_BORDER_RGB,
            width=1,
        )
        draw.text(
            (VLM_CARD_PADDING_PX, y),
            title,
            font=font,
            fill=VLM_CARD_SECTION_RGB,
        )
        y += line_height
        for line in lines:
            draw.text(
                (VLM_CARD_PADDING_PX, y),
                line,
                font=font,
                fill=STATUS_TEXT_RGB,
            )
            y += line_height

        if title != "完整输入提示词":
            continue
        y += VLM_CARD_GAP_PX
        draw.text(
            (VLM_CARD_PADDING_PX, y),
            (
                "实际输入 RGB "
                f"({interaction.image.width_px}x"
                f"{interaction.image.height_px})"
            ),
            font=font,
            fill=VLM_CARD_SECTION_RGB,
        )
        y += line_height
        image_x = (card_width - input_image.width) // 2
        card.paste(input_image, (image_x, y))
        draw.rectangle(
            (
                image_x,
                y,
                image_x + input_image.width - 1,
                y + input_image.height - 1,
            ),
            outline=VLM_CARD_IMAGE_BORDER_RGB,
            width=2,
        )
        y += input_image.height
    return np.asarray(card)


def motion_status_text(
    frame: NavigationFrame,
    result: NavigationResult,
) -> str:
    """构造侧栏实时摘要；位姿随运动帧更新，阶段来自最近一次决策。"""
    lines = [
        f"decision: {result.debug.stage} | next: {result.state.phase.value}",
        (
            f"pose x={frame.pose.x_m:.2f} y={frame.pose.y_m:.2f} "
            f"yaw={math.degrees(frame.pose.yaw_rad):.1f}deg"
        ),
    ]
    headings = result.state.scan_headings_world_rad
    if result.state.asynchronous_perception:
        lines.append(
            f"semantic pending={result.state.pending_semantic_jobs} "
            f"failed={result.state.failed_semantic_jobs} "
            f"queued views={len(result.state.pending_observation_views)}"
        )
    if result.state.active_target_clue is not None:
        lines.append(f"target clue: {result.state.active_target_clue.clue_id}")
    approach = result.state.object_approach
    if approach.target is not None:
        lines.append(
            f"object source={approach.target.source} target={approach.target.target_world_xy} "
            f"moves={len(approach.tried_positions)} depth points={approach.target.sample_count}"
        )
    planning = result.debug.details
    if "standoff_map" in planning:
        lines.append(
            f"standoff map={planning['standoff_map']} "
            f"clearance={planning['standoff_clearance_m']:.2f}m "
            f"candidates={planning['standoff_candidate_count']}"
        )
        if "standoff_distance_m" in planning:
            lines.append(f"planned distance to target={planning['standoff_distance_m']:.2f}m")
    if headings and "scan_mode" in result.debug.details:
        scan_index = min(result.state.next_scan_index, len(headings) - 1)
        lines.append(
            f"scan {scan_index + 1}/{len(headings)} "
            f"yaw={math.degrees(headings[scan_index]):.1f}deg"
        )
        lines.append(_scan_basis_text(result.debug.details["scan_mode"]))
    if action_command(result.action, frame.pose) is not None:
        command = action_command(result.action, frame.pose)
        lines.append(
            f"command move={math.hypot(command.forward_m, command.left_m):.2f}m "
            f"turn={math.degrees(command.yaw_rad):.1f}deg"
        )
    selected_id = result.debug.details.get("candidate_id")
    parent_id = result.debug.details.get("parent_node_id") or result.state.backtrack_node_id
    if parent_id is not None:
        lines.append(f"parent: {parent_id}")
    if "branch_depth" in result.debug.details:
        lines.append(
            f"return depth: {result.debug.details['branch_depth']} | "
            f"pending at parent: {result.debug.details['pending_direction_count']}"
        )
    if selected_id is not None:
        lines.append(
            f"frontier: {selected_id} ({result.debug.details.get('frontier_selection_source')})"
        )
    return "\n".join(lines)


def frontier_table_text(
    candidates: Tuple[Mapping[str, Any], ...],
    selected_id: Optional[str],
    result: NavigationResult,
) -> str:
    """表格行与 Points2D 实例顺序一致；编号链接到对应点，暂存状态取当前状态。"""
    blocked_ids = ", ".join(
        region.region_id for region in result.state.blocked_frontier_regions
    )
    blocked_text = (
        f"Blocked regions (no retry during this run): {blocked_ids}."
        if blocked_ids else "Blocked regions: none."
    )
    if not candidates:
        return f"No current frontier candidates.\n\n{blocked_text}"
    orders = {region.region_id: region.deferred_order for region in result.state.frontier_regions}
    lines = [
        "Click an ID to select its World point. Yellow = selected; green = candidate.",
        "",
        "| Frontier | State | Saved order | Path (m) | Score |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        order = orders.get(candidate_id, candidate.get("deferred_order"))
        state = "selected" if candidate_id == selected_id else "deferred" if order is not None else "new"
        order_text = "-" if order is None else f"{order[0]}:{order[1]}"
        label = f"[{candidate_id}](recording://world/current_frontiers[#{index}])"
        lines.append(
            f"| {label} | {state} | {order_text} | "
            f"{float(candidate['path_distance_m']):.2f} | {float(candidate['score']):.2f} |"
        )
    lines.extend([
        "",
        blocked_text,
        "",
        "New directions rank by score. When they run out, return through the branch "
        "one node at a time until a reached node has a remaining direction to explore.",
        "Positions and distances are from the latest frontier update.",
    ])
    return "\n".join(lines)


def load_panel_font() -> Optional[object]:
    """加载面板 CJK 字体；Pillow 缺失或所有候选加载失败时返回 None。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in STATUS_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size=STATUS_FONT_SIZE)
        except OSError:
            continue
    return None


def print_font_notice_once() -> None:
    """Pillow/CJK 字体缺失时最多打印一次提示，避免每个周期刷屏。"""
    global _FONT_NOTICE_PRINTED
    if _FONT_NOTICE_PRINTED:
        return
    _FONT_NOTICE_PRINTED = True
    print("提示：Pillow 或 CJK 字体不可用，面板回退为 ASCII 文本")


def render_status_image(font: object, lines: List[str]) -> np.ndarray:
    """把状态行渲染为深色背景 RGB 图像，供 rr.Image 记录。"""
    from PIL import Image, ImageDraw

    max_pixels = STATUS_IMAGE_WIDTH - 2 * STATUS_PADDING_PX
    wrapped: List[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(font, line, max_pixels))

    line_height = STATUS_FONT_SIZE + STATUS_LINE_SPACING_PX
    height = 2 * STATUS_PADDING_PX + len(wrapped) * line_height
    image = Image.new("RGB", (STATUS_IMAGE_WIDTH, height), STATUS_BG_RGB)
    draw = ImageDraw.Draw(image)
    y = STATUS_PADDING_PX
    for line in wrapped:
        draw.text((STATUS_PADDING_PX, y), line, font=font, fill=STATUS_TEXT_RGB)
        y += line_height
    return np.asarray(image)


def ascii_only(text: str) -> str:
    """把非 ASCII 字符替换为 '?'，供无字体时生成纯 ASCII 回退文本。"""
    return "".join(char if ord(char) < 128 else "?" for char in text)


def status_lines(
    target_text: str,
    frame: NavigationFrame,
    observation: Optional[TargetObservation],
    result: NavigationResult,
) -> List[str]:
    """构造按决策顺序组织的状态面板，避免直接打印难读的 details 字典。"""
    lines = [
        f"target: {target_text}",
        (
            f"decision: status={result.status.value}, "
            f"phase={result.state.phase.value}, stage={result.debug.stage}"
        ),
        f"message: {result.debug.message}",
        (
            "pose: "
            f"x={frame.pose.x_m:.2f} m, y={frame.pose.y_m:.2f} m, "
            f"yaw={math.degrees(frame.pose.yaw_rad):.1f} deg"
        ),
    ]
    if action_command(result.action, frame.pose) is not None:
        command = action_command(result.action, frame.pose)
        world_vector = command_world_vector(command, frame.pose)
        target_world = (
            frame.pose.x_m + world_vector[0],
            frame.pose.y_m + world_vector[1],
        )
        lines.append(
            "command: "
            f"move={math.hypot(command.forward_m, command.left_m):.2f} m, "
            f"forward={command.forward_m:.2f} m, left={command.left_m:.2f} m, "
            f"turn={math.degrees(command.yaw_rad):.1f} deg, "
            f"target=({target_world[0]:.2f}, {target_world[1]:.2f})"
        )

    lines.extend(_scan_status_lines(result))
    lines.extend(_frontier_status_lines(result.debug.details))
    lines.extend(_observation_status_lines(frame, observation))
    lines.extend(_history_status_lines(result))
    lines.extend(
        (
            "map legend: robot=blue, frontier=green, selected=yellow, "
            "command=orange, adapter target=red, adapter path=purple",
            "history links: pending=yellow, committed=blue, explored=gray, "
            "invalidated=red, stalled=orange",
        )
    )
    return lines


def target_mask_to_numpy(mask: MaskImage) -> Optional[np.ndarray]:
    """把矩形二维掩码转换为 bool 数组；非法输入返回 None。"""
    try:
        mask_array = np.asarray(mask, dtype=bool)
    except (TypeError, ValueError):
        return None
    if mask_array.ndim != 2 or mask_array.size == 0:
        return None
    return mask_array


def _scan_status_lines(result):
    """汇总队列计数、扫描计划与覆盖情况。"""
    lines = []
    details = result.debug.details
    if result.state.asynchronous_perception:
        lines.append(
            "semantic queue: "
            f"pending={result.state.pending_semantic_jobs}, "
            f"finished={details.get('semantic_finished', 0)}, "
            f"failed={result.state.failed_semantic_jobs}, "
            f"active={details.get('semantic_active_job')}, "
            f"pending views={len(result.state.pending_observation_views)}"
        )
    if "scan_heading_count" in details:
        target_heading_deg = math.degrees(
            float(details.get("target_heading_world_rad", 0.0))
        )
        lines.append(
            "scan plan: "
            f"mode={details.get('scan_mode')}, "
            f"view={int(details.get('scan_index', 0)) + 1}/"
            f"{details.get('scan_heading_count')}, "
            f"target yaw={target_heading_deg:.1f} deg"
        )
        lines.append(_scan_basis_text(details.get("scan_mode", "unknown")))
    if "frontier_scan_candidate_count" in details:
        lines.append(
            "frontier map: "
            f"clusters={details['frontier_scan_candidate_count']}, "
            f"candidate_cells={details['frontier_scan_cell_count']}"
        )
    if "local_observation_point_count" in details:
        lines.append(
            "frontier coverage: "
            f"local_frontier_points={details.get('local_observation_point_count', 0)}, "
            f"need_check={details.get('observation_point_count', 0)}, "
            f"reused={details.get('reused_observation_point_count', 0)}, "
            f"scan_skipped={details.get('scan_skipped', False)}"
        )

    return lines


def _frontier_status_lines(details):
    """展示选点分数与本次决策关联的节点信息。"""
    lines = []
    candidates = details.get("frontier_candidates", ())
    if "frontier_selection_source" in details:
        lines.append(
            f"frontier choice: source={details['frontier_selection_source']}, "
            f"new={details['new_frontier_count']}, "
            f"deferred={details['deferred_frontier_count']}"
        )
    selected_id = details.get("candidate_id")
    if selected_id is not None:
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["candidate_id"] == selected_id
            ),
            None,
        )
        if selected is not None:
            semantic_score = selected["semantic_score"]
            semantic_text = (
                "none"
                if semantic_score is None
                else f"{float(semantic_score):.2f}"
            )
            lines.append(
                "selected frontier: "
                f"{selected_id}, rank=1/{details.get('candidate_count')}, "
                f"path={float(selected['path_distance_m']):.2f} m, "
                f"span={float(selected['frontier_span_m']):.2f} m, "
                f"vlm={semantic_text}, "
                f"semantic bonus={float(selected['semantic_bonus']):.2f}, "
                f"deferred order={selected.get('deferred_order')}, "
                f"score={float(selected['score']):.2f}"
            )

    context = [
        f"{key}={details[key]}"
        for key in ("node_id", "parent_node_id", "direction_id", "reason")
        if details.get(key) is not None
    ]
    if context:
        lines.append("context: " + ", ".join(context))

    return lines


def _observation_status_lines(frame, observation):
    """展示目标检测与掩码尺寸，不参与定位决策。"""
    lines = []
    if observation is not None:
        lines.append(f"visibility: {observation.visibility.value}")
        if observation.source:
            confidence = (
                "none"
                if observation.confidence is None
                else f"{observation.confidence:.3f}"
            )
            lines.append(
                f"detector: source={observation.source}, confidence={confidence}"
            )
        if observation.target_mask is not None:
            mask = target_mask_to_numpy(observation.target_mask)
            if mask is None:
                lines.append("sam2 mask: invalid")
            elif (
                frame.rgb is not None
                and len(frame.rgb) > 0
                and len(frame.rgb[0]) > 0
                and mask.shape != (len(frame.rgb), len(frame.rgb[0]))
            ):
                lines.append(
                    "sam2 mask: size mismatch, "
                    f"mask={mask.shape[1]}x{mask.shape[0]}, "
                    f"rgb={len(frame.rgb[0])}x{len(frame.rgb)}"
                )
            else:
                lines.append(
                    "sam2 mask: magenta overlay, "
                    f"size={mask.shape[1]}x{mask.shape[0]}, "
                    f"foreground={int(np.count_nonzero(mask))} px"
                )
        if observation.reason:
            lines.append(f"observation: {observation.reason}")

    return lines


def _history_status_lines(result):
    """统计探索历史并显示最近的执行异常。"""
    lines = []
    details = result.debug.details
    direction_counts = {state.value: 0 for state in SearchDirectionState}
    for node in result.state.observation_history:
        for direction in node.directions:
            direction_counts[direction.state.value] += 1
    lines.append(
        "history: "
        f"nodes={len(result.state.observation_history)}, "
        f"pending={direction_counts['pending']}, "
        f"committed={direction_counts['committed']}, "
        f"explored={direction_counts['explored']}, "
        f"invalidated={direction_counts['invalidated']}, "
        f"stalled={direction_counts['stalled']}"
    )
    lines.append(
        f"regions={len(result.state.frontier_regions)}, "
        f"blocked={len(result.state.blocked_frontier_regions)}, "
        f"deferred={sum(region.deferred_order is not None for region in result.state.frontier_regions)}, "
        f"active={result.state.active_frontier_id}, "
        f"checked_views={len(result.state.observed_views)}, "
        f"planned_frontier_points={len(result.state.scan_observation_points)}"
    )
    if result.state.observed_views:
        view = result.state.observed_views[-1]
        lines.append(
            f"last checked view: visible_points={len(view.visible_world_xy)}, "
            f"depth_coverage={view.depth_coverage_available}"
        )
    latest_issue = next(
        (
            (node.node_id, direction)
            for node in reversed(result.state.observation_history)
            for direction in node.directions
            if direction.execution_reason
        ),
        None,
    )
    if latest_issue is not None:
        node_id, direction = latest_issue
        lines.append(
            f"last exploration issue: {node_id}, {direction.state.value}, "
            f"{direction.execution_reason}"
        )
    if "destination_world_xy" in details:
        lines.append(f"exploration destination={details['destination_world_xy']}")
    return lines


def _vlm_request_metadata(interaction: VlmInteraction) -> str:
    """构造不含凭据、但覆盖全部模型请求参数的可读文本。"""
    reasoning = interaction.reasoning_effort or "none"
    return "\n".join(
        (
            f"interaction_id: {interaction.interaction_id}",
            f"task: {interaction.task}",
            f"endpoint: {interaction.endpoint_url}",
            f"model: {interaction.model}",
            f"api_format: {interaction.api_format}",
            f"reasoning_effort: {reasoning}",
            f"max_output_tokens: {interaction.max_output_tokens}",
            f"request_elapsed_s: {interaction.elapsed_s if interaction.elapsed_s is not None else 'in flight'}",
            (
                "image: "
                f"{interaction.image.width_px}x"
                f"{interaction.image.height_px} RGB"
            ),
        )
    )


def _vlm_card_sections(
    interaction: VlmInteraction,
) -> Tuple[Tuple[str, str], ...]:
    """按请求顺序组织卡片文本，不丢弃模型原始回应。"""
    sections = [
        ("来源与编号对应", context_text(interaction.context)),
        ("请求参数", _vlm_request_metadata(interaction)),
        ("完整输入提示词", interaction.prompt or "<empty>"),
    ]
    if interaction.phase == "request":
        sections.append(("模型输出", "等待模型返回……"))
        return tuple(sections)

    sections.extend(
        (
            ("Assistant 输出", interaction.assistant_text or "<empty>"),
            (
                "完整 HTTP JSON",
                interaction.response_json or "<no response>",
            ),
            ("解析结果", interaction.parsed_result or "<not parsed>"),
            ("错误", interaction.error or "<none>"),
        )
    )
    return tuple(sections)


def _vlm_card_status(interaction: VlmInteraction) -> str:
    if interaction.phase == "request":
        return "等待模型返回"
    if interaction.error:
        return "已返回（含请求或解析错误，查看有效结果）"
    return "已完成"


def _vlm_input_pil_image(font: object, interaction: VlmInteraction):
    """还原模型实际输入 RGB，并可选叠加解析成功的目标框。"""
    from PIL import Image, ImageDraw

    packed = interaction.image
    rgb = np.frombuffer(packed.rgb_bytes, dtype=np.uint8).reshape(
        packed.height_px,
        packed.width_px,
        3,
    )
    image = Image.fromarray(rgb.copy(), mode="RGB")
    if interaction.bbox_norm is None:
        return image

    x_min, y_min, x_max, y_max = interaction.bbox_norm
    left = max(0, min(image.width - 1, round(x_min * image.width)))
    top = max(0, min(image.height - 1, round(y_min * image.height)))
    right = max(0, min(image.width - 1, round(x_max * image.width)))
    bottom = max(0, min(image.height - 1, round(y_max * image.height)))
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.width, image.height) / 120))
    draw.rectangle(
        (left, top, right, bottom),
        outline=BBOX_RGB,
        width=line_width,
    )
    label = (
        "YOLO-World 候选"
        if interaction.task == "target_confirmation"
        else "VLM 目标定位"
    )
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 8
    label_height = label_box[3] - label_box[1] + 6
    label_left = min(left, max(0, image.width - label_width))
    label_top = max(0, top - label_height)
    draw.rectangle(
        (
            label_left,
            label_top,
            label_left + label_width,
            label_top + label_height,
        ),
        fill=(15, 45, 30),
    )
    draw.text(
        (label_left + 4, label_top + 2),
        label,
        font=font,
        fill=BBOX_RGB,
    )
    return image


def _wrap_multiline_text(
    font: object,
    text: str,
    max_pixels: float,
) -> List[str]:
    """保留原始换行，再按卡片宽度折行。"""
    wrapped: List[str] = []
    for source_line in text.expandtabs(4).split("\n"):
        line_parts = _wrap_text(font, source_line, max_pixels)
        wrapped.extend(line_parts if line_parts else [""])
    return wrapped


def _scan_basis_text(mode: str) -> str:
    """说明本次观察由 Frontier、首次环扫还是当前位置场景确认触发。"""
    basis = {
        "initial": "initial 360 deg sweep",
        "frontier": "unchecked local Frontier directions",
        "current_view": "target check at current pose (no extra turn)",
    }
    return f"scan basis: {basis.get(mode, mode)}"


def _wrap_text(font: object, text: str, max_pixels: float) -> List[str]:
    """按像素宽度把文本折成多行，不打断单个字符。"""
    lines: List[str] = []
    current = ""
    for char in text:
        if font.getlength(current + char) <= max_pixels:
            current += char
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines
