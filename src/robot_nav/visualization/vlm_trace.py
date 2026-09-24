"""VLM 简要视图的数据关联与文本；不保留 RGB，不影响导航决策。"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from ..adapters.perception import VlmInteraction
from ..core.models import NavigationResult


def request_path(request_id: int) -> str:
    return f"model/vlm/requests/R{request_id:06d}"


def job_path(job_id: int) -> str:
    return f"model/vlm/jobs/J{job_id:06d}"


def view_node_path(job_id: int, view_id: int) -> str:
    return f"world/observations/J{job_id:06d}/V{view_id}"


def frontier_node_path(job_id: int, label: str) -> str:
    return f"world/observations/J{job_id:06d}/{label}"


def interaction_node_path(context, view_id: int) -> str:
    job_id = context.get("job_id")
    return view_node_path(job_id, context.get("view_id", view_id)) if job_id is not None else "world/observations"


def interaction_result(interaction: VlmInteraction) -> Mapping[str, Any]:
    try:
        value = json.loads(interaction.parsed_result)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def result_summary(result: Mapping[str, Any]) -> str:
    if "target_view_ids" in result:
        view_ids = result["target_view_ids"]
        target = ("detected: " + " → ".join(f"V{view_id}" for view_id in view_ids)
                  if view_ids else "no target clues" if view_ids is not None else "detection failed")
        return f"{target}; {len(result.get('frontier_scores', {}))} scores"
    for name in ("confirmation", "scene", "visibility"):
        if name in result:
            return f"{name}: {result[name]}"
    if "bbox_norm" in result:
        return "target box located"
    return "no parsed result"


def context_text(context: Mapping[str, Any]) -> str:
    """完整卡片明确区分 J 任务、R 请求、V 画面、F 快照候选与世界区域。"""
    job = context.get("job_id")
    lines = [f"来源: {context.get('source', 'synchronous')}",
             f"FIFO 任务: {'J' + str(job) if job is not None else '现场直接请求'}"]
    if context.get("parent_request_id") is not None:
        lines.append(f"前置请求: R{context['parent_request_id']}")
    if context.get("clue_id"):
        lines.append(f"目标线索: {context['clue_id']}")
    for view in context.get("views", ()):
        pose = view["pose"]
        lines.append(
            f"V{view['view_id']}: t={view['timestamp_s']:.3f}s, "
            f"robot=({pose['x_m']:.2f}, {pose['y_m']:.2f})m, "
            f"yaw={pose['yaw_rad']:.3f}rad, map={view['map_frame_id']}"
        )
    for marker in context.get("markers", ()):
        x, y = marker["world_xy"]
        lines.append(f"{marker['label']} -> {marker['region_id']} @ ({x:.2f}, {y:.2f})m [{marker['candidate_id']}]")
        if marker.get("source_pixel_xy") is not None:
            u, v = marker["source_pixel_xy"]
            lines.append(f"  V{marker['view_id']} 原始 RGB 像素锚点=({u:.1f}, {v:.1f})，局部地面投影已核对深度。")
    if context.get("snapshot"):
        lines.append(f"快照目录: {context['snapshot']}")
    lines.append("拍摄位姿是输入来源；结果返回时机器人可能已移动。F 编号只在本请求内有效。")
    return "\n".join(lines)


class VlmTraceHistory:
    """记录轻量事件摘要；完整输入输出由 Rerun 按请求 ID 归档。"""

    def __init__(self) -> None:
        self.calls: Dict[int, Dict[str, Any]] = {}
        self.jobs: Dict[int, Dict[str, Any]] = {}
        self.latest_response: Optional[Mapping[str, Any]] = None
        self.cycle_index = 0
        self.current_usage = "No model result used by a navigation cycle yet."
        self.latest_warning = ""
        self.queue_stopped = False

    def record_interaction(self, interaction: VlmInteraction, frame_index: int) -> None:
        row = self.calls.setdefault(interaction.interaction_id, {})
        row.update(context=interaction.context, task=interaction.task, phase=interaction.phase,
                   elapsed_s=interaction.elapsed_s, error=interaction.error,
                   result=interaction_result(interaction), request_id=interaction.interaction_id)
        row[f"{interaction.phase}_frame"] = frame_index
        job_id = interaction.context.get("job_id")
        if job_id is not None:
            self.jobs.setdefault(job_id, {}).update(interaction_id=interaction.interaction_id)
        if interaction.phase == "response":
            self.latest_response = dict(row)

    def record_queue_event(self, event: Mapping[str, Any], frame_index: int) -> None:
        if event["event"].startswith("object_"):
            return
        if event["event"] == "stopped":
            self.queue_stopped = True
            for job_id in event["job_ids"]:
                self.jobs.setdefault(job_id, {}).update(stopped_frame=frame_index)
            return
        job_id = event.get("job_id")
        if job_id is None:
            self.latest_warning = str(event.get("reason", event.get("event", "")))
            return
        row = self.jobs.setdefault(job_id, {})
        row.update(event)
        row[f"{event['event']}_frame"] = frame_index
        if event["event"] in ("snapshot_failed", "result_write_failed"):
            self.latest_warning = str(event.get("reason", ""))

    def record_cycle(self, result: NavigationResult, frame_index: int) -> None:
        self.cycle_index += 1
        received = result.debug.details.get("semantic_received_jobs", ())
        sources = result.debug.details.get("semantic_score_sources", ())
        for item in received:
            self.jobs.setdefault(item["job_id"], {}).update(
                received_cycle=self.cycle_index, received_frame=frame_index, clue_ids=item["clue_ids"],
            )
        for item in sources:
            self.jobs.setdefault(item["job_id"], {}).update(ranked_cycle=self.cycle_index)
        parts = [f"C{self.cycle_index} / frame {frame_index}: {result.debug.stage}"]
        if result.debug.details.get("semantic_background_paused"):
            parts.append("Ordinary VLM queue paused.")
        if received:
            parts.append("Received: " + ", ".join(f"J{item['job_id']}" for item in received))
        if sources:
            parts.append("Ranking inputs: " + "; ".join(
                f"{item['candidate_id']}={item['score']:.2f} from J{item['job_id']}/"
                + (f"R{item['interaction_id']}" if item['interaction_id'] is not None else "random")
                for item in sources
            ))
        else:
            parts.append("No cached VLM scores supplied for ranking this cycle.")
        clue = result.state.active_target_clue
        if clue is not None:
            parts.append(f"Active clue: {clue.clue_id}")
        self.current_usage = "\n\n".join(parts)

    def overview(self) -> str:
        active = [key for key, row in self.calls.items() if row.get("phase") == "request"]
        queued = [key for key, row in self.jobs.items() if "queued_frame" in row
                  and "started_frame" not in row and "stopped_frame" not in row]
        waiting = sum("completed_frame" in row and "received_cycle" not in row for row in self.jobs.values())
        lines = ["# VLM summary", "",
                 "J = FIFO job; R = model request; V/F numbers belong to that request.", "",
                 f"**In flight:** {', '.join('R' + str(key) for key in active) or '-'} | "
                 f"**Queued:** {len(queued)} | **Returned, awaiting navigation:** {waiting}", "",
                 f"FIFO next: {', '.join('J' + str(key) for key in sorted(queued)[:5]) or '-'}", "",
                 self.current_usage, ""]
        if self.queue_stopped:
            lines.extend(["**Queue stopped. An in-flight HTTP request may still finish; pending snapshots remain on disk.**", ""])
        latest = self.latest_response
        if latest is not None:
            job_id = latest["context"].get("job_id")
            lines.extend([f"**Latest return: R{latest['request_id']}"
                          + (f" / J{job_id}" if job_id is not None else " / direct") + "**", "",
                          result_summary(latest["result"]), ""])
            views = latest["context"].get("views", ())
            if views:
                lines.extend(["Observations links open RGB + scores for each job; V links open a single-view card.",
                              "World shows grouped jobs; World history keeps every V/F marker.", ""])
                target_views = latest["result"].get("target_view_ids") or ()
                lines.extend(["| V | Clue order | Capture (x, y) m | Timestamp s |", "| --- | --- | --- | --- |"])
                for index, view in enumerate(views):
                    pose = view["pose"]
                    order = target_views.index(view["view_id"]) + 1 if view["view_id"] in target_views else "-"
                    path = interaction_node_path(latest["context"], view["view_id"])
                    lines.append(f"| [V{view['view_id']}](recording://{path}) | "
                                 f"{order} | ({pose['x_m']:.2f}, {pose['y_m']:.2f}) | {view['timestamp_s']:.3f} |")
                lines.append("")
            markers = latest["context"].get("markers", ())
            if markers:
                lines.extend(["| F | V | Region at capture | Score |", "| --- | --- | --- | --- |"])
                for index, marker in enumerate(markers):
                    score = latest["result"].get("frontier_scores", {}).get(marker["candidate_id"])
                    score_text = "missing" if score is None else f"{score:.2f}"
                    path = frontier_node_path(job_id, marker['label']) if job_id is not None else "world/observations"
                    lines.append(f"| [{marker['label']}](recording://{path}) | "
                                 f"{marker.get('view_id', '-')} | {marker['region_id']} | {score_text} |")
                lines.append("")
        lines.extend(["## Recent requests", "", "| R / J | Source | State / result | Time | Send / return frame |",
                      "| --- | --- | --- | --- | --- |"])
        for request_id in sorted(self.calls, reverse=True)[:10]:
            row = self.calls[request_id]
            context = row["context"]
            job_id = context.get("job_id")
            status = "in flight" if row["phase"] == "request" else result_summary(row["result"])
            if row.get("error"):
                status += "; error (see full)"
            elapsed = row.get("elapsed_s")
            duration = "-" if elapsed is None else f"{elapsed:.1f}s"
            identity = f"[R{request_id}](recording://{request_path(request_id)})"
            identity += f" / J{job_id}" if job_id is not None else " / direct"
            source = context.get('source', row['task'])
            if context.get("clue_id"):
                source += f" from {context['clue_id']}"
            if context.get("parent_request_id") is not None:
                source += f" after R{context['parent_request_id']}"
            lines.append(f"| {identity} | {source} | {status} | {duration} | "
                         f"{row.get('request_frame', '-')} / {row.get('response_frame', '-')} |")
        lines.extend(["", "## Recent FIFO jobs", "", "| J | Source / views | State | Navigation |",
                      "| --- | --- | --- | --- |"])
        for job_id in sorted(self.jobs, reverse=True)[:8]:
            row = self.jobs[job_id]
            status = "stopped" if "stopped_frame" in row else "failed" if "snapshot_failed_frame" in row else (
                "received" if "received_cycle" in row else "returned" if "completed_frame" in row
                else "running" if "started_frame" in row else "queued"
            )
            usage = f"received C{row['received_cycle']}" if "received_cycle" in row else "-"
            if "ranked_cycle" in row:
                usage += f"; ranked C{row['ranked_cycle']}"
            if row.get("clue_ids"):
                usage += "; clue created"
            lines.append(f"| [J{job_id}](recording://{job_path(job_id)}) | "
                         f"{row.get('source', '-')} / {row.get('view_count', '-')} | {status} | {usage} |")
        if self.latest_warning:
            lines.extend(["", "Latest queue issue: " + self.latest_warning.replace("\n", " ")])
        lines.extend(["", "Open VLM full for the current event. Set the frame cursor to a send/return frame to replay it.",
                      "R/J links select archived entities; older entries remain under model/vlm in the recording.",
                      "Received means cached/processed; ranked means supplied to that cycle, not necessarily selected."])
        return "\n".join(lines)
