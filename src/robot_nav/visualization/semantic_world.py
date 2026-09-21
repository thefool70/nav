"""World 中持久保存拍摄节点：状态、固定 RGB、评分和导航使用情况始终关联同一节点。"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

from .vlm_trace import frontier_node_path, interaction_node_path, interaction_result, view_node_path
from .observation_card import render_observation_card, score_lines


NODE_COLORS = {
    "queued": (240, 165, 45), "running": (185, 100, 245),
    "checked": (100, 160, 215), "target": (60, 220, 110),
    "localizing": (185, 100, 245),
    "localized": (50, 220, 175), "approaching": (50, 220, 175),
    "returning": (240, 165, 45), "complete": (60, 220, 110),
    "failed": (245, 80, 80), "stopped": (135, 135, 145),
}

NODE_LEGEND = "Orange: queued | Purple: inference | Green: target | Blue: checked | Red: failed"
RECENT_FINISHED_JOBS = 3
JOB_SPACING_M = 0.45


class SemanticWorldNodes:
    """由 RerunVisualizer 持锁调用；只记录数据，文件读取仅用于已保存的 RGB。"""

    def __init__(self, rr, log):
        self._rr, self._log = rr, log
        self._nodes = {}
        self._jobs = {}
        self._candidates = {}
        self._clues = {}
        self._local_paths = {}
        self._active_local_path = None
        self._map_id = None
        self._dirty_groups = set()
        self._cards = {}
        self._view_cards = {}
        self._focus = None

    def set_map(self, map_id):
        if map_id == self._map_id:
            return
        if self._map_id is not None:
            self._log("world/observations", self._rr.Clear(recursive=True))
            self._log("world/jobs", self._rr.Clear(recursive=True))
            self._log("observations/focus", self._rr.Clear(recursive=True))
        self._map_id = map_id
        for node in self._nodes.values():
            node["image_logged"] = False
            self._render(node)

    def record_queue_event(self, event):
        name = event["event"]
        job_id = event.get("job_id")
        if name == "queued":
            self._register_job(event)
        elif name == "started":
            self._set_job_state(job_id, "running")
            if self._active_local_path is None:
                self._focus = self._job_path(job_id)
        elif name == "completed":
            self._complete_job(job_id, event)
        elif name == "stopped":
            for stopped in event["job_ids"]:
                self._set_job_state(stopped, "stopped")
        elif name == "object_localization_started":
            path = interaction_node_path(event, event.get("view_id", 1))
            if path not in self._nodes:
                return
            self._local_paths[event["localization_directory"]] = path
            self._active_local_path = path
            self._focus = path
            node = self._nodes[path]
            node.update(state="localizing", progress="starting", reason="")
            self._render(node)
        elif name == "object_progress":
            path = self._local_paths.get(event["localization_directory"])
            if path in self._nodes:
                node = self._nodes[path]
                node["progress"] = f"{event['model']}: {event['stage']} ({event['elapsed_s']:.1f}s)"
                self._render(node)
        elif name == "object_localized":
            path = self._local_paths.get(event["localization_directory"])
            if path in self._nodes:
                node = self._nodes[path]
                node.update(state="localized" if event.get("target_world_xy") is not None else "failed",
                            reason=event.get("reason", ""), progress="finished",
                            target_position=str(event.get("target_world_xy")),
                            detector=event.get("target_source", ""))
                self._render(node)

    def record_interaction(self, interaction):
        context = interaction.context
        job_id = context.get("job_id")
        if interaction.task == "object_localization":
            path = interaction_node_path(context, 1)
            if path in self._nodes:
                self._nodes[path]["localization_request_id"] = interaction.interaction_id
                self._render(self._nodes[path])
            return
        if interaction.task != "semantic_analysis" or job_id not in self._jobs:
            return
        for path in self._jobs[job_id]:
            self._nodes[path]["request_id"] = interaction.interaction_id
        for marker in context.get("markers", ()):
            view_path = view_node_path(job_id, marker["view_id"])
            if view_path not in self._nodes:
                continue
            view = self._nodes[view_path]
            candidate = self._candidates[job_id].get(marker["candidate_id"], {})
            score = {"label": marker["label"], "region": marker["region_id"], "value": None,
                     "pixel_xy": marker.get("source_pixel_xy"), "frontier_score": candidate.get("score")}
            view["scores"].setdefault(marker["candidate_id"], score)
            path = frontier_node_path(job_id, marker["label"])
            if path not in self._nodes:
                self._register(path, {
                    **view, "label": f"J{job_id}/{marker['label']}", "kind": "frontier",
                    "point_xy": (marker["world_xy"][0], -marker["world_xy"][1]),
                    "candidate_id": marker["candidate_id"], "region": marker["region_id"],
                    "scores": {marker["candidate_id"]: dict(score)},
                })
                self._jobs[job_id].append(path)
        if interaction.phase == "response":
            self._complete_job(job_id, {**interaction_result(interaction), "detection_error": interaction.error})
        else:
            self._set_job_state(job_id, "running")

    def record_cycle(self, result):
        for item in result.debug.details.get("semantic_received_jobs", ()):
            for path in self._jobs.get(item["job_id"], ()):
                self._nodes[path]["navigation"] = "received"
                self._render(self._nodes[path])
        for item in result.debug.details.get("semantic_score_sources", ()):
            for path in self._jobs.get(item["job_id"], ()):
                node = self._nodes[path]
                if any(value["region"] == item["candidate_id"] for value in node["scores"].values()):
                    node["navigation"] = "score used for ranking"
                    self._render(node)
        clue = result.state.active_target_clue
        failed = result.debug.details.get("failed_clue_id")
        if failed:
            self._set_node_state(self._clues.get(failed), "failed", result.debug.details.get("reason", ""))
        if result.state.phase.value == "complete":
            self._set_node_state(self._clues.get(result.debug.details.get("clue_id")), "complete")
        if result.state.phase.value == "stopped":
            fallback = result.state.object_approach.fallback_clue
            if fallback is not None:
                self._set_node_state(self._clues.get(fallback.clue_id), "stopped", result.debug.message)
        if clue is not None and clue.job_id is not None and clue.view_id is not None:
            state = "returning" if "return" in result.debug.stage or "turn" in result.debug.stage else "approaching"
            if result.action is not None:
                self._set_node_state(view_node_path(clue.job_id, clue.view_id), state)
        if self._active_local_path is not None:
            if result.state.phase.value == "complete":
                self._set_node_state(self._active_local_path, "complete")
            elif failed:
                self._set_node_state(self._active_local_path, "failed", result.debug.details.get("reason", ""))
            self._active_local_path = None

    def _register_job(self, event):
        folder = Path(event["snapshot"])
        metadata = json.loads((folder / "snapshot.json").read_text())
        job_id = event["job_id"]
        self._jobs[job_id] = []
        if self._focus is None:
            self._focus = self._job_path(job_id)
        self._candidates[job_id] = {item["candidate_id"]: item for item in metadata["candidates"]}
        for index, view in enumerate(metadata["views"], 1):
            coverage = view["coverage"]
            path = view_node_path(job_id, index)
            self._register(path, {
                "label": f"J{job_id}/V{index}", "kind": "view", "job_id": job_id, "view_id": index,
                "pose": coverage["pose"], "heading": coverage["camera_heading_world_rad"],
                "map_id": view["map_frame_id"], "timestamp_s": coverage["timestamp_s"],
                "rgb_file": folder / f"view-{index}.rgb.gz", "width": view["width_px"],
                "height": view["height_px"], "source": metadata["source"],
            })
            self._jobs[job_id].append(path)
            self._clues[f"semantic:{job_id}:{index}"] = path

    def _register(self, path, values):
        node = {"state": "queued", "scores": {}, "reason": "", "navigation": "pending",
                "request_id": 0, "localization_request_id": 0, "progress": "", "target_position": "", "detector": "", **values,
                "path": path, "image_logged": False}
        if "point_xy" not in node:
            pose, heading = node["pose"], node["heading"]
            # 视角节点放在朝向线上，区分同一拍摄位置的多张转向图；详情仍保留真实位姿。
            node["point_xy"] = (pose["x_m"] + 0.30 * math.cos(heading), -pose["y_m"] - 0.30 * math.sin(heading))
        self._nodes[path] = node
        self._render(node)

    def _complete_job(self, job_id, result):
        for path in self._jobs.get(job_id, ()):
            node = self._nodes[path]
            scores = result.get("frontier_scores", {})
            for candidate, value in node["scores"].items():
                value["value"] = scores.get(candidate)
            views = result.get("target_view_ids")
            if node["kind"] == "frontier":
                node["state"] = "checked" if node["candidate_id"] in scores else "failed"
                node["reason"] = result.get("scoring_error", "") or ("missing score" if node["state"] == "failed" else "")
            else:
                node["state"] = "failed" if views is None else "target" if node["view_id"] in views else "checked"
                node["clue_order"] = views.index(node["view_id"]) + 1 if views and node["view_id"] in views else 0
                node["reason"] = result.get("detection_error", "")
            self._render(node)

    def _set_job_state(self, job_id, state):
        for path in self._jobs.get(job_id, ()):
            self._set_node_state(path, state)

    def _set_node_state(self, path, state, reason=""):
        if path in self._nodes:
            self._nodes[path].update(state=state, reason=reason)
            self._render(self._nodes[path])

    def _render(self, node):
        self._dirty_groups.add(self._group_path(node))
        if node["map_id"] != self._map_id:
            return
        color = NODE_COLORS[node["state"]]
        missing = "pending" if node["state"] in ("queued", "running") else "missing"
        scores = "; ".join(
            f"{item['region']}: VLM={missing if item['value'] is None else format(item['value'], '.2f')}"
            + (f", frontier={item['frontier_score']:.2f}" if item.get("frontier_score") is not None else "")
            for item in node["scores"].values()
        ) or "no projected Frontier"
        pose = node["pose"]
        self._log(node["path"], self._rr.Points2D(
            [node["point_xy"]], colors=[color], radii=0.075 if node["kind"] == "view" else 0.04,
            labels=[node["label"]], show_labels=node["state"] in ("running", "localizing"),
        ), self._rr.AnyValues(
            status=node["state"], scores=scores, navigation=node["navigation"],
            capture=f"({pose['x_m']:.2f}, {pose['y_m']:.2f}) m / {math.degrees(pose['yaw_rad']):.1f} deg",
            capture_time_s=node["timestamp_s"], source=node["source"],
            scoring_request=f"R{node['request_id']}" if node["request_id"] else "none",
            localization_request=f"R{node['localization_request_id']}" if node["localization_request_id"] else "none",
            progress=node["progress"], reason=node["reason"], target_position=node["target_position"],
            detector=node["detector"],
        ))
        if node["kind"] == "view":
            start = (pose["x_m"], -pose["y_m"])
            self._log(node["path"] + "/heading", self._rr.Arrows2D(
                origins=[start], vectors=[(node["point_xy"][0] - start[0], node["point_xy"][1] - start[1])],
                colors=[color], radii=0.009, show_labels=False,
            ))
        if not node["image_logged"]:
            with gzip.open(node["rgb_file"], "rb") as stream:
                rgb = np.frombuffer(stream.read(), dtype=np.uint8).reshape(node["height"], node["width"], 3)
            image = self._rr.Image(rgb)
            # 原图单独归档；V 节点的预览在批量刷新时更新为单图评分卡。
            self._log(node["path"] + "/rgb", [image.buffer, image.format])
            self._log(node["path"], [image.buffer, image.format])
            node["image_logged"] = True

    @staticmethod
    def _job_path(job_id):
        return f"world/jobs/J{job_id:06d}"

    def _group_path(self, node):
        return self._job_path(node["job_id"])

    def refresh_panels(self, font):
        """每个回调结束后统一更新任务标记和卡片，不在单个视角更新中反复重画。"""
        if not self._dirty_groups:
            return
        groups = {}
        for node in self._nodes.values():
            if node["kind"] == "view" and node["map_id"] == self._map_id:
                groups.setdefault(self._group_path(node), []).append(node)
        for path in self._dirty_groups:
            if path not in groups:
                continue
            nodes = groups[path]
            card = self._rr.Image(render_observation_card(self._group_label(nodes), nodes, NODE_COLORS, font))
            self._cards[path] = card
            # 任务点预览整组，V 点预览单图评分卡；无标注原图位于 V/rgb。
            self._log(path, [card.buffer, card.format])
            for node in nodes:
                view_card = self._rr.Image(render_observation_card(
                    f"{node['label']} / {len(nodes)} views in this capture", [node], NODE_COLORS, font,
                ))
                self._view_cards[node["path"]] = view_card
                self._log(node["path"], [view_card.buffer, view_card.format])
        self._render_job_markers(groups)
        focused = None
        if self._focus in self._nodes:
            node = self._nodes[self._focus]
            if node["map_id"] == self._map_id:
                focused = node
        elif self._focus in groups:
            # 固定面板只放大一张图，避免八视角拼图缩小后又看不清。
            # 优先第一条目标线索，否则展示语义分最高的视角，未返回时用第一张。
            focused = min(groups[self._focus], key=lambda node: (
                0 if node.get("clue_order") else 1,
                node.get("clue_order", 0),
                -max((score["value"] or 0 for score in node["scores"].values()), default=0),
                node["view_id"],
            ))
        if focused is not None:
            self._log("observations/focus", self._view_cards[focused["path"]])
        self._log("observations/index", self._rr.TextDocument(self._index_text(groups), media_type="text/markdown"))
        self._dirty_groups.clear()

    @staticmethod
    def _group_label(nodes):
        first = nodes[0]
        return f"J{first['job_id']} / {first['source']} / {len(nodes)} views"

    @staticmethod
    def _group_state(nodes):
        # 正在处理与有效线索优先于同组中未命中的视角。
        for state in ("localizing", "approaching", "returning", "running",
                      "complete", "localized", "target", "queued", "failed", "stopped", "checked"):
            if any(node["state"] == state for node in nodes):
                return state

    def _render_job_markers(self, groups):
        """主图每个拍摄任务至多一个点；邻近任务共用位置，完整条目保留在索引中。"""
        active_states = {"queued", "running", "localizing", "target", "localized",
                         "approaching", "returning", "complete"}
        states = {path: self._group_state(nodes) for path, nodes in groups.items()}
        recent = [path for path in groups if states[path] not in active_states][-RECENT_FINISHED_JOBS:]
        focus_group = self._group_path(self._nodes[self._focus]) if self._focus in self._nodes else self._focus
        candidates = [path for path in groups if states[path] in active_states or path in recent or path == focus_group]
        candidates.sort(key=lambda path: (path != focus_group, states[path] not in active_states, states[path] == "queued"))
        shown = {}
        nearby = {}
        for path in candidates:
            pose = groups[path][0]["pose"]
            xy = (pose["x_m"], -pose["y_m"])
            owner = next((other for other, pos in shown.items() if math.dist(pos, xy) < JOB_SPACING_M), None)
            if owner is None:
                shown[path] = xy
            else:
                nearby.setdefault(owner, []).append(path)
        for path, nodes in groups.items():
            if path not in shown:
                # 仅清空点位置，保留节点的卡片供历史索引直接选择。
                self._log(path, self._rr.Points2D.from_fields(positions=[]))
                continue
            state = states[path]
            label = f"J{nodes[0]['job_id']}"
            if nearby.get(path):
                label += f" +{len(nearby[path])}"
            focus = path == focus_group
            radius = 8 if focus else 5 if state in active_states else 3
            self._log(path, self._rr.Points2D(
                [shown[path]], colors=[NODE_COLORS[state]], radii=self._rr.Radius.ui_points(radius),
                labels=[label], show_labels=focus or bool(nearby.get(path)), draw_order=40,
            ), self._rr.AnyValues(
                status=state,
                scores="; ".join(f"V{node['view_id']}: {', '.join(score_lines(node))}" for node in nodes),
                nearby_jobs=", ".join(self._group_label(groups[item]) for item in nearby.get(path, ())) or "none",
                inspect="Select this task for RGB + scores; use Observations for individual V images and nearby jobs.",
            ))

    def _index_text(self, groups):
        queued = sum(self._group_state(nodes) == "queued" for nodes in groups.values())
        lines = [f"**{len(groups)} captures | {queued} queued**", "",
                 "J: all views / V: RGB + scores / raw: original", "",
                 "| Capture | State | Images / raw RGB |", "| --- | --- | --- |"]
        focus_group = self._group_path(self._nodes[self._focus]) if self._focus in self._nodes else self._focus
        rows = list(groups.items())
        # 当前输入置顶，接着按 FIFO 显示待处理任务，历史按拍摄时间倒序。
        rows.sort(key=lambda item: (
            0 if item[0] == focus_group else 1 if self._group_state(item[1]) == "queued" else 2,
            item[1][0]["timestamp_s"] * (1 if self._group_state(item[1]) == "queued" else -1),
        ))
        for path, nodes in rows:
            label = f"J{nodes[0]['job_id']}"
            views = " / ".join(f"[V{node['view_id']}](recording://{node['path']}) "
                               f"[raw](recording://{node['path']}/rgb)" for node in nodes)
            state = self._group_state(nodes)
            if path == focus_group:
                state = f"**{state}**"
            lines.append(f"| [{label}](recording://{path}) | {state} | {views} |")
        lines.extend(["", NODE_LEGEND, "", "World +N: nearby captures. Click a link to preview in Selection."])
        return "\n".join(lines)
