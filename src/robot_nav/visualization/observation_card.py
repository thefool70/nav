"""将同一次任务的固定 RGB、状态和评分排在一起，供主面板和节点预览共用。"""

from __future__ import annotations

import gzip
import math

import numpy as np


def score_lines(node):
    """使用图片中的 F 编号，明确区分语义分和拍摄时的综合分。"""
    lines = []
    for score in node["scores"].values():
        value = "pending" if node["state"] in ("queued", "running") else "missing"
        if score["value"] is not None:
            value = f"{score['value']:.2f}"
        line = f"{score['label']}: VLM {value}"
        if score.get("frontier_score") is not None:
            line += f" / total {score['frontier_score']:.2f}"
        lines.append(line)
    return lines or ["No projected Frontier"]


def render_observation_card(title, nodes, colors, font):
    """一到两列排列原图；原始 RGB 另存于 V/rgb，缺少 Pillow 时只显示首张原图。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        node = nodes[0]
        with gzip.open(node["rgb_file"], "rb") as stream:
            return np.frombuffer(stream.read(), dtype=np.uint8).reshape(node["height"], node["width"], 3)

    font = font or ImageFont.load_default()
    pad, gap, cell_width, image_height, line_height = 12, 12, 360, 240, 23
    columns = min(2, len(nodes))
    rows = math.ceil(len(nodes) / columns)
    score_rows = max(len(score_lines(node)) for node in nodes)
    cell_height = image_height + (score_rows + 3) * line_height + 2 * pad
    width = 2 * pad + columns * cell_width + (columns - 1) * gap
    header_height = 2 * line_height + 2 * pad
    card = Image.new("RGB", (width, header_height + rows * (cell_height + gap)), (23, 27, 34))
    draw = ImageDraw.Draw(card)
    draw.text((pad, pad), title, font=font, fill=(239, 243, 250))
    draw.text((pad, pad + line_height), "VLM = semantic / total = score at capture", font=font, fill=(158, 170, 189))
    for index, node in enumerate(nodes):
        x = pad + (index % columns) * (cell_width + gap)
        y = header_height + (index // columns) * (cell_height + gap)
        color = colors[node["state"]]
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline=color, width=2)
        draw.text((x + pad, y + 6), f"{node['label']}  |  {node['state']}", font=font, fill=color)
        if "thumbnail" not in node:
            with gzip.open(node["rgb_file"], "rb") as stream:
                raw = Image.frombytes("RGB", (node["width"], node["height"]), stream.read())
            raw.thumbnail((cell_width - 2 * pad, image_height))
            node["thumbnail"] = raw
        rgb = node["thumbnail"]
        image_top = y + line_height + pad
        image_x = x + (cell_width - rgb.width) // 2
        image_y = image_top + (image_height - rgb.height) // 2
        card.paste(rgb, (image_x, image_y))
        for score in node["scores"].values():
            if score.get("pixel_xy") is None:
                continue
            u, v = score["pixel_xy"]
            u, v = image_x + u * rgb.width / node["width"], image_y + v * rgb.height / node["height"]
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 207, 75), outline=(15, 18, 24))
            draw.text((min(u + 5, image_x + rgb.width - 34), max(image_y, v - line_height)),
                      score["label"], font=font, fill=(255, 207, 75), stroke_width=2, stroke_fill=(15, 18, 24))
        line_y = image_top + image_height + 6
        pose = node["pose"]
        details = [f"({pose['x_m']:.2f}, {pose['y_m']:.2f}) m  /  {math.degrees(node['heading']):.0f} deg"]
        details.extend(score_lines(node))
        # 导航接收状态与模型状态分开，避免把有分数误读为本轮已采用。
        details.append(node["navigation"])
        for line in details:
            draw.text((x + pad, line_y), line, font=font, fill=(225, 231, 240))
            line_y += line_height
    return np.asarray(card)
