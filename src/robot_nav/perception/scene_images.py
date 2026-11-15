"""场景尾处理的网格合图；原始检测和定位始终使用各方向的同帧 RGB-D。"""

from __future__ import annotations

import math

import numpy as np

from robot_nav.perception.vlm_analysis import VlmInputImage


def scene_contact_sheet(views) -> VlmInputImage:
    """按采集顺序排列带编号缩略图，不依赖全景拼接或额外图像库。"""
    columns = math.ceil(math.sqrt(len(views)))
    rows = math.ceil(len(views) / columns)
    width = min(640, max(view.image.width_px for view in views))
    height = max(round(width * view.image.height_px / view.image.width_px) for view in views)
    header = 28
    canvas = np.zeros((rows * (height + header), columns * width, 3), dtype=np.uint8)
    # 3×5 数字点阵，仅用于原图外的方向编号，不改变送检图坐标。
    digits = ("111101101101111", "010110010010111", "111001111100111",
              "111001111001111", "101101111001001", "111100111001111",
              "111100111101111", "111001001001001", "111101111101111", "111101111001111")
    for index, view in enumerate(views):
        image = view.image
        rgb = np.frombuffer(image.rgb_bytes, dtype=np.uint8).reshape(image.height_px, image.width_px, 3)
        tile_height = round(width * image.height_px / image.width_px)
        y = np.minimum(np.arange(tile_height) * image.height_px // tile_height, image.height_px - 1)
        x = np.minimum(np.arange(width) * image.width_px // width, image.width_px - 1)
        top, left = (index // columns) * (height + header), (index % columns) * width
        canvas[top + header:top + header + tile_height, left:left + width] = rgb[y[:, None], x]
        for offset, digit in enumerate(str(index + 1)):
            glyph = np.array([int(bit) for bit in digits[int(digit)]], dtype=np.uint8).reshape(5, 3)
            glyph = np.repeat(np.repeat(glyph, 4, axis=0), 4, axis=1) * 255
            canvas[top + 4:top + 24, left + 8 + offset * 16:left + 20 + offset * 16] = glyph[:, :, None]
    return VlmInputImage(canvas.shape[1], canvas.shape[0], canvas.tobytes())
