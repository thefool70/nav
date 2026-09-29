"""常驻 YOLOE 子进程；标准输出传协议，日志保存阶段与卡顿时的调用栈。"""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main() -> int:
    """持续接收单行 JSON 请求；模型首次请求时加载，此后复用并返回阶段事件和结果。"""
    model = None
    faulthandler.enable(file=sys.stderr)
    for line in sys.stdin.buffer:
        request = json.loads(line)
        rgb_bytes = sys.stdin.buffer.read(request["rgb_length"]) if "rgb_length" in request else None
        if rgb_bytes is not None and len(rgb_bytes) != request["rgb_length"]:
            raise EOFError("视频 RGB 管道提前结束")
        started = time.monotonic()

        def progress(stage):
            if not request.get("initialize"):
                return
            event = {"event": "progress", "stage": stage, "elapsed_s": time.monotonic() - started}
            _send(event)
            print(f"YOLOE: {stage}", file=sys.stderr, flush=True)

        # 长时间未返回时保留 Python 栈，以区分加载、图像编码和掩码推理。
        faulthandler.dump_traceback_later(20.0, repeat=True, file=sys.stderr)
        try:
            # 第三方模型常向 stdout 打印；重定向后，stdout 才能保持逐行 JSON 协议。
            with redirect_stdout(sys.stderr):
                progress("importing")
                import numpy as np
                if model is None:
                    model = _load_model(request, progress)
                    progress("ready")
                if request.get("initialize"):
                    _send({"event": "result", "result": {"ready": True}})
                    continue
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    request["height_px"], request["width_px"], 3)
                observations, count = model.detect_boxes(rgb)
                result = {"observations": [{
                    "bbox_norm": item.bbox_norm, "confidence": item.confidence,
                    "mask_bits": (base64.b64encode(np.packbits(item.target_mask).tobytes()).decode("ascii")
                                  if item.target_mask is not None else None),
                } for item in observations], "candidate_count": count}
                progress("finished")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(exc) or type(exc).__name__}
        finally:
            faulthandler.cancel_dump_traceback_later()
        _send({"event": "result", "result": result})
    return 0


def _load_model(request, progress):
    from .yoloe import YoloEConfig, YoloEDetector
    model_path = Path(request["yolo_model"]).resolve()
    # Ultralytics 在工作目录查找 mobileclip2_b.ts，避免切换运行目录时重下载。
    os.chdir(model_path.parent)
    return YoloEDetector(YoloEConfig(
        class_text=request["class_text"], model_path=model_path,
        device=request["device"], confidence_threshold=request["confidence_threshold"],
    ), on_stage=progress)


def _send(message):
    """协议写入原始 stdout，避免模型库的普通打印混入父进程消息。"""
    sys.__stdout__.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    raise SystemExit(main())
