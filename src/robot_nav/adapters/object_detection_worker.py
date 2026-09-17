"""常驻 YOLO 或 SAM2 子进程；标准输出传协议，日志保存阶段与卡顿时的调用栈。"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import faulthandler
import gzip
import json
from pathlib import Path
import sys
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolo", "sam2"), required=True)
    kind = parser.parse_args().model
    model = None
    encoded_image_file = None
    faulthandler.enable(file=sys.stderr)
    for line in sys.stdin:
        request = json.loads(line)
        folder = Path(request["folder"])
        started = time.monotonic()

        def progress(stage):
            event = {"event": "progress", "stage": stage, "elapsed_s": time.monotonic() - started}
            _send(event)
            with (folder / f"{kind}.progress.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(f"{folder} {kind}: {stage}", file=sys.stderr, flush=True)

        # 长时间未返回时保留 Python 栈，以区分加载、图像编码和掩码推理。
        faulthandler.dump_traceback_later(20.0, repeat=True, file=sys.stderr)
        try:
            with redirect_stdout(sys.stderr):
                progress("importing")
                import numpy as np
                with gzip.open(request["rgb_file"], "rb") as stream:
                    rgb = np.frombuffer(stream.read(), dtype=np.uint8).reshape(
                        request["height_px"], request["width_px"], 3,
                    ).copy()
                if model is None:
                    model = _load_model(kind, request, progress)
                    progress("ready")
                if kind == "yolo":
                    progress("detecting")
                    observations, count = model.detect_boxes(rgb)
                    result = {"observations": [{
                        "bbox_norm": item.bbox_norm, "confidence": item.confidence,
                    } for item in observations], "candidate_count": count}
                else:
                    if request["rgb_file"] != encoded_image_file:
                        progress("encoding_image")
                        model.set_image(rgb)
                        encoded_image_file = request["rgb_file"]
                    else:
                        progress("using_cached_image")
                    progress("segmenting")
                    mask = model.segment_box(tuple(request["bbox_norm"]))
                    mask_file = None
                    if mask is not None:
                        mask_file = "mask.json.gz"
                        with gzip.open(folder / mask_file, "wt", encoding="utf-8") as stream:
                            json.dump(np.asarray(mask, dtype=bool).tolist(), stream)
                    result = {"mask_file": mask_file}
                progress("finished")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(exc) or type(exc).__name__}
        finally:
            faulthandler.cancel_dump_traceback_later()
        (folder / f"{kind}.result.json").write_text(json.dumps(result, ensure_ascii=False))
        _send({"event": "result", "result": result})
    return 0


def _load_model(kind, request, progress):
    if kind == "yolo":
        from .yolo_world_sam2 import YoloWorldSam2Config, YoloWorldDetector
        return YoloWorldDetector(YoloWorldSam2Config(
            class_text=request["class_text"], model_path=Path(request["yolo_model"]),
            device=request["device"],
        ), on_stage=progress)
    from .sam2_observer import Sam2BoxSegmenter, Sam2ObserverConfig
    progress("loading_sam2")
    return Sam2BoxSegmenter(Sam2ObserverConfig(
        checkpoint_path=Path(request["sam2_checkpoint"]), device=request["device"],
    ))


def _send(message):
    sys.__stdout__.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    raise SystemExit(main())
