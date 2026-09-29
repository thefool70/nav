"""导航启动时预热本机 Qwen；请求后仅保留 5 分钟，不发送保活心跳。"""

import base64
import json
import struct
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zlib


def warmup_local_qwen(endpoint_url: str, model: str, api_format: str) -> None:
    """仅预热当前支持的本机 Ollama 配置；在设备创建前完成，不产生导航线索。"""
    address = urlsplit(endpoint_url.strip())
    if (model.strip().lower() != "qwen3.5:4b" or api_format != "chat_completions"
            or address.hostname not in ("localhost", "127.0.0.1", "::1")
            or address.path != "/v1/chat/completions"):
        return
    print("预热本机 Qwen 视觉模型；连续 5 分钟无请求后自动卸载。", flush=True)
    # 使用固定黑图，不读取相机、不接触导航队列；不仅加载权重，也初始化视觉计算。
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 848, 480, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(480 * (1 + 848 * 3))))
           + chunk(b"IEND", b""))
    endpoint = address._replace(path="", query="", fragment="").geturl()
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(endpoint + "/api/version", timeout=2) as response:
                response.read()
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("本机 Ollama 服务尚未就绪，请先启动 nav-vlm.service。")
            time.sleep(0.5)

    payload = {
        "model": "qwen3.5:4b", "stream": False, "think": False, "keep_alive": "5m",
        "format": "json", "options": {"num_predict": 32, "temperature": 0},
        "messages": [{"role": "user", "content": 'Return {"score":0.5,"bbox_2d":null}.',
                      "images": [base64.b64encode(png).decode("ascii")]}],
    }
    request = Request(endpoint + "/api/chat", json.dumps(payload).encode("utf-8"),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"本机 Qwen 视觉预热失败：{exc}") from exc
    if not result.get("done") or result.get("done_reason") != "stop":
        raise RuntimeError("Qwen 视觉预热未正常完成")
    print(f"Qwen visual warmup complete: {result['total_duration'] / 1e9:.2f}s", flush=True)
