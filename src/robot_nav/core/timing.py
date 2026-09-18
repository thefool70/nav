"""显式传递的阶段计时；仅收集诊断，不参与算法决策或文件写入。"""

from contextlib import contextmanager
from time import monotonic
from typing import Iterator, MutableSequence, Optional


TimingSpans = MutableSequence[dict]


@contextmanager
def measure_stage(spans: Optional[TimingSpans], stage: str) -> Iterator[None]:
    """记录单调时钟起止与秒数；保留重复调用，嵌套阶段不能直接相加。"""
    if spans is None:
        yield
        return
    started = monotonic()
    completed = False
    try:
        yield
        completed = True
    finally:
        ended = monotonic()
        spans.append({
            "stage": stage,
            "started_monotonic_s": started,
            "ended_monotonic_s": ended,
            "duration_s": ended - started,
            "completed": completed,
        })
