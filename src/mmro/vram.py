"""VRAM measurement and OOM containment.

The sweep deliberately pushes each model until it OOMs, so an OOM must be a
*data point*, not a crash: :func:`track` catches it, records peak memory, and
lets the caller continue down the ladder.

torch is imported lazily so this module (and the CPU test suite) works in an
environment without it.
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Iterator


@dataclass
class MemoryTrace:
    peak_gb: float = 0.0
    seconds: float = 0.0
    oom: bool = False
    error: str | None = None


def _torch():
    import torch  # noqa: PLC0415 - lazy so CPU-only environments can import this

    return torch


def cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def reset_peak() -> None:
    if cuda_available():
        _torch().cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    if not cuda_available():
        return 0.0
    return _torch().cuda.max_memory_allocated() / (1024**3)


def free_total_gb() -> tuple[float, float]:
    if not cuda_available():
        return (0.0, 0.0)
    free, total = _torch().cuda.mem_get_info()
    return (free / (1024**3), total / (1024**3))


def empty_cache() -> None:
    """Best-effort VRAM release.

    Note this is *not* sufficient to swap models within one process on a 24GB
    L4 -- fragmentation reliably defeats it, which is why smoke and sweep run
    one model per subprocess. It is still worth doing between rungs.
    """
    gc.collect()
    if cuda_available():
        _torch().cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory condition.

    Matches on message text as well as type: OOM surfaces as several different
    exception classes depending on where it happens (allocator, cuBLAS
    workspace, cuDNN), and the sweep must classify all of them as OOM rather
    than as a hard failure.
    """
    torch_oom = ()
    try:
        torch = _torch()
        torch_oom = (torch.cuda.OutOfMemoryError,)
    except Exception:
        pass
    if torch_oom and isinstance(exc, torch_oom):
        return True
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


@contextlib.contextmanager
def track(trace: MemoryTrace | None = None) -> Iterator[MemoryTrace]:
    """Measure peak VRAM and wall time, converting OOM into a recorded result.

    Non-OOM exceptions are recorded and re-raised: a genuine bug should stop the
    run, while an OOM is the measurement we came for.
    """
    result = trace or MemoryTrace()
    reset_peak()
    start = time.perf_counter()
    try:
        yield result
    except BaseException as exc:  # noqa: BLE001 - classified immediately below
        result.seconds = time.perf_counter() - start
        result.peak_gb = peak_gb()
        result.error = f"{type(exc).__name__}: {exc}"
        if is_oom(exc):
            result.oom = True
            empty_cache()
            return
        raise
    else:
        result.seconds = time.perf_counter() - start
        result.peak_gb = peak_gb()

