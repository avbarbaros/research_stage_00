"""Benchmark harness backends (stage 0.2).

The protocol, the normalized :class:`~mmro.records.ScoreRecord`, and task-ID
resolution are live now so stage 0 can prove the seam. The two backends are
stubs until stage 0.2.
"""

from .base import BACKENDS, HarnessBackend, get_backend_class, resolve_task_id

__all__ = ["BACKENDS", "HarnessBackend", "get_backend_class", "resolve_task_id"]
