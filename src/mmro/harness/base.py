"""Harness backend protocol and task-ID resolution.

Stage 0.2 runs MMMU and MathVista through **both** VLMEvalKit and lmms-eval and
compares the scores. That imposes three requirements, all satisfied here rather
than inside either backend:

1. **One task, two names.** The projects use different task IDs for the same
   benchmark, so ``config.eval.tasks`` carries both and :func:`resolve_task_id`
   maps between them. Getting this wrong means silently comparing two different
   benchmarks.
2. **One result shape.** Both backends emit :class:`~mmro.records.ScoreRecord`,
   so ``report.py`` joins on ``(model, task, metric)`` without special-casing.
3. **No shared dependencies.** VLMEvalKit and lmms-eval pin conflicting
   versions, so each backend shells out to its own venv. They never import each
   other, and neither is imported by the base environment.

Score deltas between the two harnesses are *expected* -- different prompt
templates, different answer extraction. The report shows both side by side.
Explaining a delta is stage-0.2 research output; hiding it would be a bug.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Config, ConfigError, TaskRow
from ..records import ScoreRecord

BACKENDS: dict[str, tuple[str, str]] = {
    "vlmevalkit": ("mmro.harness.vlmevalkit", "VLMEvalKitBackend"),
    "lmms_eval": ("mmro.harness.lmms_eval", "LMMSEvalBackend"),
}


def resolve_task_id(cfg: Config, task_name: str, harness: str) -> str:
    """The task ID ``harness`` uses for our canonical ``task_name``."""
    return cfg.eval.task(task_name).id_for(harness)


def task_matrix(cfg: Config) -> list[tuple[TaskRow, str]]:
    """Every (task, enabled harness) pair stage 0.2 will run."""
    return [
        (row, harness)
        for row in cfg.eval.tasks
        for harness in cfg.eval.enabled_harnesses
    ]


class HarnessBackend(ABC):
    """Runs one benchmark task against one model and returns scores."""

    name: str = ""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.harness_cfg = cfg.eval.harnesses.get(self.name)
        if self.harness_cfg is None:
            raise ConfigError(
                f"eval.harnesses.{self.name}: not configured; add it to config.yaml"
            )

    @abstractmethod
    def run(self, model_name: str, task: TaskRow) -> list[ScoreRecord]:
        """Run ``task`` against ``model_name``, returning normalized scores."""

    # -- shared helpers -----------------------------------------------------
    def python(self) -> Path:
        """Interpreter for this harness's isolated venv."""
        return self.harness_cfg.python

    def check_available(self) -> list[str]:
        """Problems that would stop this backend from running."""
        problems = []
        if not self.harness_cfg.enabled:
            problems.append(f"{self.name} is disabled in config.eval.harnesses")
        if not self.python().exists():
            problems.append(
                f"{self.name} venv is missing: {self.harness_cfg.venv_path} "
                f"(create it with scripts/03_setup_harness.sh)"
            )
        return problems

    def decode_echo(self) -> dict:
        """Decoding settings recorded on every score.

        Both harnesses must decode identically or the comparison is meaningless,
        so the settings travel with the result rather than being assumed.
        """
        return self.cfg.decode_for("eval").to_generate_kwargs()


def get_backend_class(harness: str) -> type[HarnessBackend]:
    if harness not in BACKENDS:
        raise ValueError(f"unknown harness {harness!r}; valid: {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[harness]
    return getattr(importlib.import_module(module_name), class_name)
