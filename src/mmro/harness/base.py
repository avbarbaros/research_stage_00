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
import subprocess
import sys
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


def budget_for_task(cfg: Config, model_name: str, task: TaskRow) -> dict:
    """The visual budget to run ``task`` at, derived from stage-0 measurements.

    Stage 0 established a **total** visual-token ceiling per model (VRAM for
    Qwen, comprehension for InternVL). That total is what the hardware and the
    model actually constrain; the per-image budget follows from dividing it by
    the most images a single question in this benchmark can contain.

    This matters for matching published numbers. MathVista is single-image, so
    it gets the whole ceiling for its one image and effectively runs at native
    resolution -- as the published evaluations did. Handing it the 7-image MMMU
    budget instead would downscale every chart and cost accuracy for a reason
    that is entirely our own doing.
    """
    model_cfg = cfg.model(model_name)
    budget = dict(model_cfg.budget)
    total = budget.pop("safe_total_visual_tokens", None)
    if total is None:
        # No measured ceiling: run at the configured budget and say nothing
        # further, rather than inventing a division.
        return budget

    per_image = max(1, int(total) // max(1, task.max_images))
    if model_cfg.adapter == "qwen25vl":
        budget["max_visual_tokens_per_image"] = per_image
    else:
        # InternVL is quantized to whole 448px tiles at 256 tokens each, and one
        # of those is the thumbnail, so convert tokens -> grid tiles.
        budget["max_tiles"] = max(1, per_image // 256 - 1)
    return budget


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

    def run_command(self, argv: list[str], log_path: Path) -> int:
        """Run a harness in its own venv, teeing output to a log.

        Streamed rather than captured: these runs take hours, and a silent
        terminal for that long is indistinguishable from a hang.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  $ {' '.join(argv)}")
        print(f"  log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
            return proc.wait()

    def command_fields(self, model_name: str, task: TaskRow) -> dict:
        """Placeholders available to a harness's command template."""
        model_cfg = self.cfg.model(model_name)
        decode = self.cfg.decode_for("eval")
        return {
            "python": str(self.python()),
            "repo_dir": str(self.harness_cfg.repo_path),
            "work_dir": str(self.harness_cfg.work_path),
            "task_id": task.id_for(self.name),
            "task": task.name,
            "model_id": self.harness_cfg.model_id or model_cfg.repo_id,
            "model_path": model_cfg.repo_id,
            "max_new_tokens": decode.max_new_tokens,
            # A ready CLI chunk, not a bare value: "" when unset so a template
            # that appends {limit} bare (no --limit prefix of its own) gets
            # nothing, and shlex.split drops the empty token cleanly. A bare
            # number would leave a template's own "--limit {limit}" rendering
            # as a dangling "--limit" with no value when limit is None.
            "limit": (
                "" if self.cfg.eval.limit is None else f"--limit {self.cfg.eval.limit}"
            ),
        }


def get_backend_class(harness: str) -> type[HarnessBackend]:
    if harness not in BACKENDS:
        raise ValueError(f"unknown harness {harness!r}; valid: {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[harness]
    return getattr(importlib.import_module(module_name), class_name)
