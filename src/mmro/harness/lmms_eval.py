"""lmms-eval backend -- STUB until stage 0.2.

Integration notes, recorded now while the reasoning is fresh:

* lmms-eval wants a class deriving from ``lmms_eval.api.model.lmms``, registered
  with ``@register_model("mmro")``. The method that matters is
  ``generate_until(self, requests) -> list[str]``; each request carries the
  context, generation kwargs, and a doc-to-visual callable. The wrapper
  materializes the visuals to temp files and calls
  ``mmro.models.<adapter>.generate(images, text, decode)``.
* ``generate_until`` receives per-request ``until`` stop strings and gen kwargs.
  Honour the stop strings, but take decoding from ``cfg.decode_for("eval")`` so
  both harnesses provably decode the same way -- otherwise the cross-harness
  comparison measures decoding differences rather than harness differences.
* Runs from its own venv (``eval.harnesses.lmms_eval.venv``); shell out, never
  import lmms_eval in the base environment.
* Results land as JSON under ``--output_path``; parse ``results[task][metric]``
  and emit one ScoreRecord per metric.
* Task IDs differ from VLMEvalKit's (``mmmu_val`` vs ``MMMU_DEV_VAL``), which is
  why config.eval.tasks carries both and resolution goes through
  :func:`mmro.harness.base.resolve_task_id`.
* Note lmms-eval's MMMU split naming: confirm whether the configured ID targets
  validation or dev+validation, since VLMEvalKit's ``MMMU_DEV_VAL`` spans both.
  A split mismatch is the most likely cause of an unexplained score delta.
"""

from __future__ import annotations

from ..config import TaskRow
from ..records import ScoreRecord
from .base import HarnessBackend


class LMMSEvalBackend(HarnessBackend):
    name = "lmms_eval"

    def run(self, model_name: str, task: TaskRow) -> list[ScoreRecord]:
        task_id = task.id_for(self.name)
        raise NotImplementedError(
            f"lmms-eval backend is a stage-0.2 deliverable (would run "
            f"{task.name!r} as {task_id!r} against {model_name!r}). "
            f"Stage 0 ships the protocol and task mapping only."
        )
