"""VLMEvalKit backend -- STUB until stage 0.2.

Integration notes, recorded now while the reasoning is fresh:

* VLMEvalKit wants a model class deriving from ``vlmeval.vlm.base.BaseModel``
  with ``generate_inner(self, message, dataset=None) -> str``. ``message`` is a
  list of ``{"type": "image"|"text", "value": ...}`` dicts, so the wrapper
  splits it into image paths and a text string and calls
  ``mmro.models.<adapter>.generate(images, text, decode)`` -- the contract stage
  0 already proved.
* ``INTERLEAVE = True`` on the wrapper class, otherwise MMMU's multi-image items
  get flattened to the first image. Stage 0's multi-image evidence is exactly
  what makes this worth setting.
* It runs from its own venv (``eval.harnesses.vlmevalkit.venv``), so this
  backend builds a command and shells out. It must not import vlmeval here --
  that package's pins conflict with the base environment's.
* Results land as ``{work_dir}/{model}/{model}_{task}*.csv|.json``; parse the
  score file and emit one ScoreRecord per metric.
* MathVista free-form items need answer extraction. VLMEvalKit uses a judge
  model for this; with ``eval.judge.backend: none`` it falls back to rule-based
  extraction, which understates the score. Report which was used -- comparing a
  judged score against an unjudged one across harnesses is a trap.
"""

from __future__ import annotations

from ..config import TaskRow
from ..records import ScoreRecord
from .base import HarnessBackend


class VLMEvalKitBackend(HarnessBackend):
    name = "vlmevalkit"

    def run(self, model_name: str, task: TaskRow) -> list[ScoreRecord]:
        task_id = task.id_for(self.name)
        raise NotImplementedError(
            f"VLMEvalKit backend is a stage-0.2 deliverable (would run "
            f"{task.name!r} as {task_id!r} against {model_name!r}). "
            f"Stage 0 ships the protocol and task mapping only."
        )
