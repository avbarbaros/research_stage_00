"""VLMEvalKit backend.

Runs entirely inside ``eval.harnesses.vlmevalkit.venv``. Nothing from vlmeval is
imported here -- its pins conflict with the base environment, which is the whole
reason the venvs are separate.

The invocation is a template in config (``eval.harnesses.vlmevalkit.command``)
because VLMEvalKit's CLI changes between releases. When it does, you edit a
string in config.yaml rather than patching this file.

**Which model implementation runs.** By default the command names VLMEvalKit's
*own* registered Qwen2.5-VL class, not our adapter. That is deliberate for the
stage-0.2 gate: published numbers were produced by the harness's implementation,
so reproducing them means using it. Swapping in our adapter adds a variable and
belongs in a separate comparison, once the gate has passed.

Results land under ``work_dir`` as ``{model}_{dataset}_acc.csv`` alongside the
per-item ``.xlsx``. We search rather than assume an exact filename, because the
layout has moved across versions.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from ..config import TaskRow
from ..records import ScoreRecord
from .base import HarnessBackend, budget_for_task

# Column names VLMEvalKit has used for the headline accuracy, most recent first.
ACCURACY_COLUMNS = ("Overall", "overall", "Average", "acc", "Accuracy")


class VLMEvalKitBackend(HarnessBackend):
    name = "vlmevalkit"

    def run(self, model_name: str, task: TaskRow) -> list[ScoreRecord]:
        problems = self.check_available()
        if problems:
            raise RuntimeError("; ".join(problems))

        if self.cfg.eval.limit is not None:
            print(
                f"  NOTE: eval.limit={self.cfg.eval.limit} has NO effect on "
                f"VLMEvalKit -- its run.py has no per-dataset sample cap, so the "
                f"full {task.name} split will run regardless. Use lmms_eval for "
                f"a quick shakeout."
            )
        self._warn_if_judge_unconfigured()

        task_id = task.id_for(self.name)
        fields = self.command_fields(model_name, task)
        argv = self.harness_cfg.render_command(**fields)

        work = self.harness_cfg.work_path
        work.mkdir(parents=True, exist_ok=True)
        code = self.run_command(argv, work / f"{model_name}_{task.name}.log")
        if code != 0:
            raise RuntimeError(
                f"VLMEvalKit exited {code} for {task.name}; see the log in {work}"
            )

        records = self.parse_results(work, model_name, task, task_id)
        if not records:
            raise RuntimeError(
                f"VLMEvalKit produced no parseable score for {task_id} under {work}. "
                f"It may have written a layout this parser does not know; check the "
                f"directory and widen ACCURACY_COLUMNS or the glob if so."
            )
        return records

    def judge_key_present(self) -> bool:
        # VLMEvalKit also loads its key from a .env file in its own repo.
        if super().judge_key_present():
            return True
        env_file = self.harness_cfg.repo_path / ".env"
        key_env = self.cfg.eval.judge.api_key_env
        if not (key_env and env_file.exists()):
            return False
        prefix = f"{key_env}="
        return any(
            line.strip().startswith(prefix) and line.split("=", 1)[1].strip()
            for line in env_file.read_text(encoding="utf-8").splitlines()
        )

    def _warn_if_judge_unconfigured(self) -> None:
        """With eval.judge.backend 'none' VLMEvalKit still tries its own default
        OpenAI judge for MMMU/MathVista, and falls back silently to exact-match
        scoring when there is no key. Say so before hours of inference."""
        if self.cfg.eval.judge.backend != "none":
            return
        env_file = self.harness_cfg.repo_path / ".env"
        if os.environ.get("OPENAI_API_KEY") or env_file.exists():
            return
        print(
            "  WARNING: eval.judge.backend is 'none' and no OPENAI_API_KEY / "
            f"{env_file} exists -- VLMEvalKit will likely fall back to exact-match "
            "scoring (tens of points under baseline). Set eval.judge for a gated run."
        )

    # -- parsing ------------------------------------------------------------
    def find_score_files(self, work: Path, task_id: str) -> list[Path]:
        """Score CSVs for this task, newest first."""
        hits = [
            p for p in work.rglob("*.csv")
            if task_id.lower() in p.name.lower() and "acc" in p.name.lower()
        ]
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)

    def parse_results(
        self, work: Path, model_name: str, task: TaskRow, task_id: str
    ) -> list[ScoreRecord]:
        records: list[ScoreRecord] = []
        for path in self.find_score_files(work, task_id)[:1]:
            for metric, score in self._read_csv(path).items():
                records.append(
                    ScoreRecord(
                        harness=self.name,
                        model=model_name,
                        task=task.name,
                        harness_task_id=task_id,
                        metric=metric,
                        score=score,
                        decode=self.decode_echo(),
                        budget=budget_for_task(self.cfg, model_name, task),
                        raw_path=str(path),
                    )
                )
        return records

    @staticmethod
    def _read_csv(path: Path) -> dict[str, float]:
        """Pull the headline accuracy out of a VLMEvalKit ``*_acc.csv``.

        The file is typically one row of splits/categories. We take the overall
        column when present and fall back to the first numeric column, so a
        renamed header degrades to a wrong-but-visible metric name rather than a
        crash -- the report shows the metric, so a surprise is legible.
        """
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}
        row = rows[0]

        def as_pct(raw: str) -> float | None:
            try:
                value = float(str(raw).strip().rstrip("%"))
            except (TypeError, ValueError):
                return None
            # VLMEvalKit reports percentages; older builds emitted fractions.
            return value * 100 if 0.0 <= value <= 1.0 else value

        for column in ACCURACY_COLUMNS:
            if column in row:
                value = as_pct(row[column])
                if value is not None:
                    return {"accuracy": value}

        for key, raw in row.items():
            value = as_pct(raw)
            if value is not None:
                return {key.strip() or "accuracy": value}
        return {}
