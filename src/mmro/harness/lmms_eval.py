"""lmms-eval backend.

Runs entirely inside ``eval.harnesses.lmms_eval.venv``; nothing from lmms_eval
is imported here. The invocation is a config template for the same reason as
VLMEvalKit's -- the CLI moves between releases.

**Which model implementation runs.** The command names lmms-eval's own
registered ``qwen2_5_vl`` model, matching how the published numbers were
produced. Substituting our adapter is a separate experiment, after the gate.

Results land as JSON under ``--output_path``, shaped roughly:

    {"results": {"<task_id>": {"<metric>,none": 0.586, ...}}, ...}

Metric keys carry a ``,filter`` suffix (``exact_match,none``) which we strip.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import TaskRow
from ..records import ScoreRecord
from .base import HarnessBackend, budget_for_task

# Keys that are bookkeeping rather than scores.
NON_METRIC_KEYS = {"alias", "samples"}

# lmms-eval's own ``mathvista_testmini`` task is a bare *group* -- three
# prompting variants (``_cot``, ``_format``, ``_solution``) with no
# ``aggregate_metric_list`` in the group's yaml. That means the group key
# itself never carries a numeric score in the results JSON, only bookkeeping
# (the subtask table lands under the three suffixed keys instead). Of those,
# ``_solution`` is direct-answer prompting with no intermediate reasoning
# step -- the protocol VLMEvalKit's ``MathVista_MINI`` and the published
# numbers use -- so it is the one comparable to the configured baseline.
# ``_cot`` and ``_format`` are separate prompting experiments, not the
# headline metric. Add other headline-less groups here as they turn up.
GROUP_HEADLINE_SUBTASK = {"mathvista_testmini": "mathvista_testmini_solution"}


class LMMSEvalBackend(HarnessBackend):
    name = "lmms_eval"

    def judge_env(self) -> dict[str, str]:
        # lmms-eval takes its judge model and provider from the environment,
        # not from a CLI flag -- and tasks disagree on the variable names:
        # - MathVista reads MODEL_VERSION / API_TYPE (+ OPENAI_API_KEY);
        # - the *_reasoning tasks (tasks/_task_utils/reasoning_utils.py) read
        #   JUDGE_MODEL_NAME / JUDGE_API_KEY, and consult the judge at all only
        #   when USE_LLM_JUDGE == "True" (default "False": rule-based only).
        #   Even then it re-checks just the items rule-based matching scored 0.
        env = super().judge_env()
        judge = self.cfg.eval.judge
        if judge.backend != "none":
            env["MODEL_VERSION"] = str(judge.model)
            env.setdefault("API_TYPE", "openai")
            env["JUDGE_MODEL_NAME"] = str(judge.model)
            env["USE_LLM_JUDGE"] = "True"
            if "OPENAI_API_KEY" in env:
                env["JUDGE_API_KEY"] = env["OPENAI_API_KEY"]
        return env

    def run(self, model_name: str, task: TaskRow) -> list[ScoreRecord]:
        problems = self.check_available(task)
        if problems:
            raise RuntimeError("; ".join(problems))

        task_id = task.id_for(self.name)
        self.ensure_weights(model_name)
        fields = self.command_fields(model_name, task)
        argv = self.harness_cfg.render_command(**fields)

        work = self.harness_cfg.work_path
        work.mkdir(parents=True, exist_ok=True)
        code = self.run_command(argv, work / f"{model_name}_{task.name}.log")
        if code != 0:
            raise RuntimeError(
                f"lmms-eval exited {code} for {task.name}; see the log in {work}"
            )

        records = self.parse_results(work, model_name, task, task_id)
        if not records:
            raise RuntimeError(
                f"lmms-eval produced no parseable score for {task_id} under {work}. "
                f"Check the results JSON layout; the parser looks for "
                f"results['{task_id}']."
            )
        return records

    # -- parsing ------------------------------------------------------------
    def find_result_files(self, work: Path) -> list[Path]:
        """Result JSONs, newest first. lmms-eval nests them under timestamps."""
        hits = [p for p in work.rglob("*results*.json")] or list(work.rglob("*.json"))
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)

    def parse_results(
        self, work: Path, model_name: str, task: TaskRow, task_id: str
    ) -> list[ScoreRecord]:
        for path in self.find_result_files(work):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            effective_id = self.resolve_scored_task_id(payload, task_id)
            metrics = self.extract_metrics(payload, effective_id)
            if not metrics:
                continue
            return [
                ScoreRecord(
                    harness=self.name,
                    model=model_name,
                    task=task.name,
                    harness_task_id=effective_id,
                    metric=metric,
                    score=score,
                    n_samples=self.extract_n_samples(payload, effective_id),
                    decode=self.decode_echo(),
                    budget=budget_for_task(self.cfg, model_name, task),
                    raw_path=str(path),
                )
                for metric, score in metrics.items()
            ]
        return []

    @staticmethod
    def resolve_scored_task_id(payload: dict, task_id: str) -> str:
        """``task_id``, or the subtask that actually carries its score.

        Most lmms-eval tasks put their metric directly under ``results[task_id]``.
        A few (see :data:`GROUP_HEADLINE_SUBTASK`) are bare groups whose own key
        never gets a number -- the real score lives one level down, under a
        suffixed subtask key.
        """
        block = (payload.get("results") or {}).get(task_id)
        has_direct_score = isinstance(block, dict) and any(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            and str(k).split(",")[0].strip() not in NON_METRIC_KEYS
            for k, v in block.items()
        )
        if has_direct_score:
            return task_id
        return GROUP_HEADLINE_SUBTASK.get(task_id, task_id)

    @staticmethod
    def extract_metrics(payload: dict, task_id: str) -> dict[str, float]:
        """Numeric metrics for ``task_id``, with filter suffixes stripped.

        Scores arrive as fractions; the gate compares against published
        percentages, so convert once here rather than at every comparison.
        """
        block = (payload.get("results") or {}).get(task_id)
        if not isinstance(block, dict):
            return {}

        out: dict[str, float] = {}
        for key, value in block.items():
            name = str(key).split(",")[0].strip()
            if name in NON_METRIC_KEYS or not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool):
                continue
            out[name] = float(value) * 100 if 0.0 <= float(value) <= 1.0 else float(value)
        return out

    @staticmethod
    def extract_n_samples(payload: dict, task_id: str) -> int:
        counts = payload.get("n-samples") or payload.get("n_samples") or {}
        entry = counts.get(task_id) if isinstance(counts, dict) else None
        if isinstance(entry, dict):
            for key in ("effective", "original"):
                if isinstance(entry.get(key), int):
                    return int(entry[key])
        return int(entry) if isinstance(entry, int) else 0
