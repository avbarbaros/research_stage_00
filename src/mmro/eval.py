"""Stage 0.2 — run the benchmark harnesses and check the gate.

Two jobs, deliberately separated:

1. **The gate.** Reproduce published Qwen2.5-VL-7B numbers within
   ``eval.gate_tolerance`` absolute points. This validates the *plumbing*: if we
   cannot reproduce a known result, no later number from this pipeline means
   anything.
2. **The comparison.** Run both harnesses on the same task and record the delta.
   Deltas are expected -- different prompt templates, different answer
   extraction -- and explaining them is the research output.

A run that fails the gate is not a finding about the model. It is a finding
about our setup, and the most common causes are listed by :func:`diagnose`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .config import Config, ConfigError, TaskRow, load_config
from .harness import get_backend_class
from .harness.base import budget_for_task
from .records import ScoreRecord, append_jsonl, iter_results


@dataclass
class GateResult:
    task: str
    harness: str
    model: str
    metric: str
    score: float
    baseline: float | None
    tolerance: float

    @property
    def delta(self) -> float | None:
        return None if self.baseline is None else self.score - self.baseline

    @property
    def passed(self) -> bool | None:
        """None when there is no baseline to check against."""
        d = self.delta
        return None if d is None else abs(d) <= self.tolerance


def check_gate(cfg: Config, scores: list[ScoreRecord]) -> list[GateResult]:
    """Compare each headline score against its published baseline."""
    out: list[GateResult] = []
    for record in scores:
        try:
            cfg.eval.task(record.task)  # only tasks we configured are gated
        except ConfigError:
            continue
        # Only the headline metric is gated; sub-scores vary in definition
        # between harnesses and would produce noise, not signal.
        if record.metric not in ("accuracy", "exact_match"):
            continue
        try:
            baseline = cfg.model(record.model).baselines.get(record.task)
        except ConfigError:
            baseline = None  # score from a model no longer in the registry
        out.append(
            GateResult(
                task=record.task,
                harness=record.harness,
                model=record.model,
                metric=record.metric,
                score=record.score,
                baseline=baseline,
                tolerance=cfg.eval.gate_tolerance,
            )
        )
    return out


def diagnose(cfg: Config, gate: list[GateResult]) -> list[str]:
    """Likely causes for a failed gate, most probable first.

    Ordered by what has actually gone wrong in practice, not by severity. The
    first two account for most large misses and are both configuration, not
    model behaviour.
    """
    notes: list[str] = []
    failures = [g for g in gate if g.passed is False]
    if not failures:
        return notes

    needs_judge = any(cfg.eval.task(g.task).needs_judge(g.harness) for g in failures)
    if cfg.eval.judge.backend == "none" and needs_judge:
        notes.append(
            "eval.judge.backend is 'none'. MathVista and MMMU's open-ended items "
            "need LLM answer extraction; published numbers use it. Rule-based "
            "fallback systematically UNDER-scores, often by several points. If a "
            "score is low, fix this first."
        )

    low = [g for g in failures if (g.delta or 0) < 0]
    if low:
        notes.append(
            "Scores below baseline are usually resolution or extraction, not the "
            "model. Check the `budget` recorded on each score: a task whose "
            "max_images is set too high in config gets a needlessly small "
            "per-image budget and loses fine detail (charts, diagrams, text)."
        )
        notes.append(
            "Confirm the split matches the published one. VLMEvalKit's "
            "MMMU_DEV_VAL spans dev+validation; lmms-eval's mmmu_val is "
            "validation only. Comparing either against a number computed on the "
            "other is a split mismatch, not a bug in the model."
        )

    if cfg.eval.limit is not None:
        notes.append(
            f"eval.limit is {cfg.eval.limit}, so this ran on a subset. Subset "
            f"scores are noisy and must not be gated against a full-set "
            f"baseline. Set limit: null for the real run."
        )

    if any(g.delta and abs(g.delta) > 15 for g in failures):
        notes.append(
            "A miss above 15 points usually means the wrong thing ran entirely: "
            "wrong checkpoint, wrong task ID, or the harness silently fell back "
            "to a different model. Check the log and the raw_path on the score."
        )
    return notes


def run_task(cfg: Config, model_name: str, task: TaskRow, harness: str) -> list[ScoreRecord]:
    backend = get_backend_class(harness)(cfg)
    print(f"\n=== {harness}: {task.name} ({task.id_for(harness)}) on {model_name} ===")
    budget = budget_for_task(cfg, model_name, task)
    print(f"  max_images={task.max_images} -> budget {budget}")
    return backend.run(model_name, task)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the stage-0.2 benchmark harnesses.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", default="qwen2_5_vl_7b", help="model from the registry")
    parser.add_argument("--task", default=None, help="one task (default: all configured)")
    parser.add_argument("--harness", default=None, help="one harness (default: all enabled)")
    parser.add_argument(
        "--gate-only", action="store_true",
        help="skip running; just re-check the gate against existing scores",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    results_dir = cfg.paths.results_path

    if not cfg.eval.enabled and not args.gate_only:
        print("eval.enabled is false in config; set it to true to run stage 0.2")
        return 1

    try:
        cfg.model(args.model)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1

    if not args.gate_only:
        from .env import apply_env, seed_everything  # noqa: PLC0415

        apply_env(cfg)
        seed_everything(cfg.runtime.seed)

        tasks = [cfg.eval.task(args.task)] if args.task else cfg.eval.tasks
        harnesses = [args.harness] if args.harness else list(cfg.eval.enabled_harnesses)
        if not tasks or not harnesses:
            print("nothing to run: no tasks or no enabled harnesses in config")
            return 1

        failed = False
        for harness in harnesses:
            out_path = results_dir / f"scores_{harness}.jsonl"
            for task in tasks:
                try:
                    for record in run_task(cfg, args.model, task, harness):
                        append_jsonl(out_path, record)
                        print(f"  {record.metric}: {record.score:.2f}")
                except Exception as exc:  # noqa: BLE001 - one task must not sink the rest
                    print(f"  FAILED: {exc}")
                    failed = True
        if failed:
            print("\nsome runs failed; the gate below covers only what completed")

    # -- gate ---------------------------------------------------------------
    scores = list(iter_results(results_dir, "scores_*.jsonl", ScoreRecord))
    scores = [s for s in scores if s.model == args.model]
    if not scores:
        print("\nno scores recorded; nothing to gate")
        return 1

    gate = check_gate(cfg, scores)
    print(f"\n=== GATE (±{cfg.eval.gate_tolerance} absolute points vs published) ===")
    print(f"  {'task':<22}{'harness':<12}{'ours':>8}{'published':>11}{'delta':>8}  result")
    blocked = False
    for g in sorted(gate, key=lambda x: (x.task, x.harness)):
        if g.baseline is None:
            print(f"  {g.task:<22}{g.harness:<12}{g.score:>8.2f}{'—':>11}{'—':>8}  no baseline")
            continue
        verdict = "PASS" if g.passed else "**FAIL**"
        blocked = blocked or not g.passed
        print(
            f"  {g.task:<22}{g.harness:<12}{g.score:>8.2f}{g.baseline:>11.2f}"
            f"{g.delta:>+8.2f}  {verdict}"
        )

    for note in diagnose(cfg, gate):
        print(f"\n  - {note}")

    if blocked:
        print("\nGATE FAILED — debug before proceeding. This is the stage-0.2 gate.")
        return 1
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
