"""OOM-boundary sweep.

Stage 0's real deliverable. "Full resolution without OOM" is not a yes/no
question on a 24GB L4 -- it depends on how many images share the budget -- so we
measure where the boundary actually is and hand stage 0.2 a number.

For each (model, image count) cell, walk the configured ladder from the highest
rung down and stop at the first rung that produces a caption. That rung is the
measured boundary.

Two structural points:

* Each cell runs in a **fresh subprocess**. Both model families bake
  preprocessing limits into the processor at construction, so a rung change
  means a reload; and a process that has just OOM'd is not a trustworthy place
  to measure the next rung's peak memory.
* An OOM is a data point, not a failure. Only an unexpected exception stops the
  sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .budget import budget_at_rung, distinct_rungs, estimate_visual_tokens, ladder_for
from .config import Config, ConfigError, load_config
from .images import ImageSetError, provenance, require, sizes_of
from .records import GenerationResult, append_jsonl, read_jsonl


# ---------------------------------------------------------------------------
# Inner: one cell, one process
# ---------------------------------------------------------------------------
def run_cell(cfg: Config, model_name: str, n_images: int, rung: int) -> GenerationResult:
    """Attempt one (model, image count, rung) cell in this process."""
    from .models import load_adapter  # noqa: PLC0415

    model_cfg = cfg.model(model_name)
    budget = budget_at_rung(model_cfg.adapter, model_cfg.budget, rung)
    images = require(cfg.paths.images_path, n_images)
    estimate = estimate_visual_tokens(model_cfg.adapter, sizes_of(images), budget)

    adapter = load_adapter(cfg, model_name, load=False)
    # Load at this rung: preprocessing limits are fixed at processor
    # construction for both families, so they cannot be changed after loading.
    adapter.load(budget=budget)

    result = adapter.generate_measured(
        images=[i.path for i in images],
        text=cfg.smoke.multi_image_prompt if n_images > 1 else cfg.smoke.prompt,
        decode=cfg.decode_for("sweep"),
        budget=budget,
        estimated_visual_tokens=estimate,
        rung=rung,
        image_provenance=provenance(images),
        request_id=f"sweep-{model_name}-{n_images}img-rung{rung}",
    )
    adapter.unload()
    return result


# ---------------------------------------------------------------------------
# Outer: orchestrate cells as subprocesses
# ---------------------------------------------------------------------------
def _cell_command(args_ns, model_name: str, n_images: int, rung: int) -> list[str]:
    cmd = [sys.executable, "-m", "mmro.sweep", "--cell",
           "--model", model_name, "--images", str(n_images), "--rung", str(rung)]
    if args_ns.config:
        cmd += ["--config", args_ns.config]
    for override in args_ns.overrides:
        cmd += ["--set", override]
    return cmd


def sweep_model(cfg: Config, model_name: str, args_ns, out_path: Path) -> list[GenerationResult]:
    model_cfg = cfg.model(model_name)
    ladder = ladder_for(model_cfg.adapter, cfg.sweep)
    results: list[GenerationResult] = []

    for n_images in sorted(set(cfg.sweep.image_counts)):
        print(f"\n=== {model_name}: {n_images} image(s) ===")

        images = require(cfg.paths.images_path, n_images)
        rungs, skipped = distinct_rungs(
            model_cfg.adapter, model_cfg.budget, sizes_of(images), ladder
        )
        # Never narrow the sweep silently: say which rungs were dropped and why.
        for rung, estimate, kept_rung in skipped:
            print(
                f"  rung {rung} skipped: identical to rung {kept_rung} for these "
                f"images ({estimate} visual tokens)"
            )

        for rung, _estimate in rungs:
            print(f"  rung {rung} ... ", end="", flush=True)
            proc = subprocess.run(
                _cell_command(args_ns, model_name, n_images, rung),
                capture_output=True,
                text=True,
            )
            result = _parse_cell_output(proc, model_name, n_images, rung)
            append_jsonl(out_path, result)
            results.append(result)

            if result.ok:
                print(
                    f"OK  ({result.visual_tokens} visual tokens, "
                    f"peak {result.peak_vram_gb:.2f} GB) -- boundary for "
                    f"{n_images} image(s)"
                )
                break
            if result.oom:
                print(f"OOM (peak {result.peak_vram_gb:.2f} GB)")
            else:
                print(f"FAILED: {result.error}")
                # A non-OOM failure is not a boundary measurement -- descending
                # the ladder would just repeat the same bug at every rung.
                break
        else:
            print(f"  no rung survived at {n_images} image(s)")

    return results


def _parse_cell_output(proc, model_name: str, n_images: int, rung: int) -> GenerationResult:
    """Recover the cell's record from its stdout.

    A cell that dies hard (CUDA abort, OOM killer) prints nothing parseable, so
    we synthesize a record instead of losing the cell entirely.
    """
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CELL_RESULT "):
            data = json.loads(line[len("CELL_RESULT "):])
            valid = {f for f in GenerationResult.__dataclass_fields__}
            return GenerationResult(**{k: v for k, v in data.items() if k in valid})

    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit code {proc.returncode}"
    killed = proc.returncode in (-9, 137)
    return GenerationResult(
        model=model_name,
        request_id=f"sweep-{model_name}-{n_images}img-rung{rung}",
        n_images=n_images,
        rung=rung,
        # A SIGKILL here is host RAM exhaustion during load, not CUDA OOM.
        # Recording it as OOM would put the boundary in the wrong place.
        oom=killed,
        error=("process killed (likely host RAM, not VRAM): " if killed else "") + detail,
    )


def boundaries(results: list[GenerationResult]) -> dict[tuple[str, int], GenerationResult]:
    """The highest surviving rung per (model, image count)."""
    best: dict[tuple[str, int], GenerationResult] = {}
    for result in results:
        if not result.ok or result.rung is None:
            continue
        key = (result.model, result.n_images)
        if key not in best or result.rung > (best[key].rung or 0):
            best[key] = result
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the OOM boundary per model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", default=None, help="sweep one model (default: all enabled)")
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--images", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--rung", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)

    from .env import apply_env, seed_everything  # noqa: PLC0415

    apply_env(cfg)
    seed_everything(cfg.runtime.seed)

    # -- inner mode: one cell, report on stdout, always exit 0 --------------
    if args.cell:
        import dataclasses  # noqa: PLC0415

        try:
            result = run_cell(cfg, args.model, args.images, args.rung)
        except ImageSetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("CELL_RESULT " + json.dumps(dataclasses.asdict(result)))
        return 0

    # -- outer mode ---------------------------------------------------------
    try:
        targets = {args.model: cfg.model(args.model)} if args.model else cfg.enabled_models
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1
    if not targets:
        print("no enabled models in config; nothing to sweep")
        return 1

    try:
        require(cfg.paths.images_path, max(cfg.sweep.image_counts))
    except ImageSetError as exc:
        print(f"error: {exc}")
        return 1

    all_results: list[GenerationResult] = []
    for name in targets:
        out_path = cfg.paths.results_path / f"sweep_{name}.jsonl"
        if out_path.exists():
            out_path.unlink()  # a sweep is a fresh measurement, not an append
        all_results += sweep_model(cfg, name, args, out_path)

    print("\n=== measured boundaries ===")
    found = boundaries(all_results)
    if not found:
        print("  none -- no cell survived at any rung")
        return 1
    for (model, n_images), result in sorted(found.items()):
        print(
            f"  {model:<22} {n_images} image(s): rung {result.rung:<5} "
            f"{result.visual_tokens:>6} visual tokens  "
            f"peak {result.peak_vram_gb:.2f} GB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
