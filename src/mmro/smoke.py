"""Smoke test: does this model caption held-out images end to end?

Runs **one model per process**. Loading a second 16.5GB model into a process
that has already held one does not work reliably on a 24GB L4 -- fragmentation
defeats ``empty_cache`` -- so ``scripts/01_smoke.sh`` invokes this once per
model and this module refuses to loop over models itself.

What it proves, in order of what actually goes wrong:

1. the weights load in the configured dtype without OOM
2. a caption comes back that is non-empty and on-topic (read it -- a broken
   vision path yields fluent, confident nonsense, not an error)
3. multi-image prompts reach the model intact, which we check by asserting the
   measured visual-token count *grows* with image count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .budget import estimate_visual_tokens
from .config import Config, ConfigError, load_config
from .images import ImageSetError, advisories, provenance, require, sizes_of
from .records import GenerationResult, append_jsonl


def run_model(cfg: Config, model_name: str, out_path: Path) -> list[GenerationResult]:
    from .models import load_adapter  # noqa: PLC0415

    model_cfg = cfg.model(model_name)
    decode = cfg.decode_for("smoke")
    counts = sorted(set(cfg.smoke.image_counts))

    # Fail before loading 16.5GB of weights if the image set cannot satisfy the
    # largest requested count.
    largest = max(counts)
    largest_set = require(cfg.paths.images_path, largest)

    # Surfaced here, not just in `mmro.images`: a caption from a 0.07 MP image
    # proves the plumbing works and nothing about high-resolution behaviour.
    notes = advisories(largest_set, cfg.images.min_long_side)
    for note in notes:
        print(f"NOTE: {note}")
    if notes:
        print(
            "NOTE: this run tests the pipeline, not resolution limits — "
            "the sweep needs a larger image set to measure a real boundary\n"
        )

    print(f"loading {model_name} ({model_cfg.repo_id}) ...")
    adapter = load_adapter(cfg, model_name)
    print(f"  loaded in {adapter.load_seconds:.1f}s")

    results: list[GenerationResult] = []
    for n in counts:
        images = require(cfg.paths.images_path, n)
        prompt = cfg.smoke.prompt if n == 1 else cfg.smoke.multi_image_prompt
        estimate = estimate_visual_tokens(model_cfg.adapter, sizes_of(images), model_cfg.budget)

        print(f"\n--- {model_name}: {n} image(s), ~{estimate} visual tokens estimated ---")
        result = adapter.generate_measured(
            images=[i.path for i in images],
            text=prompt,
            decode=decode,
            estimated_visual_tokens=estimate,
            image_provenance=provenance(images),
            request_id=f"smoke-{model_name}-{n}img",
        )
        append_jsonl(out_path, result)
        results.append(result)

        if result.oom:
            print(f"  OOM at {n} image(s) (peak {result.peak_vram_gb:.2f} GB)")
        elif result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(
                f"  visual tokens: {result.visual_tokens} measured "
                f"vs {estimate} estimated"
                f"{'  MISMATCH' if result.visual_tokens != estimate else ''}"
            )
            print(f"  peak VRAM    : {result.peak_vram_gb:.2f} GB")
            print(f"  generated in : {result.generate_seconds:.1f}s")
            print(f"  caption      : {result.text[:400]}")

    adapter.unload()
    return results


def summarize(model_name: str, results: list[GenerationResult]) -> tuple[bool, list[str]]:
    """Decide pass/fail and explain why. Returns ``(passed, notes)``."""
    notes: list[str] = []
    passed = True

    for result in results:
        if result.oom:
            notes.append(f"{result.n_images} image(s): OOM")
            passed = False
        elif result.error:
            notes.append(f"{result.n_images} image(s): {result.error}")
            passed = False
        elif not result.text.strip():
            notes.append(f"{result.n_images} image(s): empty caption")
            passed = False
        elif result.token_estimate_matches is False:
            # Not fatal: the caption is real and the model ran. But budget.py is
            # then wrong about this model, and the sweep's rungs would mean
            # something other than what they claim.
            notes.append(
                f"{result.n_images} image(s): visual-token estimate "
                f"{result.estimated_visual_tokens} != measured {result.visual_tokens} "
                f"-- budget.py needs correcting before the sweep is trustworthy"
            )

    # Multi-image evidence: tokens must grow with image count, or images are
    # being silently dropped somewhere in preprocessing.
    ok = [r for r in results if r.ok]
    if len(ok) >= 2:
        ordered = sorted(ok, key=lambda r: r.n_images)
        for lower, higher in zip(ordered, ordered[1:]):
            if higher.visual_tokens <= lower.visual_tokens:
                notes.append(
                    f"visual tokens did not grow from {lower.n_images} to "
                    f"{higher.n_images} images ({lower.visual_tokens} -> "
                    f"{higher.visual_tokens}) -- images are likely being dropped"
                )
                passed = False
    elif len(ok) < 2:
        notes.append("fewer than two successful runs; multi-image path unproven")
        passed = False

    return passed, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test one model end to end.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", required=True, help="model name from the config registry")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)

    from .env import apply_env, seed_everything  # noqa: PLC0415

    apply_env(cfg)
    seed_everything(cfg.runtime.seed)

    try:
        model_cfg = cfg.model(args.model)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1
    if not model_cfg.enabled:
        print(f"model {args.model!r} is disabled in config; nothing to do")
        return 0

    out_path = cfg.paths.results_path / f"smoke_{args.model}.jsonl"
    try:
        results = run_model(cfg, args.model, out_path)
    except ImageSetError as exc:
        print(f"error: {exc}")
        return 1

    passed, notes = summarize(args.model, results)
    print(f"\n=== {args.model}: {'PASS' if passed else 'FAIL'} ===")
    for note in notes:
        print(f"  - {note}")
    print(f"results: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
