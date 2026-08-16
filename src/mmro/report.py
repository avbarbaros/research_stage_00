"""Aggregate results into results/STAGE0.md.

Reads the JSONL that smoke and sweep wrote and produces the document a human
actually judges the milestone from. The captions are reproduced in full on
purpose: criterion 2 of the milestone -- "on-topic" -- cannot be checked by a
script. A model with a silently broken vision path returns fluent, confident
text, and only reading it against the image catches that.

The stage-0.2 harness-comparison section renders when ScoreRecords exist.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import Config, load_config
from .records import GenerationResult, ScoreRecord, iter_results, latest_per_image_count
from .sweep import boundaries, unbounded_cells


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def smoke_section(cfg: Config, results: list[GenerationResult]) -> list[str]:
    if not results:
        return ["_No smoke results. Run `bash scripts/01_smoke.sh`._", ""]

    stale = len(results) - len(latest_per_image_count(results))
    results = latest_per_image_count(results)

    lines = []
    if stale:
        lines += [
            f"_Note: {stale} superseded result(s) from earlier smoke runs were "
            f"ignored; only the most recent run per image count is shown._",
            "",
        ]
    lines += [
        "| Model | Images | Visual tokens (measured / estimated) | Prompt tok | Peak VRAM (GB) | Gen (s) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x.model, x.n_images)):
        if r.oom:
            status = "**OOM**"
        elif r.error:
            status = "**ERROR**"
        elif not r.text.strip():
            status = "**EMPTY**"
        elif r.token_estimate_matches is False:
            status = "ok (estimate mismatch)"
        else:
            status = "ok"
        tokens = f"{r.visual_tokens} / {r.estimated_visual_tokens}"
        lines.append(
            f"| {r.model} | {r.n_images} | {tokens} | {r.prompt_tokens} | "
            f"{_fmt(r.peak_vram_gb)} | {_fmt(r.generate_seconds, 1)} | {status} |"
        )
    lines.append("")

    # Multi-image evidence, stated explicitly rather than left to the reader.
    lines += ["### Multi-image evidence", ""]
    by_model: dict[str, list[GenerationResult]] = defaultdict(list)
    for r in results:
        if r.ok:
            by_model[r.model].append(r)
    for model, runs in sorted(by_model.items()):
        ordered = sorted(runs, key=lambda x: x.n_images)
        if len(ordered) < 2:
            lines.append(f"- **{model}**: only one successful run — multi-image path unproven.")
            continue
        trail = " → ".join(f"{r.n_images} img: {r.visual_tokens} tok" for r in ordered)
        grows = all(b.visual_tokens > a.visual_tokens for a, b in zip(ordered, ordered[1:]))
        verdict = (
            "visual tokens grow with image count, so every image reached the model"
            if grows
            else "**visual tokens did not grow — images are likely being dropped**"
        )
        lines.append(f"- **{model}**: {trail} — {verdict}.")
    lines.append("")

    lines += ["### Captions (read these)", ""]
    for r in sorted(results, key=lambda x: (x.model, x.n_images)):
        lines.append(f"**{r.model} — {r.n_images} image(s)**")
        lines.append("")
        lines.append(f"Images: `{', '.join(r.images)}`")
        lines.append("")
        if r.oom or r.error:
            lines.append(f"> _{'OOM' if r.oom else 'ERROR'}: {r.error}_")
        else:
            lines.append("> " + (r.text.strip() or "_(empty)_").replace("\n", "\n> "))
        lines.append("")
    return lines


def sweep_section(cfg: Config, results: list[GenerationResult]) -> list[str]:
    if not results:
        return ["_No sweep results. Run `bash scripts/02_sweep.sh`._", ""]

    lines = [
        "The highest ladder rung that produced a caption, per model and image count.",
        "This is stage 0's deliverable: the input to stage 0.2's token budgeting.",
        "",
        "| Model | Images | Max rung | Visual tokens | Peak VRAM (GB) | Estimate agrees |",
        "|---|---|---|---|---|---|",
    ]
    found = boundaries(results)
    unbounded = unbounded_cells(results)
    for (model, n_images), r in sorted(found.items()):
        agrees = {True: "yes", False: "**no**", None: "—"}[r.token_estimate_matches]
        note = " **(top rung — not a real limit)**" if (model, n_images) in unbounded else ""
        lines.append(
            f"| {model} | {n_images} | {r.rung}{note} | {r.visual_tokens} | "
            f"{_fmt(r.peak_vram_gb)} | {agrees} |"
        )
    lines.append("")

    if unbounded:
        lines += [
            f"> **{len(unbounded)} of {len(found)} cell(s) never OOM'd.** They "
            f"survived at the highest rung attempted, so the ladder ran out before "
            f"the hardware did — these rows report a value we chose in config, not "
            f"a measured limit. Raise the ladders in `config.yaml` and re-run; if "
            f"that changes nothing, the images ran out of pixels first and you need "
            f"higher-resolution originals.",
            "",
        ]

    # Cells where nothing survived are the most important thing on the page.
    attempted = {(r.model, r.n_images) for r in results}
    for key in sorted(attempted - set(found)):
        lines.append(f"- **{key[0]} at {key[1]} image(s): no rung survived.**")
    if attempted - set(found):
        lines.append("")

    # The boundary is a VRAM limit, not a usable limit. A model can fit in memory
    # and still lose track of its inputs at extreme token counts -- and it is the
    # boundary settings, not the smoke settings, that stage 0.2 would inherit.
    # So these captions need reading just as carefully as the smoke ones.
    lines += [
        "### Captions at the boundary (read these too)",
        "",
        "The rows above are **VRAM** limits. A model can fit in memory and still "
        "stop tracking its inputs at extreme token counts, so check that the "
        "caption at each boundary still describes the right images in the right "
        "order — especially for multi-image cells.",
        "",
    ]
    # Every successful rung, not just the boundary. With --all-rungs this becomes
    # a quality ladder: read down it to find where the model stops tracking its
    # inputs, which can be far below where it stops fitting in memory.
    successes = sorted(
        (r for r in results if r.ok),
        key=lambda r: (r.model, r.n_images, -(r.rung or 0)),
    )
    for r in successes:
        marker = " — **boundary**" if found.get((r.model, r.n_images)) is r else ""
        lines.append(
            f"**{r.model} — {r.n_images} image(s) @ rung {r.rung} "
            f"({r.visual_tokens} tok){marker}**"
        )
        lines.append("")
        lines.append(f"Images: `{', '.join(r.images)}`")
        lines.append("")
        lines.append("> " + (r.text.strip() or "_(empty)_").replace("\n", "\n> "))
        lines.append("")

    lines += ["<details><summary>All attempts</summary>", "",
              "| Model | Images | Rung | Result | Visual tokens | Peak VRAM (GB) |",
              "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (x.model, x.n_images, -(x.rung or 0))):
        outcome = "OOM" if r.oom else ("ok" if r.ok else "failed")
        lines.append(
            f"| {r.model} | {r.n_images} | {r.rung} | {outcome} | "
            f"{r.visual_tokens} | {_fmt(r.peak_vram_gb)} |"
        )
    lines += ["", "</details>", ""]
    return lines


def eval_section(cfg: Config, scores: list[ScoreRecord]) -> list[str]:
    if not scores:
        harnesses = ", ".join(cfg.eval.enabled_harnesses) or "none enabled"
        tasks = ", ".join(t.name for t in cfg.eval.tasks) or "none configured"
        return [
            f"_Stage 0.2, not yet run. Configured harnesses: {harnesses}. "
            f"Tasks: {tasks}._",
            "",
        ]

    lines = [
        "Same model, same task, both harnesses. Deltas are expected — the two use",
        "different prompt templates and answer extraction — and explaining them is",
        "the point of running both.",
        "",
        "| Model | Task | Metric | VLMEvalKit | lmms-eval | Δ |",
        "|---|---|---|---|---|---|",
    ]
    joined: dict[tuple[str, str, str], dict[str, ScoreRecord]] = defaultdict(dict)
    for s in scores:
        joined[(s.model, s.task, s.metric)][s.harness] = s

    for (model, task, metric), by_harness in sorted(joined.items()):
        a = by_harness.get("vlmevalkit")
        b = by_harness.get("lmms_eval")
        a_txt = _fmt(a.score) if a else "—"
        b_txt = _fmt(b.score) if b else "—"
        delta = _fmt(a.score - b.score) if a and b else "—"
        lines.append(f"| {model} | {task} | {metric} | {a_txt} | {b_txt} | {delta} |")
    lines.append("")

    # Decoding AND visual budget must match, or the comparison measures our own
    # configuration rather than the harnesses.
    for label, index in (("decoding", 0), ("visual budget", 1)):
        if len({s.conditions[index] for s in scores}) > 1:
            lines += [
                f"> **Warning: not all scores were produced under the same "
                f"{label}**, so these numbers are not directly comparable. A gap "
                f"here is our config, not a harness difference.",
                "",
            ]
    return lines


def build(cfg: Config) -> str:
    results_dir = cfg.paths.results_path
    smoke = list(iter_results(results_dir, "smoke_*.jsonl", GenerationResult))
    sweep = list(iter_results(results_dir, "sweep_*.jsonl", GenerationResult))
    scores = list(iter_results(results_dir, "scores_*.jsonl", ScoreRecord))

    lines = [
        "# Stage 0 — Multimodal Reasoning Orientation",
        "",
        "Generated by `python -m mmro.report`.",
        "",
        "## 1. Smoke test — do the models caption held-out images?",
        "",
        *smoke_section(cfg, smoke),
        "## 2. OOM boundary",
        "",
        *sweep_section(cfg, sweep),
        "## 3. Benchmark comparison (stage 0.2)",
        "",
        *eval_section(cfg, scores),
        "## Milestone checklist",
        "",
        "1. [ ] Every enabled model loads in the configured dtype without OOM",
        "2. [ ] Each model returns a non-empty, **on-topic** caption "
        "(judged by reading §1 — a fluent caption that does not match the image "
        "is a broken vision path, and fails even though nothing errored)",
        "3. [ ] Each model handles ≥3 images in one turn, with visual tokens "
        "growing by image count (§1)",
        "4. [ ] A concrete resolution boundary per image count, with peak VRAM, "
        "and `budget.py`'s estimate agreeing with the measurement (§2)",
        "5. [ ] Config control is real — `bash scripts/check_config_control.sh`",
        "6. [ ] The stage-0.2 seam is proven — `pytest tests/test_seam.py`",
        "",
    ]

    env_lock = results_dir / "env_lock.json"
    if env_lock.exists():
        lines += ["---", "", f"Environment recorded in `{env_lock.name}`.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build results/STAGE0.md from result JSONL.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default=None, help="output path (default: results/STAGE0.md)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    out = Path(args.out) if args.out else cfg.paths.results_path / "STAGE0.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(cfg))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
