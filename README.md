# research_stage_00

Multimodal reasoning models — **Stage 0**.

Before any fine-tuning is meaningful, establish that the two candidate backbones
actually run on the target hardware with the inputs the research needs
(**multi-image prompts at high resolution**), and that our benchmark pipeline can
reproduce a published number.

| | |
|---|---|
| Models | `Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3-8B-hf` (+ native fallback) |
| Target | Google Colab terminal, **L4 24GB**, weights cached on Google Drive |
| Stack | PyTorch · Transformers · PEFT · FSDP · VLMEvalKit · lmms-eval (vLLM at stage 0.1) |

### Staging

| Stage | Scope | Status |
|---|---|---|
| **0** | Load both models, prove multi-image + high-res, measure the OOM boundary | **done** — see [docs/qwen-vs-internvl.md](docs/qwen-vs-internvl.md) |
| **0.1** | vLLM serving path (isolated venv) | not started |
| **0.2** | MMMU + MathVista through **both** VLMEvalKit and lmms-eval, gated against published numbers | implemented, not yet run |

---

## Quick start

### Stage 0 — does it run, and where does it break?

```bash
bash setup_colab.sh                 # creates config.yaml, installs deps, probes the GPU
$EDITOR config.yaml                 # set your paths — nothing is guessed for you
# copy your held-out images into paths.images_dir, then:
python -m mmro.images               # confirms the set is usable
bash scripts/00_fetch_weights.sh    # resumable download + staging
bash scripts/01_smoke.sh            # one subprocess per model
bash scripts/02_sweep.sh            # measures the OOM boundary
python -m mmro.report               # -> results/STAGE0.md
```

Read `results/STAGE0.md`. Criterion 2 — captions being *on topic* — cannot be
checked by a script and is the one that catches a silently broken vision path.

To find where a model stops *understanding* rather than where it stops fitting,
run every rung instead of stopping at the first success:

```bash
python -m mmro.sweep --all-rungs --model internvl3_8b_hf \
  --set 'sweep.image_counts=[3]'
```

### Stage 0.2 — do our numbers match the published ones?

```bash
bash scripts/03_setup_harness.sh --archive   # clones + two isolated venvs, saved to Drive
# then, before running:
#   1. verify each eval.harnesses.<name>.command against that project's README
#   2. verify the models.<name>.baselines values against the official model card
#   3. set eval.judge.* — see the blockers below
python -m mmro.eval --set eval.enabled=true --set eval.limit=25   # shakeout, minutes
bash scripts/04_eval.sh             # the real run — hours
```

**The venvs must be on local disk, not Drive.** A virtualenv needs symlinks and
`ensurepip` needs normal file semantics; Drive's FUSE mount has neither, so
`python -m venv` fails. (A `git clone` there succeeds, because it only writes
regular files — which is why the failure surfaces halfway through setup.) The
defaults point `venv` and `repo_dir` at `/content/MMRM/...` while `work_dir`
stays on Drive so results survive; the script probes for symlink support and
stops with that explanation before installing anything.

Local disk is wiped between Colab sessions, so `--archive` tars each built venv
(~5GB, mostly torch) to Drive. Later runs restore in minutes instead of
reinstalling. Venvs are not relocatable, which is why the venv paths are pinned
absolutely in config rather than derived from the repo location.

`mmro.eval` runs each task on each enabled harness, then checks the gate and
exits non-zero if it fails. `--gate-only` re-checks existing scores without
re-running anything.

---

## Stage 0.2: the gate

**The gate:** reproduce published Qwen2.5-VL-7B numbers within
`eval.gate_tolerance` (default ±2 absolute points). This validates the
*plumbing*. If we cannot reproduce a known result, no later number from this
pipeline means anything.

A failed gate is a finding about our setup, not about the model. `mmro.eval`
prints the likely causes in order of how often they are actually to blame:

### Three things that will cost you the gate

**1. The judge.** MathVista and MMMU's open-ended items need LLM answer
extraction, and the published numbers were produced *with* it. Leaving
`eval.judge.backend: none` falls back to rule-based matching, which
systematically under-scores — routinely by more than 2 points. Set this first.

**2. The per-task visual budget.** Stage 0's budget was sized for a 7-image worst
case. MathVista is single-image; giving it that budget downscales every chart for
no reason. So `eval.tasks[].max_images` divides each model's measured
`safe_total_visual_tokens` into a per-image budget:

| task | `max_images` | Qwen budget/image |
|---|---|---|
| `mathvista_testmini` | 1 | 20,000 → effectively native resolution |
| `mmmu_val` | 7 | 2,857 |

**3. Split mismatch.** VLMEvalKit's `MMMU_DEV_VAL` spans **dev+validation**;
lmms-eval's `mmmu_val` is **validation only**. That is a real difference between
the harnesses and the most likely source of an MMMU delta — not a bug.

### Which model implementation runs

For the gate, each harness runs **its own** registered Qwen2.5-VL class
(`models.<name>.harness_ids`), because that is what produced the published
numbers. Substituting our adapter through the seam adds a variable and belongs in
a separate comparison, *after* the gate passes.

### Verify before you trust

Two things are config values I could not check from this machine, both marked in
`config.example.yaml`:

- **`models.<name>.baselines`** — the published scores the gate compares against,
  per model. A wrong baseline makes the gate noise in either direction; an
  unfilled (null) one makes the gate report "no baseline".
- **`eval.harnesses[].command`** — both projects change their CLI between
  releases. It is a template string in config precisely so a change there means
  editing config, not patching code.

`scripts/03_setup_harness.sh` writes `results/{harness}_commit.txt` pinning each
clone's SHA, since unpinned harnesses will not reproduce these scores later.

---

## Configuration

`config.yaml` is the single file you edit; `config.example.yaml` is the committed
template. **If a value might change between runs, it is in there and not in
`src/`** — model registry, paths, pixel/tile budgets, sweep ladders, decoding,
harness commands, task mapping, baselines, and the gate tolerance.

```bash
python -m mmro.config --validate                       # print the resolved config
python -m mmro.smoke --model qwen2_5_vl_7b \
  --set paths.cache_mode=local                         # CLI override
MMRO_DECODE__MAX_NEW_TOKENS=64 python -m mmro.smoke ...  # env override
```

Precedence: **CLI `--set` > `MMRO_*` env var > `config.yaml` > defaults.**

Unknown keys are hard errors. A typo that silently reverted to a default would
invalidate a whole sweep or benchmark run, and you would only find out by
disbelieving the results. `bash scripts/check_config_control.sh` enforces that
repo IDs and environment paths appear nowhere in `src/`.

### Paths are yours to set

Nothing mounts Google Drive for you and no path is guessed. Mount Drive yourself,
then point `paths.hf_home` at it. Both models together are ~33GB, so the free
15GB Drive tier is not enough — use `cache_mode: local` if that is what you have.
Weights are always *staged* to local disk before loading, because mmap-ing
multi-shard safetensors across Drive's FUSE mount stalls (six minutes per load
versus thirty seconds). `python -m mmro.cache --status` reports which you are
getting.

### Held-out images

Copy them into `paths.images_dir` yourself; see
[assets/heldout/README.md](assets/heldout/README.md). Used in sorted filename
order, so put a dense-text or multi-object image first — those make a broken
vision path visible, where an open scene invites plausible confabulation.

**Use 4K+ originals.** `images.min_long_side` (default 2048) is enforced: below
it, budget rungs stop binding and the sweep would report a boundary it never
tested. `01_smoke.sh` warns; `02_sweep.sh` refuses without `--allow-small-images`.

---

## Measured results

[**docs/qwen-vs-internvl.md**](docs/qwen-vs-internvl.md) — how each model turns
pixels into tokens (with worked arithmetic), the VRAM law measured on the L4, and
the stage-0 finding that matters most:

> **Fitting in memory is not the same as still understanding the images.**
> InternVL3's usable multi-image limit is ~47% of its VRAM limit. Past it the
> model silently drops an image and describes the rest as if that were all it was
> given — no OOM, no error. Qwen2.5-VL has no such cliff; its usable limit is its
> VRAM limit.

Read it before setting any benchmark budget.

## What the sweep measures, and why

"Full resolution without OOM" is not a yes/no question on a 24GB L4 — it depends
on how many images share the budget. The sweep walks a descending ladder per
(model, image count) and reports the highest rung that produces a caption.

Rungs that would measure the same thing for your images are skipped and logged:
InternVL's tiling is quantized by aspect ratio, so `max_tiles` 3–7 all produce
the same 3 tiles for a 16:9 image. If a cell survives at the *top* rung the sweep
says so and exits non-zero, because that is the top of a list we chose rather
than a limit of the hardware.

Every cell runs in a **fresh subprocess**. Two 16.5GB models will not coexist in
one process on a 24GB L4, and `empty_cache()` does not reliably return fragmented
VRAM. This is load-bearing, not caution.

---

## Layout

```
config.example.yaml       the template; config.yaml is yours and gitignored
setup_colab.sh            bootstrap (never reinstalls torch)
scripts/                  00_fetch_weights  01_smoke  02_sweep
                          03_setup_harness  04_eval  check_config_control
docs/qwen-vs-internvl.md  how the two models see, and where each breaks
src/mmro/
  config.py               load + validate + override precedence
  env.py  cache.py        HF cache, resumable fetch, local staging
  images.py budget.py     held-out set; visual-token estimators (pure, CPU-tested)
  vram.py records.py      peak-memory tracing; shared result schemas
  models/                 adapter seam + Qwen2.5-VL and InternVL3 adapters
  harness/                VLMEvalKit + lmms-eval backends (isolated venvs)
  smoke.py sweep.py       stage-0 GPU entrypoints
  eval.py                 stage-0.2 runner + gate
  report.py scaffold.py   STAGE0.md; PEFT/FSDP readiness check
tests/                    191 CPU-only tests, no GPU or weights required
```

## The seam that matters

VLMEvalKit's `generate_inner` and lmms-eval's `generate_until` both reduce to one
call, so stage 0 was written against it directly:

```python
adapter.generate(images: list[Path], text: str, decode) -> str
```

`generate_measured` is the same code path with telemetry — peak VRAM, timings,
and the visual-token count read back from the model's own tensors rather than
estimated.

`ScoreRecord` carries both `decode` **and** `budget`, so a score gap between the
two harnesses cannot be silently caused by them defaulting to different
resolutions or sampling. The report warns when they disagree.

## Tests

```bash
python -m pytest tests/ -q          # 191 tests, CPU-only
```

Config validation and override precedence, budget estimators, image-set failure
modes, record round-trips, weight-staging (with a real symlink farm, which is how
the Drive staging bug was caught), sweep boundary logic, harness result parsing
against fixtures shaped like the real output, and the gate. GPU behaviour —
loading, captioning, OOM — is by definition only verifiable on the L4.
