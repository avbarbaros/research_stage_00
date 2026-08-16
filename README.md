# research_stage_00

Multimodal reasoning models — **Stage 0**.

Before any fine-tuning is meaningful, establish that the two candidate backbones
actually run on the target hardware with the inputs the research needs:
**multi-image prompts at high resolution**.

| | |
|---|---|
| Models | `Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3-8B-hf` (+ native fallback) |
| Target | Google Colab terminal, **L4 24GB**, weights cached on Google Drive |
| Stack | PyTorch · Transformers · PEFT · FSDP (vLLM at stage 0.1) |

**Stage 0 succeeds when** both models caption held-out images sensibly, handle
≥3 images in one turn, and we know precisely where the OOM boundary sits.

### Staging

| Stage | Scope |
|---|---|
| **0** | Load both models, prove multi-image + high-res, measure the OOM boundary |
| **0.1** | vLLM serving path (isolated venv) |
| **0.2** | MMMU + MathVista through **both** VLMEvalKit and lmms-eval, scores compared |

Stage 0 ships the *seams* 0.1 and 0.2 plug into — it does not integrate them.

---

## Quick start

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

## Configuration

`config.yaml` is the single file you edit; `config.example.yaml` is the
committed template. **If a value might change between runs, it is in there and
not in `src/`** — model registry, paths, pixel/tile budgets, sweep ladders,
decoding, and the stage-0.2 harness and task mapping.

```bash
python -m mmro.config --validate                       # print the resolved config
python -m mmro.smoke --model qwen2_5_vl_7b \
  --set paths.cache_mode=local                         # CLI override
MMRO_DECODE__MAX_NEW_TOKENS=64 python -m mmro.smoke ...  # env override
```

Precedence: **CLI `--set` > `MMRO_*` env var > `config.yaml` > defaults.**

Unknown keys are hard errors. A typo that silently reverted to a default would
invalidate a whole sweep, and you would only find out by disbelieving the
results. `bash scripts/check_config_control.sh` enforces that repo IDs and
environment paths appear nowhere in `src/`.

### Paths are yours to set

Nothing mounts Google Drive for you and no path is guessed. Mount Drive
yourself, then point `paths.hf_home` at it. Both models together are ~33GB, so
the free 15GB Drive tier is not enough — use `cache_mode: local` if that is what
you have. Weights are always *staged* to local disk before loading, because
mmap-ing multi-shard safetensors across Drive's FUSE mount stalls.

### Held-out images

Copy them into `paths.images_dir` yourself; see
[assets/heldout/README.md](assets/heldout/README.md). Used in sorted filename
order, so put a dense-text or multi-object image first — those make a broken
vision path visible, where an open scene invites plausible confabulation.

**Use 4K+ originals.** `images.min_long_side` (default 2048) is enforced: below
it, budget rungs stop binding and the sweep would report a boundary it never
tested. `01_smoke.sh` warns; `02_sweep.sh` refuses without `--allow-small-images`.

## What the sweep measures, and why

"Full resolution without OOM" is not a yes/no question on a 24GB L4 — it depends
on how many images share the budget. Qwen2.5-VL inherits a **16,384 visual token
per image** ceiling from `qwen-vl-utils`, roughly 13× our default budget, against
about 6GB of headroom once weights are loaded (see the module docstring in
[budget.py](src/mmro/budget.py) for the arithmetic).

So the sweep walks a descending ladder per (model, image count) and reports the
highest rung that produces a caption. Rungs that would measure the same thing
for your images are skipped and logged — InternVL's tiling is quantized by
aspect ratio, so for 16:9 images `max_tiles` 12 and 8 both settle on the same
grid.

Every cell runs in a **fresh subprocess**. Two 16.5GB models will not coexist in
one process on a 24GB L4, and `empty_cache()` does not reliably return
fragmented VRAM. This is load-bearing, not caution.

## Layout

```
config.example.yaml       the template; config.yaml is yours and gitignored
setup_colab.sh            bootstrap (never reinstalls torch)
scripts/                  00_fetch_weights, 01_smoke, 02_sweep, check_config_control
src/mmro/
  config.py               load + validate + override precedence
  env.py  cache.py        HF cache, resumable fetch, local staging
  images.py budget.py     held-out set; visual-token estimators (pure, CPU-tested)
  vram.py records.py      peak-memory tracing; shared result schemas
  models/                 adapter seam + Qwen2.5-VL and InternVL3 adapters
  harness/                stage-0.2 backend protocol (stubs until then)
  smoke.py sweep.py       the two GPU entrypoints
  report.py scaffold.py   STAGE0.md; PEFT/FSDP readiness check
tests/                    136 CPU-only tests, no GPU or weights required
```

## The seam that matters

VLMEvalKit's `generate_inner` and lmms-eval's `generate_until` both reduce to one
call, so stage 0 is written against it directly:

```python
adapter.generate(images: list[Path], text: str, decode) -> str
```

`generate_measured` is the same code path with telemetry — peak VRAM, timings,
and the visual-token count read back from the model's own tensors rather than
estimated. `tests/test_seam.py` proves the contract, the task-ID mapping
(`mathvista_testmini` → `MathVista_MINI` / `mathvista_testmini`), and that
`mmro.models` imports without torch, since the harness venvs will import it.

## Tests

```bash
python -m pytest tests/ -q          # 136 tests, CPU-only
```

They cover config validation and override precedence, the budget estimators, the
image set's failure modes, record round-trips, sweep boundary logic, and the
stage-0.2 seam. GPU behaviour — loading, captioning, OOM — is by definition only
verifiable on the L4.
