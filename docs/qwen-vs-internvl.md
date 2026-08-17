# Qwen2.5-VL vs InternVL3 — how they see, and where they break

Everything in this document was measured in stage 0 on a **Colab L4 (22.03 GiB
usable)** with `Qwen/Qwen2.5-VL-7B-Instruct` and `OpenGVLab/InternVL3-8B-hf`,
both in bfloat16 with SDPA attention. Raw data is in `results/*.jsonl`; the
generated summary is `results/STAGE0.md`.

The held-out set:

| file | dimensions | megapixels |
|---|---|---|
| `00_IC_342_Hidden_Galaxy.png` | 7985 × 8409 | 67.15 |
| `01_Signing_of_the_Declaration_4K.jpg` | 3840 × 2547 | 9.78 |
| `02_Tuerlersee_helicopter_4K.jpg` | 4608 × 2592 | 11.94 |
| `03_Sakura_Kameyama_Park.jpg` | 5218 × 2928 | 15.28 |

---

## 1. The core difference in one sentence

**Qwen shrinks the image. InternVL cuts it into pieces.**

Everything else — how budgets behave, why one has a smooth response curve and
the other a staircase, why one degrades gracefully and the other falls off a
cliff — follows from that.

| | Qwen2.5-VL | InternVL3 |
|---|---|---|
| Strategy | resize to fit a pixel budget | tile into 448×448 crops |
| Budget knob | `max_visual_tokens_per_image` | `max_tiles` |
| Token cost | varies continuously with pixels | 256 per tile, always |
| Response to the knob | smooth | staircase |
| Sees the whole image | yes, at lower resolution | yes, via a thumbnail tile |
| Sees fine detail | no, it was scaled away | yes, within each crop |

An analogy: Qwen is looking at a photograph from across the room. InternVL is
looking at it through a magnifying glass, one square at a time, plus one glance
at the whole thing.

---

## 2. Qwen — resize to a pixel budget

### 2.1 The unit

```
ViT patch size    = 14 px
2×2 merge         → one visual token covers 28 × 28 px
                  = 784 pixels per visual token
```

A "rung" is a token allowance converted to a pixel allowance:

```
rung 8192 → max_pixels = 8192 × 784 = 6,422,528 px
```

That is the *only* thing the rung does: cap pixels per image.

### 2.2 Worked example — the Declaration at rung 8192

**a. Is it over budget?**
```
3840 × 2547 = 9,780,480 px   vs   6,422,528 allowed   → 1.52× over
```

**b. Snap each side to a multiple of 28** (the model only sees whole blocks):
```
height: 2547 / 28 = 90.9643  → round 91  → 91 × 28 = 2548
width:  3840 / 28 = 137.1429 → round 137 → 137 × 28 = 3836
```

**c. Shrink — and here is why it's a square root.** You need to cut *area* by
1.52×, but you can only change *sides*. Area is the product of two sides, so
dividing each side by `s` cuts area by `s²`. To hit an area ratio R you need
`s = √R`:
```
R = 9,780,480 / 6,422,528 = 1.522840
s = √1.522840             = 1.234034

height: 2547 / 1.234034 = 2063.963 px
width:  3840 / 1.234034 = 3111.746 px
```
Check: 2063.963 × 3111.746 = 6,422,433 px ≈ the budget. ✓

**d. Snap again — with floor, not round.** Rounding up could push you back over
budget; flooring guarantees you land under it:
```
height: 2063.963 / 28 = 73.7130  → floor 73  → 2044 px
width:  3111.746 / 28 = 111.1338 → floor 111 → 3108 px
```

**e. Count blocks:**
```
(2044 / 28) × (3108 / 28) = 73 × 111 = 8,103 tokens
```

### 2.3 All three images at rung 8192

```
galaxy       7985×8409 → 2464×2576 → 92 × 88 = 8,096
declaration  3840×2547 → 3108×2044 → 73 × 111 = 8,103
helicopter   4608×2592 → 3360×1876 → 67 × 120 = 8,040
                                              ─────────
                                                24,239
```
The sweep recorded exactly **24,239**.

### 2.4 Why none reach 8,192

Two truncations in step (d). `73.7130 → 73` discards 0.71 of a row; `111.1338 →
111` discards 0.13 of a column. The galaxy loses most (8,096) because it is
nearly square: a 92×88 grid has fewer, larger cells, so each floored fraction
costs proportionally more.

### 2.5 A budget only binds if the image has the pixels

The galaxy at native resolution would be `67,145,865 / 784 = 85,645` tokens.
Every rung we tested is below that, so the budget always bound:

| rung | max_pixels | tokens |
|---|---|---|
| 40960 | 32,112,640 | 40,779 |
| 20480 | 16,056,320 | 20,294 |
| 10240 | 8,028,160 | 10,094 |
| 8192 | 6,422,528 | 8,096 |
| 1280 | 1,003,520 | 1,224 |

A 4K image (3840×2160) is only **10,549 tokens** native, so rungs above that do
nothing to it. This is why an early sweep on 1 MP images was meaningless: the
top rungs never bound, nothing was stressed, and nothing OOM'd.

---

## 3. InternVL — tile into fixed crops

### 3.1 The unit

```
tile              = 448 × 448 px
ViT patch         = 14 px
patches per side  = 448 / 14 = 32
patches per tile  = 32 × 32  = 1024
pixel shuffle 2×2 = ÷ 4
tokens per tile   = 1024 / 4 = 256
```

**This is fixed.** Image size never changes what a tile costs. The only variable
is *how many tiles*:

```
tokens = tile_count × 256
```

### 3.2 The grid search

For each image, InternVL enumerates every `cols × rows` grid with
`cols × rows ≤ max_tiles` and picks the one whose **aspect ratio** is closest to
the image's. At `max_tiles: 40` there are 158 candidates.

**Declaration** — 3840 × 2547, aspect 1.5077:
```
  6×4 = 24 tiles   ratio 1.5000   diff 0.0077   ← chosen
  3×2 =  6 tiles   ratio 1.5000   diff 0.0077
  8×5 = 40 tiles   ratio 1.6000   diff 0.0923
```
6×4 and 3×2 are *equally good* on aspect — both are exactly 3:2. A tie-break
prefers the larger grid when the image has enough pixels to fill it, so 24 wins.

**Galaxy** — 7985 × 8409, aspect 0.9496 — shows this most starkly:
```
  6×6 = 36 tiles   ratio 1.0000   diff 0.0504   ← chosen
  5×5 = 25 tiles   ratio 1.0000   diff 0.0504
  4×4 = 16 tiles   ratio 1.0000   diff 0.0504
  3×3 =  9 tiles   ratio 1.0000   diff 0.0504
  2×2 =  4 tiles   ratio 1.0000   diff 0.0504
```
Every square grid ties. Aspect ratio alone cannot choose, so **the tie-break
does all the work** and takes the largest square that fits.

> **Consequence for MMMU:** near-square images (diagrams, charts, screenshots)
> tie across every square grid and grab the maximum available tiles. Budget for
> the squarest image, not the average one.

### 3.3 The thumbnail

When the chosen grid has more than one tile, InternVL appends a downscaled copy
of the whole image, so the model sees global context alongside the crops.

### 3.4 Four images at rung 40

```
galaxy       6×6 grid = 36 + 1 thumbnail = 37 tiles →  9,472 tok
declaration  6×4 grid = 24 + 1           = 25 tiles →  6,400 tok
helicopter   7×4 grid = 28 + 1           = 29 tiles →  7,424 tok
sakura       7×4 grid = 28 + 1           = 29 tiles →  7,424 tok
                                          ─────────────────────
                                          120 tiles → 30,720 tok
```
The sweep recorded exactly **30,720**.

Note tiles are **not** split evenly — 37/25/29/29. Helicopter (1.7778) and
sakura (1.7821) land on the same grid because their aspect ratios nearly match,
despite different dimensions.

### 3.5 The staircase

A 16:9 image as `max_tiles` rises:

| max_tiles | tiles used | tokens | |
|---|---|---|---|
| 1 | 1 | 256 | |
| 2 | 3 | 768 | jump |
| 3–7 | 3 | 768 | **flat — five wasted rungs** |
| 8 | 9 | 2,304 | jump |
| 9–12 | 9 | 2,304 | flat |
| 16 | 16 | 4,096 | jump |
| 24 | 16 | 4,096 | flat |
| 28 | 29 | 7,424 | jump |
| 32–40 | 29 | 7,424 | flat |

Qwen, same image, moves on every rung: 299 → 594 → 1,222 → 2,479 → 4,080 →
6,032 → 8,040 → 10,050.

This is why the sweep deduplicates rungs at run time (`budget.distinct_rungs`)
rather than relying on a well-chosen ladder: where the flat treads fall depends
on each image's aspect ratio, so no fixed ladder avoids them.

---

## 4. The VRAM law

Fitting peak VRAM against total visual tokens across all successful cells:

| | fixed cost | per 1,000 tokens | extrapolated ceiling |
|---|---|---|---|
| **Qwen** | 15.46 GB | 0.212 GB | 31,037 tokens |
| **InternVL** | 14.81 GB | 0.202 GB | 35,753 tokens |

Read it as: **weights are rent, tokens are metered usage.** ~15 GB is gone
before a single pixel is shown; the remaining ~7 GB buys roughly **5,000 visual
tokens per gigabyte**.

Bracketing with the actual failures:

- **Qwen**: works at 24,239 · fails at 30,312 → true limit between them
- **InternVL**: works at 30,720 · fails at 39,424 → true limit between them

The linear prediction (31,037) lands just above Qwen's measured failure
(30,312). Model and measurement agree.

### 4.1 The boundary is a total, not a per-image number

| | 1 img | 2 img | 3 img | 4 img |
|---|---|---|---|---|
| Qwen rung | 20480 | 10240 | 8192 | 6144 |
| **Qwen total tokens** | 20,294 | 20,262 | 24,239 | 24,192 |
| InternVL rung | 128 | 72 | 48 | 40 |
| **InternVL total tokens** | 28,416 | 30,720 | 27,648 | 30,720 |

The per-image rung falls 3–4× as images are added, but the total stays flat.
**The GPU does not care how you slice the pie, only how big it is.** Budget the
total, then divide by worst-case image count.

### 4.2 The estimator is exact

`budget.py` reproduced **every** recorded token count — 16/16 for InternVL and
all Qwen cells — from image dimensions alone, on CPU, with no GPU involved.
Ranges from 256 to 160,768 tokens. You can compute a budget for a new dataset or
a new GPU without running anything.

---

## 5. The finding that matters most: fitting ≠ understanding

The sweep measures **VRAM**. Running `--all-rungs` (every rung, not just until
the first success) measures **comprehension**. They are not the same limit.

### 5.1 InternVL has a cliff at ~14,336 tokens

Across 31 multi-image cells, judged by whether the caption correctly identifies
all the images given:

```
      512 → 14,336 tokens    all correct  (22 cells)
           14,848 tokens     WRONG         ← first failure
           15,872 tokens     correct       ← one outlier above the failure
      18,944 → 30,720        all wrong     (6 cells)
```

**VRAM limit 30,720. Usable limit 14,336 — 47% of it.**

The failure is not gradual degradation. Past the cliff an image vanishes
entirely and the model describes the rest as though that were all it received:

> **2 images @ 30,720** (galaxy + painting): *"The two images you provided are
> **both historical paintings** depicting the signing of the United States
> Declaration of Independence…"*

> **3 images @ 14,848**: *"**The two images** depict vastly different scenes…
> First Image: a historical painting…"* — three given, two described, galaxy
> gone.

No OOM. No error. Fluent, confident, wrong.

**Single-image was correct at every rung tested, up to 28,416 tokens.** So this
is about tracking *multiple* images, not resolution alone. A plausible reading:
at 30,720 tokens the model is at 94% of its 32,768 context window, holding 120
separate 448×448 crops, and loses track of which crop belongs to which image.
That mechanism is inferred, not proven.

The 15,872 outlier matters — a 2-image cell that succeeded *above* a 3-image
failure. The threshold is not a clean constant; it interacts with image count.
Treat 14,336 as a bound to stay under, not a line to sit on.

### 5.2 Qwen has no cliff

All 15 multi-image cells correct, up to the VRAM boundary at 24,239 tokens. It
named the galaxy first, in order, every time. **Usable limit = VRAM limit.**

### 5.3 Qwen's style floor

Qwen's output *style* collapses at low resolution, though content stays correct:

| tokens/image | output |
|---|---|
| 1,231 (3 img @ rung 1280) | *"the first image is of a galaxy , the second is of a painting…"* |
| 1,228 (4 img @ rung 1280) | same degenerate lowercase style |
| 2,506 (3 img @ rung 2560) | well-formed prose |
| 2,499 (4 img @ rung 2560) | well-formed prose |

Stay at or above **~2,500 tokens per image**. This matters for benchmarking: an
LLM judge parsing terse fragments is more likely to mis-extract an answer.

---

## 6. Budgets for stage 0.2

MMMU's schema allows up to **7 images** in one question. Budget for that worst
case, not the average — the single 7-image item is what breaks an otherwise
finished run.

| | setting | 7-image worst case | VRAM | binding constraint |
|---|---|---|---|---|
| **Qwen** | `max_visual_tokens_per_image: 2560` | 17,920 tok | ~19.3 GB | VRAM (24,192) + style floor (2,500/img) |
| **InternVL** | `max_tiles: 6` | 12,544 tok | ~17.3 GB | comprehension cliff (14,336) |

Why not higher:

- Qwen at `4096` → 28,672 tokens, above the measured VRAM limit.
- InternVL at `8` → 16,128 tokens, **past the cliff**. The margin at 6 is
  genuinely thin.

Both are set in `config.example.yaml` with the measurements inline.

### 6.1 Budget in tokens, then derive the setting

MMMU's images are far smaller than this 4K/67MP held-out set, so the same
`max_tiles` will produce far fewer tiles there. **The token thresholds transfer;
the settings that achieve them do not.** Recompute with `budget.py` against the
actual dataset dimensions.

### 6.2 Pin both harnesses to the same numbers

`ScoreRecord` carries `decode` **and** `budget` precisely so a score gap between
VLMEvalKit and lmms-eval cannot be silently caused by the two harnesses
defaulting to different resolutions. If one runs at 12 tiles and the other at 6,
the difference you measure is your configuration, not the harness.

---

## 7. Rules of thumb

1. **Budget the total, not the per-image number.** VRAM tracks total visual
   tokens regardless of how they are distributed.
2. **~5,000 visual tokens per GB** of free VRAM, after ~15 GB of weights.
3. **Fitting is not understanding.** Always read captions at the boundary. The
   usable limit can be half the VRAM limit, and the failure is silent.
4. **A budget that exceeds an image's native resolution does nothing.** Check
   `pixels / 784` (Qwen) before assuming a rung binds.
5. **For InternVL, budget for the squarest image**, which will grab the most
   tiles via the tie-break.
6. **For Qwen, keep ≥2,500 tokens per image** or the prose degrades.
7. **Every number here is L4-, dtype-, and inference-specific.** Different GPU,
   different dtype, or training instead of inference voids them.

---

## 8. What is still unknown

- The comprehension threshold was measured with **one prompt** ("Compare these
  images. What differs between them?") on **one image set** with a visually
  distinctive first image. MMMU's prompts and image mix may shift it.
- Where InternVL's cliff sits between 14,336 and 14,848 is bracketed but not
  pinned; and the 15,872 outlier says image count matters too.
- Training peaks are **not** covered. Backward passes store activations, plus
  gradients and optimizer state, so a training sequence length must sit well
  below these inference numbers.
- `internvl3_8b_native` is untested. If InternVL underperforms at stage 0.2,
  that is the control to enable — it matches the published benchmark numbers,
  and this cliff is the kind of thing an `-hf` conversion could introduce.

---

## Reproducing any of this

```bash
python -m mmro.images                       # inspect the held-out set
bash scripts/01_smoke.sh                    # does it work at all?
bash scripts/02_sweep.sh                    # where does it OOM?
python -m mmro.sweep --all-rungs \
  --model internvl3_8b_hf \
  --set 'sweep.image_counts=[3]'            # where does it stop understanding?
python -m mmro.report                       # → results/STAGE0.md
```
