# Held-out images

Copy your held-out images into this directory manually. Nothing downloads or
generates them — if this directory is empty, the smoke and sweep scripts stop
with an error.

Accepted: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

## Ordering matters

Images are used in **sorted filename order**, so prefix them to control which
one a single-image test sees:

```
01_dense_text.jpg      <- receipt, sign, chart, page of text
02_multi_object.jpg    <- several countable, distinctly-colored objects
03_scene.jpg
04_scene.jpg
```

Put a dense-text or multi-object image first. A model whose vision path has
silently broken still produces fluent, plausible captions for an open scene —
those two kinds make the failure visible, because the caption either quotes the
text and counts the objects or it does not.

## Resolution — this is the part that actually matters

Use the largest originals you have. **4K+ on the long side is the target**;
`images.min_long_side` in `config.yaml` sets the enforced floor (default 2048).

Small images do not merely weaken the result, they invert it. Once an image is
smaller than a rung's pixel budget, the rung stops binding: the model sees the
image at native resolution no matter how high the budget goes. The sweep then
reports the top of the ladder as "surviving" when nothing was ever tested, and
that number becomes stage 0.2's token budget.

Concretely, a 960×1139 image saturates at ~1395 visual tokens, so every Qwen
rung above that measures the same thing — and with ~6GB of VRAM headroom left
unused, nothing OOMs and the sweep finds no boundary at all.

So:

- `01_smoke.sh` runs on small images but prints a NOTE that it is testing the
  pipeline, not resolution limits.
- `02_sweep.sh` **refuses** to run below the floor, unless you pass
  `--allow-small-images` and accept a boundary that is not a real limit.

## How many

Enough to cover the largest count in `config.yaml` — by default `sweep.image_counts`
tops out at 4. Check with:

```bash
PYTHONPATH=src python -m mmro.images
```

Image files here are gitignored; only this README is tracked.
