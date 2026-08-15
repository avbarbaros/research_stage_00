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

## Resolution

Stage 0 measures the OOM boundary at high resolution, so use the largest
originals you have (≥4K on the long side is ideal). Images under 1024px on the
long side are flagged as advisory: they still run, but they do not exercise the
budget ladder and so are not evidence of anything.

## How many

Enough to cover the largest count in `config.yaml` — by default `sweep.image_counts`
tops out at 4. Check with:

```bash
PYTHONPATH=src python -m mmro.images
```

Image files here are gitignored; only this README is tracked.
