"""Visual-token budgeting for Qwen2.5-VL and InternVL3.

These are *pure* functions with no torch dependency, so they are unit-tested on
CPU and used to **choose** settings before any VRAM is allocated. The adapters
report the token count actually observed; ``sweep.py`` compares the two. A
mismatch means this model of the preprocessing is wrong, and we want to find
that out here rather than halfway through a benchmark run.

Why this module exists at all
-----------------------------
The per-image token budget must be set explicitly, because the default we would
otherwise inherit is far above what a 24GB L4 can serve.

One Qwen2.5-VL visual token covers a 28x28 px block (14px patch, 2x2 merge), so
784 px/token. There are two different defaults in play:

* ``qwen_vl_utils.MAX_PIXELS`` = ``16384 * 28 * 28`` = 12,845,056 px
  -> **16,384 tokens** for a single image
* ``transformers.Qwen2VLImageProcessor`` default ``max_pixels`` = ``28*28*1280``
  -> 1,280 tokens

We preprocess via ``process_vision_info`` from qwen-vl-utils, so the 16,384
ceiling is the one our pipeline inherits unless we pass ``min_pixels`` /
``max_pixels`` ourselves. (The lower figure is where our 1280 default came from.)

For scale on a 24GB L4: weights take ~16.6GB of ~22.5GB usable, leaving ~6GB.
At 16,384 merged tokens the ViT processes 65,536 patches; LLM KV is only ~0.9GB
at that length and a single ViT activation tensor ~0.16GB, so no individual
allocation is decisive -- the pressure is cumulative transient activation across
the vision tower, and it grows linearly with the budget. Where it actually
breaks is what ``sweep.py`` measures; this module only makes the budget an
explicit, defensible number instead of an inherited ceiling. The sweep ladder
tops out at 4,096 rather than 16,384 for the same reason.
"""

from __future__ import annotations

import math

# --- Qwen2.5-VL --------------------------------------------------------------
# ViT patch is 14px; a 2x2 merge means one visual token covers 28x28 px.
QWEN_PATCH = 14
QWEN_MERGE = 2
QWEN_PIXELS_PER_TOKEN = (QWEN_PATCH * QWEN_MERGE) ** 2  # 784
# The processor rounds each side to a multiple of 28, so image area is quantized.
QWEN_SIDE_FACTOR = QWEN_PATCH * QWEN_MERGE  # 28

# --- InternVL3 ---------------------------------------------------------------
# 448px tiles; each tile is 32x32 patches, pixel-shuffled 4:1 -> 256 tokens.
INTERNVL_TILE_PX = 448
INTERNVL_TOKENS_PER_TILE = 256


# =============================================================================
# Qwen2.5-VL
# =============================================================================
def qwen_max_pixels(max_visual_tokens_per_image: int) -> int:
    """Convert a per-image token budget into the processor's ``max_pixels``."""
    if max_visual_tokens_per_image < 1:
        raise ValueError("max_visual_tokens_per_image must be >= 1")
    return max_visual_tokens_per_image * QWEN_PIXELS_PER_TOKEN


def qwen_smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = QWEN_SIDE_FACTOR,
) -> tuple[int, int]:
    """Reproduce Qwen2.5-VL's ``smart_resize``: the dimensions the model sees.

    Both sides are rounded to a multiple of ``factor``, then the image is scaled
    to land inside ``[min_pixels, max_pixels]`` while preserving aspect ratio.
    """
    if height < 1 or width < 1:
        raise ValueError(f"invalid image size: {height}x{width}")

    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)

    if h * w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor

    return h, w


def qwen_visual_tokens(
    height: int,
    width: int,
    max_visual_tokens_per_image: int,
    min_pixels: int = 3136,
) -> int:
    """Visual tokens a single image will occupy after preprocessing."""
    h, w = qwen_smart_resize(
        height, width, min_pixels, qwen_max_pixels(max_visual_tokens_per_image)
    )
    return (h // QWEN_SIDE_FACTOR) * (w // QWEN_SIDE_FACTOR)


def qwen_total_visual_tokens(
    sizes: list[tuple[int, int]],
    max_visual_tokens_per_image: int,
    min_pixels: int = 3136,
) -> int:
    """Visual tokens for a multi-image prompt. ``sizes`` is [(h, w), ...]."""
    return sum(
        qwen_visual_tokens(h, w, max_visual_tokens_per_image, min_pixels) for h, w in sizes
    )


# =============================================================================
# InternVL3
# =============================================================================
def internvl_tiles_for_image(
    height: int,
    width: int,
    max_tiles: int,
    use_thumbnail: bool = True,
    tile_px: int = INTERNVL_TILE_PX,
) -> int:
    """Tiles InternVL3's dynamic preprocessing will produce for one image.

    It searches candidate (cols, rows) grids with ``cols * rows <= max_tiles``
    and picks the one whose aspect ratio is closest to the image's, preferring
    the larger grid on a tie when the image is big enough to justify it. A
    thumbnail tile is appended when the chosen grid has more than one tile.
    """
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")
    if height < 1 or width < 1:
        raise ValueError(f"invalid image size: {height}x{width}")

    target = width / height
    candidates = sorted(
        {
            (cols, rows)
            for n in range(1, max_tiles + 1)
            for cols in range(1, n + 1)
            for rows in range(1, n + 1)
            if cols * rows <= max_tiles
        },
        key=lambda cr: cr[0] * cr[1],
    )

    best = (1, 1)
    best_diff = float("inf")
    for cols, rows in candidates:
        diff = abs(cols / rows - target)
        if diff < best_diff:
            best_diff, best = diff, (cols, rows)
        elif diff == best_diff:
            # Tie: prefer the larger grid only when the image has the pixels to
            # fill it, matching InternVL's own preference rule.
            if width * height > 0.5 * tile_px * tile_px * cols * rows:
                best = (cols, rows)

    tiles = best[0] * best[1]
    if use_thumbnail and tiles > 1:
        tiles += 1
    return tiles


def internvl_visual_tokens(
    height: int,
    width: int,
    max_tiles: int,
    use_thumbnail: bool = True,
) -> int:
    return internvl_tiles_for_image(height, width, max_tiles, use_thumbnail) * (
        INTERNVL_TOKENS_PER_TILE
    )


def internvl_total_visual_tokens(
    sizes: list[tuple[int, int]],
    max_tiles: int,
    use_thumbnail: bool = True,
) -> int:
    return sum(internvl_visual_tokens(h, w, max_tiles, use_thumbnail) for h, w in sizes)


# =============================================================================
# Dispatch
# =============================================================================
def estimate_visual_tokens(
    adapter: str,
    sizes: list[tuple[int, int]],
    budget: dict,
) -> int:
    """Estimate total visual tokens for ``sizes`` under an adapter's budget."""
    if adapter == "qwen25vl":
        return qwen_total_visual_tokens(
            sizes,
            budget.get("max_visual_tokens_per_image", 1280),
            budget.get("min_pixels", 3136),
        )
    if adapter in ("internvl3_hf", "internvl3_native"):
        return internvl_total_visual_tokens(
            sizes,
            budget.get("max_tiles", 6),
            budget.get("use_thumbnail", True),
        )
    raise ValueError(f"no token estimator for adapter {adapter!r}")


def budget_at_rung(adapter: str, budget: dict, rung: int) -> dict:
    """A copy of ``budget`` with the sweep's ladder value applied.

    Each adapter family has a different knob, so the sweep stays generic and the
    per-family mapping lives here.
    """
    out = dict(budget)
    if adapter == "qwen25vl":
        out["max_visual_tokens_per_image"] = rung
    elif adapter in ("internvl3_hf", "internvl3_native"):
        out["max_tiles"] = rung
    else:
        raise ValueError(f"no sweep knob for adapter {adapter!r}")
    return out


def ladder_for(adapter: str, sweep_cfg) -> list[int]:
    """The configured descending ladder for an adapter family."""
    if adapter == "qwen25vl":
        return sweep_cfg.qwen_visual_token_ladder
    if adapter in ("internvl3_hf", "internvl3_native"):
        return sweep_cfg.internvl_tile_ladder
    raise ValueError(f"no ladder for adapter {adapter!r}")


def distinct_rungs(
    adapter: str,
    budget: dict,
    sizes: list[tuple[int, int]],
    ladder: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Drop ladder rungs that are indistinguishable for *these* images.

    Different rungs can produce identical token counts. InternVL's tiling is
    quantized by aspect ratio, so for a 16:9 image ``max_tiles`` of 12 and 8 both
    settle on a 4x2 grid plus thumbnail -- the same 9 tiles. Measuring both wastes
    a GPU cell and, worse, an OOM at the higher rung guarantees an OOM at the
    lower one, so the sweep would descend through rungs that cannot possibly
    change the outcome.

    Whether rungs collapse depends on the images, which is why this is decided
    here at run time rather than by trying to pick a universally-distinct ladder
    in config.

    Keeps the **highest** rung of each duplicate group: it is the same
    computation, so the higher number is the honest boundary to report.

    Returns ``(kept, skipped)`` where kept is ``[(rung, estimate)]`` and skipped
    is ``[(rung, estimate, kept_rung)]`` -- callers are expected to log the
    skips rather than silently narrowing the sweep.
    """
    kept: list[tuple[int, int]] = []
    skipped: list[tuple[int, int, int]] = []
    seen: dict[int, int] = {}  # estimate -> the rung we kept for it

    for rung in ladder:
        estimate = estimate_visual_tokens(adapter, sizes, budget_at_rung(adapter, budget, rung))
        if estimate in seen:
            skipped.append((rung, estimate, seen[estimate]))
            continue
        seen[estimate] = rung
        kept.append((rung, estimate))

    return kept, skipped
