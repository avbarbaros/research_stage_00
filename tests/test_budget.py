"""Budget estimator tests.

These estimators decide settings *before* VRAM is allocated, and the sweep
checks them against what the model actually consumed. If the arithmetic here is
wrong, every rung of the sweep means something other than what it claims, so
the constants get pinned explicitly.
"""

from __future__ import annotations

import pytest

from mmro.budget import (
    INTERNVL_TOKENS_PER_TILE,
    QWEN_PIXELS_PER_TOKEN,
    budget_at_rung,
    distinct_rungs,
    estimate_visual_tokens,
    internvl_tiles_for_image,
    internvl_total_visual_tokens,
    internvl_visual_tokens,
    ladder_for,
    qwen_max_pixels,
    qwen_smart_resize,
    qwen_total_visual_tokens,
    qwen_visual_tokens,
)
from mmro.config import EXAMPLE_CONFIG_PATH, load_config


# --- constants --------------------------------------------------------------
def test_qwen_pixels_per_token():
    # 14px patch, 2x2 merge -> one visual token per 28x28 px block.
    assert QWEN_PIXELS_PER_TOKEN == 784


def test_qwen_max_pixels_round_trips():
    assert qwen_max_pixels(1280) == 1003520
    # The qwen-vl-utils default ceiling, restated: 16384 tokens for one image.
    assert qwen_max_pixels(16384) == 12845056


def test_internvl_tokens_per_tile():
    assert INTERNVL_TOKENS_PER_TILE == 256


# --- qwen smart_resize ------------------------------------------------------
def test_smart_resize_rounds_to_28():
    h, w = qwen_smart_resize(1000, 1000, min_pixels=3136, max_pixels=qwen_max_pixels(4096))
    assert h % 28 == 0 and w % 28 == 0


def test_smart_resize_downscales_past_budget():
    budget = 1280
    h, w = qwen_smart_resize(
        4000, 6000, min_pixels=3136, max_pixels=qwen_max_pixels(budget)
    )
    assert (h // 28) * (w // 28) <= budget


def test_smart_resize_upscales_tiny_images():
    h, w = qwen_smart_resize(16, 16, min_pixels=3136, max_pixels=qwen_max_pixels(1280))
    assert h * w >= 3136


def test_smart_resize_preserves_aspect_roughly():
    h, w = qwen_smart_resize(2160, 3840, min_pixels=3136, max_pixels=qwen_max_pixels(1280))
    assert abs((w / h) - (3840 / 2160)) < 0.1


def test_smart_resize_rejects_degenerate_size():
    with pytest.raises(ValueError):
        qwen_smart_resize(0, 100, 3136, qwen_max_pixels(1280))


# --- qwen token counting ----------------------------------------------------
@pytest.mark.parametrize("budget", [320, 640, 1280, 2560, 4096])
def test_qwen_tokens_never_exceed_budget(budget):
    # A 4K image at every configured rung must stay inside its budget --
    # the whole point of setting max_pixels explicitly.
    assert qwen_visual_tokens(2160, 3840, budget) <= budget


def test_qwen_tokens_increase_with_budget():
    counts = [qwen_visual_tokens(2160, 3840, b) for b in (320, 640, 1280, 2560)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_qwen_multi_image_tokens_are_additive():
    sizes = [(2160, 3840), (1080, 1920), (3000, 3000)]
    total = qwen_total_visual_tokens(sizes, 1280)
    assert total == sum(qwen_visual_tokens(h, w, 1280) for h, w in sizes)


def test_qwen_multi_image_scales_with_count():
    # Multi-image is where the 24GB budget actually breaks; the estimator has to
    # reflect that images do not share a budget.
    one = qwen_total_visual_tokens([(2160, 3840)], 1280)
    four = qwen_total_visual_tokens([(2160, 3840)] * 4, 1280)
    assert four == 4 * one


# --- internvl tiling --------------------------------------------------------
def test_internvl_single_tile_has_no_thumbnail():
    # A square image at max_tiles=1 needs one tile, and a thumbnail would be a
    # duplicate of it.
    assert internvl_tiles_for_image(448, 448, max_tiles=1) == 1


def test_internvl_thumbnail_added_for_multi_tile():
    tiles = internvl_tiles_for_image(2160, 3840, max_tiles=6, use_thumbnail=True)
    bare = internvl_tiles_for_image(2160, 3840, max_tiles=6, use_thumbnail=False)
    assert tiles == bare + 1
    assert bare > 1


def test_internvl_respects_max_tiles():
    for max_tiles in (1, 2, 4, 6, 8, 12):
        tiles = internvl_tiles_for_image(2160, 3840, max_tiles, use_thumbnail=False)
        assert tiles <= max_tiles


def test_internvl_wide_image_prefers_wide_grid():
    # A 4:1 panorama should tile horizontally, not squarely.
    tiles = internvl_tiles_for_image(1000, 4000, max_tiles=4, use_thumbnail=False)
    assert tiles == 4


def test_internvl_tokens_are_tiles_times_256():
    tiles = internvl_tiles_for_image(2160, 3840, 6)
    assert internvl_visual_tokens(2160, 3840, 6) == tiles * 256


def test_internvl_multi_image_scales_with_count():
    one = internvl_total_visual_tokens([(2160, 3840)], 6)
    three = internvl_total_visual_tokens([(2160, 3840)] * 3, 6)
    assert three == 3 * one


def test_internvl_rejects_bad_max_tiles():
    with pytest.raises(ValueError):
        internvl_tiles_for_image(1000, 1000, max_tiles=0)


# --- dispatch ---------------------------------------------------------------
def test_estimate_dispatches_per_adapter():
    sizes = [(2160, 3840)]
    qwen = estimate_visual_tokens("qwen25vl", sizes, {"max_visual_tokens_per_image": 1280})
    internvl = estimate_visual_tokens("internvl3_hf", sizes, {"max_tiles": 6})
    assert qwen == qwen_visual_tokens(2160, 3840, 1280)
    assert internvl == internvl_visual_tokens(2160, 3840, 6)


def test_estimate_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="no token estimator"):
        estimate_visual_tokens("llava", [(100, 100)], {})


def test_budget_at_rung_sets_the_right_knob():
    qwen = budget_at_rung("qwen25vl", {"min_pixels": 3136}, 2560)
    assert qwen == {"min_pixels": 3136, "max_visual_tokens_per_image": 2560}

    internvl = budget_at_rung("internvl3_hf", {"use_thumbnail": True}, 8)
    assert internvl == {"use_thumbnail": True, "max_tiles": 8}


def test_budget_at_rung_does_not_mutate_input():
    original = {"max_visual_tokens_per_image": 1280}
    budget_at_rung("qwen25vl", original, 4096)
    assert original == {"max_visual_tokens_per_image": 1280}


def test_ladder_selection_matches_adapter_family():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    assert ladder_for("qwen25vl", cfg.sweep) == cfg.sweep.qwen_visual_token_ladder
    assert ladder_for("internvl3_hf", cfg.sweep) == cfg.sweep.internvl_tile_ladder
    assert ladder_for("internvl3_native", cfg.sweep) == cfg.sweep.internvl_tile_ladder


def test_every_configured_adapter_has_a_ladder_and_estimator():
    # A new model added to config must not reach the sweep without both, or the
    # sweep would fail only after the weights had already been downloaded.
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    for model in cfg.models.values():
        assert ladder_for(model.adapter, cfg.sweep)
        assert estimate_visual_tokens(model.adapter, [(2160, 3840)], model.budget) > 0


def test_rungs_can_collapse_for_a_given_aspect_ratio():
    # Real behaviour worth pinning: InternVL's tiling is quantized by aspect
    # ratio, so for a 16:9 image max_tiles 12 and 8 both settle on the same grid.
    # This is why the sweep dedupes at run time instead of trusting the ladder.
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    ladder = cfg.sweep.internvl_tile_ladder
    estimates = [internvl_visual_tokens(2160, 3840, rung) for rung in ladder]
    assert len(set(estimates)) < len(estimates)


def test_distinct_rungs_drops_duplicates_keeping_the_highest():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    model = cfg.model("internvl3_8b_hf")
    ladder = cfg.sweep.internvl_tile_ladder
    kept, skipped = distinct_rungs(model.adapter, model.budget, [(2160, 3840)], ladder)

    estimates = [e for _, e in kept]
    assert len(set(estimates)) == len(estimates)
    # Highest rung of each duplicate group survives -- same computation, so the
    # larger number is the honest boundary to report. Asserted against the
    # configured ladder rather than a literal, so tuning the ladder in config
    # does not break the test that guards the dedup behaviour.
    assert kept[0][0] == max(ladder)
    assert {r for r, _, _ in skipped} == set(ladder) - {r for r, _ in kept}
    for rung, estimate, kept_rung in skipped:
        assert kept_rung > rung
        assert dict(kept)[kept_rung] == estimate


def test_distinct_rungs_stays_descending():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    for model in cfg.models.values():
        kept, _ = distinct_rungs(
            model.adapter,
            model.budget,
            [(2160, 3840)],
            ladder_for(model.adapter, cfg.sweep),
        )
        rungs = [r for r, _ in kept]
        estimates = [e for _, e in kept]
        assert rungs == sorted(rungs, reverse=True)
        assert estimates == sorted(estimates, reverse=True)


def test_distinct_rungs_keeps_everything_when_all_differ():
    # A ladder entirely below the image's native token count (~10,549 for 4K):
    # every rung binds, so nothing should be dropped.
    ladder = [8192, 6144, 4096, 2560, 1280]
    kept, skipped = distinct_rungs("qwen25vl", {"min_pixels": 3136}, [(2160, 3840)], ladder)
    assert [r for r, _ in kept] == ladder
    assert skipped == []


def test_distinct_rungs_drops_rungs_above_native_resolution():
    # Rungs above native cannot bind -- the image has no more pixels to give.
    # With a tall ladder on a 4K image this is the common case, and measuring
    # those rungs would burn GPU cells re-testing full resolution.
    ladder = [40960, 20480, 10240, 4096]
    kept, skipped = distinct_rungs("qwen25vl", {"min_pixels": 3136}, [(2160, 3840)], ladder)
    assert 20480 in {r for r, _, _ in skipped}
    assert [r for r, _ in kept][0] == 40960     # highest of the collapsed group survives
    assert len({e for _, e in kept}) == len(kept)


def test_every_rung_estimate_is_nonzero():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    for model in cfg.models.values():
        kept, _ = distinct_rungs(
            model.adapter,
            model.budget,
            [(2160, 3840)],
            ladder_for(model.adapter, cfg.sweep),
        )
        assert kept, f"{model.name}: no usable rungs"
        assert all(e > 0 for _, e in kept)
