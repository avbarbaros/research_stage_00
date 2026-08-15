"""Sweep orchestration tests (no GPU).

The sweep's job is to turn OOMs into data. These tests cover the classification
and boundary logic, which is where a wrong answer would silently mislocate the
boundary that stage 0.2 depends on.
"""

from __future__ import annotations

from types import SimpleNamespace

from mmro.records import GenerationResult
from mmro.sweep import _parse_cell_output, boundaries


def result(model="m", n=1, rung=1280, ok=True, oom=False, tokens=100):
    return GenerationResult(
        model=model,
        request_id=f"{model}-{n}-{rung}",
        text="a caption" if ok else "",
        n_images=n,
        rung=rung,
        visual_tokens=tokens,
        oom=oom,
        peak_vram_gb=20.0,
    )


# --- boundary selection -----------------------------------------------------
def test_boundary_is_the_highest_surviving_rung():
    results = [
        result(rung=4096, ok=False, oom=True),
        result(rung=2560, ok=False, oom=True),
        result(rung=1280, ok=True, tokens=1200),
    ]
    found = boundaries(results)
    assert found[("m", 1)].rung == 1280


def test_boundaries_are_tracked_per_image_count():
    # The whole reason the sweep exists: the boundary moves with image count.
    results = [
        result(n=1, rung=2560, ok=True),
        result(n=3, rung=2560, ok=False, oom=True),
        result(n=3, rung=640, ok=True),
    ]
    found = boundaries(results)
    assert found[("m", 1)].rung == 2560
    assert found[("m", 3)].rung == 640


def test_boundaries_are_tracked_per_model():
    found = boundaries([
        result(model="qwen", rung=2560, ok=True),
        result(model="internvl", rung=6, ok=True),
    ])
    assert {k[0] for k in found} == {"qwen", "internvl"}


def test_failed_and_oom_cells_are_never_boundaries():
    assert boundaries([result(rung=4096, ok=False, oom=True)]) == {}
    assert boundaries([
        GenerationResult(model="m", request_id="x", n_images=1, rung=99, error="broken")
    ]) == {}


def test_cells_without_a_rung_are_ignored():
    # Smoke results share the schema but carry no rung; they must not be read as
    # sweep boundaries.
    assert boundaries([result(rung=None)]) == {}


def test_higher_rung_wins_when_several_survive():
    found = boundaries([result(rung=640, ok=True), result(rung=2560, ok=True)])
    assert found[("m", 1)].rung == 2560


# --- subprocess result recovery --------------------------------------------
def make_proc(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_parses_the_cell_result_line():
    payload = (
        'CELL_RESULT {"model": "m", "request_id": "r", "text": "a caption", '
        '"n_images": 2, "rung": 1280, "visual_tokens": 2400}'
    )
    parsed = _parse_cell_output(make_proc(stdout=f"loading...\n{payload}\n"), "m", 2, 1280)
    assert parsed.text == "a caption"
    assert parsed.visual_tokens == 2400
    assert parsed.ok


def test_parses_the_last_cell_result_when_output_is_noisy():
    stdout = (
        'CELL_RESULT {"model": "m", "request_id": "old", "text": "stale"}\n'
        'warning: something\n'
        'CELL_RESULT {"model": "m", "request_id": "new", "text": "fresh"}\n'
    )
    assert _parse_cell_output(make_proc(stdout=stdout), "m", 1, 1280).text == "fresh"


def test_unknown_fields_from_a_newer_cell_are_dropped():
    payload = 'CELL_RESULT {"model": "m", "request_id": "r", "text": "x", "future": 1}'
    assert _parse_cell_output(make_proc(stdout=payload), "m", 1, 1280).text == "x"


def test_a_crashed_cell_still_yields_a_record():
    # A CUDA abort prints nothing parseable; losing the cell entirely would
    # leave a hole in the sweep.
    parsed = _parse_cell_output(
        make_proc(stderr="CUDA error: device-side assert triggered", returncode=1), "m", 2, 640
    )
    assert parsed.model == "m"
    assert parsed.n_images == 2
    assert parsed.rung == 640
    assert not parsed.ok
    assert "device-side assert" in parsed.error


def test_sigkill_is_not_recorded_as_a_cuda_oom():
    # A killed process means host RAM exhaustion during load, not VRAM. Calling
    # it a CUDA OOM would put the boundary in the wrong place and understate
    # what the GPU can do.
    parsed = _parse_cell_output(make_proc(returncode=-9), "m", 4, 4096)
    assert parsed.oom is True
    assert "host RAM" in parsed.error


def test_empty_output_still_produces_a_record():
    parsed = _parse_cell_output(make_proc(returncode=1), "m", 1, 320)
    assert parsed.model == "m" and not parsed.ok
