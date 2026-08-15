"""Held-out image set tests.

Images are supplied manually, so the failure modes are all about a missing or
insufficient set. Every one of them must stop the run with an actionable
message rather than proceeding on whatever happens to be there -- a stage-0
result measured on the wrong images is worse than no result.
"""

from __future__ import annotations

import pytest
from PIL import Image

from mmro.images import (
    ADVISORY_MIN_LONG_SIDE,
    ImageSetError,
    advisories,
    discover,
    paths_of,
    provenance,
    require,
    sha256_file,
    sizes_of,
)


def make_image(path, width=3840, height=2160, colour=(120, 90, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), colour).save(path)
    return path


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "heldout"
    make_image(d / "02_second.jpg", colour=(10, 20, 30))
    make_image(d / "01_first.jpg", colour=(200, 30, 40))
    make_image(d / "03_third.png", width=1920, height=1080)
    return d


# --- discovery --------------------------------------------------------------
def test_discovers_images_in_sorted_order(image_dir):
    images = discover(image_dir)
    assert [i.name for i in images] == ["01_first.jpg", "02_second.jpg", "03_third.png"]


def test_records_dimensions_and_hash(image_dir):
    first = discover(image_dir)[0]
    assert (first.height, first.width) == (2160, 3840)
    assert first.sha256 == sha256_file(first.path)
    assert first.long_side == 3840
    assert first.megapixels == pytest.approx(8.29, abs=0.01)


def test_selection_is_deterministic_and_ordered(image_dir):
    # Filename order is the control surface documented in the README, so a
    # 1-image test must reliably get the image the researcher put first.
    assert [i.name for i in require(image_dir, 1)] == ["01_first.jpg"]
    assert [i.name for i in require(image_dir, 2)] == ["01_first.jpg", "02_second.jpg"]
    assert require(image_dir, 3) == discover(image_dir)


def test_ignores_non_image_files(image_dir):
    (image_dir / "notes.txt").write_text("not an image")
    (image_dir / "manifest.json").write_text("{}")
    assert len(discover(image_dir)) == 3


def test_skips_unreadable_files_but_keeps_going(image_dir):
    (image_dir / "04_corrupt.jpg").write_bytes(b"this is not a jpeg")
    images = discover(image_dir)
    assert [i.name for i in images] == ["01_first.jpg", "02_second.jpg", "03_third.png"]


# --- failure modes ----------------------------------------------------------
def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(ImageSetError, match="does not exist"):
        discover(tmp_path / "absent")


def test_empty_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ImageSetError, match="no usable images"):
        discover(tmp_path / "empty")


def test_directory_of_only_unreadable_files_is_an_error(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"nope")
    with pytest.raises(ImageSetError, match="unreadable"):
        discover(d)


def test_requiring_more_images_than_present_is_an_error(image_dir):
    with pytest.raises(ImageSetError, match="need 5 images"):
        require(image_dir, 5)


def test_insufficient_error_names_the_files_present(image_dir):
    # The message has to be actionable: which files are there, and how many more
    # are needed.
    with pytest.raises(ImageSetError) as exc:
        require(image_dir, 4)
    message = str(exc.value)
    assert "01_first.jpg" in message
    assert "1 more" in message


def test_file_instead_of_directory_is_an_error(tmp_path):
    path = tmp_path / "afile"
    path.write_text("x")
    with pytest.raises(ImageSetError, match="not a directory"):
        discover(path)


# --- advisories -------------------------------------------------------------
def test_low_resolution_images_are_flagged_not_rejected(tmp_path):
    d = tmp_path / "small"
    make_image(d / "01_small.jpg", width=640, height=480)
    images = discover(d)          # still usable
    notes = advisories(images)
    assert len(notes) == 1
    assert "high-resolution" in notes[0]


def test_high_resolution_images_produce_no_advisory(image_dir):
    big = [i for i in discover(image_dir) if i.long_side >= ADVISORY_MIN_LONG_SIDE]
    assert advisories(big) == []


# --- helpers feeding budget.py and the records -----------------------------
def test_sizes_are_height_width_for_the_estimators(image_dir):
    # budget.py takes (height, width); swapping them would silently mis-budget
    # every non-square image.
    assert sizes_of(discover(image_dir))[0] == (2160, 3840)


def test_paths_round_trip(image_dir):
    assert [p.name for p in paths_of(discover(image_dir))][0] == "01_first.jpg"


def test_provenance_carries_name_size_and_hash(image_dir):
    entry = provenance(discover(image_dir))[0]
    name, size, digest = entry.split(":")
    assert name == "01_first.jpg"
    assert size == "3840x2160"
    assert len(digest) == 12


def test_provenance_distinguishes_different_images(image_dir):
    entries = provenance(discover(image_dir))
    assert len({e.split(":")[2] for e in entries}) == 3
