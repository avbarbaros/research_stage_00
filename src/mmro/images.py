"""Held-out image set: discovery, provenance, and measurement.

Images are supplied manually -- copy them into ``paths.images_dir`` yourself.
Nothing here downloads or generates anything. If the directory is missing or
does not hold enough images, the run stops with an error rather than quietly
proceeding on a smaller or substituted set, which would make the stage-0 result
meaningless.

Selection is deterministic: files are used in sorted filename order, so prefix
them (``01_``, ``02_``, ...) to control which image a 1-image test uses. Put a
dense-text or multi-object image first -- a model that has silently lost its
vision path still emits fluent, plausible captions for an open scene, and those
two kinds make the failure visible.

Each image's sha256 is recorded alongside results so a run can be tied to the
exact files it saw.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Fallback only; the real threshold is images.min_long_side in config.yaml.
DEFAULT_MIN_LONG_SIDE = 2048


class ImageSetError(Exception):
    """Raised when the held-out image set is missing or insufficient."""


@dataclass
class HeldOutImage:
    path: Path
    height: int
    width: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def long_side(self) -> int:
        return max(self.height, self.width)

    @property
    def megapixels(self) -> float:
        return self.height * self.width / 1e6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(images_dir: Path) -> list[HeldOutImage]:
    """Load every usable image in ``images_dir``, in sorted filename order."""
    if not images_dir.exists():
        raise ImageSetError(
            f"image directory does not exist: {images_dir}\n"
            f"Create it and copy your held-out images in, then re-run. "
            f"The directory is set by paths.images_dir in config.yaml."
        )
    if not images_dir.is_dir():
        raise ImageSetError(f"paths.images_dir is not a directory: {images_dir}")

    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    candidates = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    images: list[HeldOutImage] = []
    unreadable: list[str] = []
    for path in candidates:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                height, width = img.height, img.width
        except (UnidentifiedImageError, OSError) as exc:
            unreadable.append(f"{path.name}: {exc}")
            continue
        images.append(
            HeldOutImage(path=path, height=height, width=width, sha256=sha256_file(path))
        )

    if not images:
        detail = ""
        if unreadable:
            detail = "\nFiles present but unreadable:\n  " + "\n  ".join(unreadable)
        elif candidates:
            detail = f"\n{len(candidates)} candidate file(s) found but none opened as images."
        raise ImageSetError(
            f"no usable images in {images_dir}\n"
            f"Copy your held-out images there (accepted: "
            f"{', '.join(sorted(IMAGE_SUFFIXES))}) and re-run.{detail}"
        )

    return images


def require(images_dir: Path, count: int) -> list[HeldOutImage]:
    """Return exactly ``count`` images, or raise explaining what is missing."""
    images = discover(images_dir)
    if len(images) < count:
        raise ImageSetError(
            f"need {count} images but {images_dir} holds only {len(images)}: "
            f"{[i.name for i in images]}\n"
            f"Copy at least {count - len(images)} more in, or lower the image "
            f"counts in the smoke/sweep sections of config.yaml."
        )
    return images[:count]


def undersized(
    images: list[HeldOutImage], min_long_side: int = DEFAULT_MIN_LONG_SIDE
) -> list[HeldOutImage]:
    return [i for i in images if i.long_side < min_long_side]


def advisories(
    images: list[HeldOutImage], min_long_side: int = DEFAULT_MIN_LONG_SIDE
) -> list[str]:
    """Notes about an image set's suitability for a high-resolution test.

    Small images do not merely weaken the result, they invert it: once an image
    is smaller than a rung's budget, the rung stops binding and the sweep
    reports a boundary it never actually tested.
    """
    notes = []
    for img in undersized(images, min_long_side):
        notes.append(
            f"{img.name} is {img.width}x{img.height} ({img.megapixels:.2f} MP), "
            f"below the {min_long_side}px floor in images.min_long_side — "
            f"budget rungs above ~{img.megapixels * 1e6 / 784:.0f} visual tokens "
            f"will not bind for it"
        )
    return notes


def sizes_of(images: list[HeldOutImage]) -> list[tuple[int, int]]:
    """``[(height, width), ...]`` -- the form budget.py's estimators expect."""
    return [(i.height, i.width) for i in images]


def paths_of(images: list[HeldOutImage]) -> list[Path]:
    return [i.path for i in images]


def provenance(images: list[HeldOutImage]) -> list[str]:
    """Compact per-image provenance strings recorded with every result."""
    return [f"{i.name}:{i.width}x{i.height}:{i.sha256[:12]}" for i in images]


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    from .config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Inspect the held-out image set.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--require", type=int, default=None,
        help="exit non-zero unless at least this many images are present",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    images_dir = cfg.paths.images_path

    try:
        images = discover(images_dir)
    except ImageSetError as exc:
        print(f"error: {exc}")
        return 1

    print(f"image set: {images_dir}\n")
    for i, img in enumerate(images):
        print(
            f"  [{i}] {img.name:<32} {img.width:>5}x{img.height:<5} "
            f"{img.megapixels:>5.1f} MP  {img.sha256[:12]}"
        )

    for note in advisories(images, cfg.images.min_long_side):
        print(f"\nnote: {note}")

    needed = max([*cfg.smoke.image_counts, *cfg.sweep.image_counts])
    print(f"\n{len(images)} image(s) available; config needs up to {needed}")

    required = args.require if args.require is not None else needed
    if len(images) < required:
        print(f"error: need at least {required} images, found {len(images)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
