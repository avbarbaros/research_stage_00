"""InternVL3-8B adapter -- original OpenGVLab repo (fallback path).

Enabled by flipping ``models.internvl3_8b_native.enabled`` in config.yaml. Two
reasons it exists:

* It matches InternVL3's published benchmark numbers exactly, which matters once
  stage 0.2 starts comparing our scores against leaderboards.
* ``-hf`` conversions have historically been the weaker path for *multi-image*
  specifically -- which is precisely what stage 0 is testing. If the primary
  adapter produces suspicious multi-image behaviour, this is the control.

The cost is bespoke code: ``trust_remote_code=True``, the model's own ``.chat()``
API, and the dynamic tiling preprocessing reimplemented here because the repo
keeps it in an example script rather than the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..budget import INTERNVL_TILE_PX, INTERNVL_TOKENS_PER_TILE
from ..config import DecodeConfig
from .base import GenOutput, VLMAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL3NativeAdapter(VLMAdapter):
    """InternVL3 via the original repo's ``model.chat()`` interface."""

    def _load_impl(self, model_path: Path, budget: dict[str, Any]) -> None:
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        # This repo requires remote code; say so plainly rather than failing
        # deep inside from_pretrained with a confusing message.
        if not self.model_cfg.trust_remote_code:
            raise RuntimeError(
                f"models.{self.name}: the native InternVL3 repo requires "
                f"trust_remote_code: true in config.yaml"
            )

        self.processor = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=False
        )
        kwargs = self._load_kwargs()
        # The custom modeling code selects its own attention implementation and
        # rejects the argument transformers-native models accept.
        kwargs.pop("attn_implementation", None)
        self.model = AutoModel.from_pretrained(str(model_path), **kwargs)
        self.model.eval()

    # -- dynamic tiling -----------------------------------------------------
    def _transform(self, tile_px: int):
        import torchvision.transforms as T  # noqa: PLC0415
        from torchvision.transforms.functional import InterpolationMode  # noqa: PLC0415

        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((tile_px, tile_px), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _dynamic_preprocess(
        self, image, max_tiles: int, use_thumbnail: bool, tile_px: int = INTERNVL_TILE_PX
    ):
        """InternVL3's aspect-ratio tiling.

        Mirrors :func:`mmro.budget.internvl_tiles_for_image`; the sweep compares
        the tile count produced here against that estimate, so the two staying
        in step is itself a checked invariant.
        """
        width, height = image.size
        target = width / height

        ratios = sorted(
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
        for cols, rows in ratios:
            diff = abs(cols / rows - target)
            if diff < best_diff:
                best_diff, best = diff, (cols, rows)
            elif diff == best_diff and width * height > 0.5 * tile_px * tile_px * cols * rows:
                best = (cols, rows)

        cols, rows = best
        resized = image.resize((tile_px * cols, tile_px * rows))
        tiles = []
        for i in range(cols * rows):
            box = (
                (i % cols) * tile_px,
                (i // cols) * tile_px,
                ((i % cols) + 1) * tile_px,
                ((i // cols) + 1) * tile_px,
            )
            tiles.append(resized.crop(box))

        if use_thumbnail and len(tiles) > 1:
            tiles.append(image.resize((tile_px, tile_px)))
        return tiles

    def _pixel_values(self, images: list[Path], budget: dict[str, Any]):
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        max_tiles = int(budget.get("max_tiles", 6))
        use_thumbnail = bool(budget.get("use_thumbnail", True))
        transform = self._transform(INTERNVL_TILE_PX)

        per_image, counts = [], []
        for path in images:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
            tiles = self._dynamic_preprocess(image, max_tiles, use_thumbnail)
            per_image.append(torch.stack([transform(t) for t in tiles]))
            counts.append(len(tiles))

        pixel_values = torch.cat(per_image, dim=0).to(self._torch_dtype())
        return pixel_values.to(self.model.device), counts

    def _generate_impl(
        self, images: list[Path], text: str, decode: DecodeConfig, budget: dict[str, Any]
    ) -> GenOutput:
        import torch  # noqa: PLC0415

        pixel_values, counts = self._pixel_values(images, budget)

        # The native API needs an explicit Image-N placeholder per image; without
        # them a multi-image prompt silently attends to only the first.
        if len(images) > 1:
            prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(images)))
            question = prefix + text
        else:
            question = f"<image>\n{text}"

        generation_config = dict(decode.to_generate_kwargs())

        with torch.inference_mode():
            response = self.model.chat(
                self.processor,
                pixel_values,
                question,
                generation_config,
                num_patches_list=counts if len(images) > 1 else None,
            )

        prompt_tokens = int(len(self.processor(question)["input_ids"]))
        return GenOutput(
            text=str(response).strip(),
            visual_tokens=sum(counts) * INTERNVL_TOKENS_PER_TILE,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(len(self.processor(str(response))["input_ids"])),
            extra={"tiles_per_image": counts},
        )

    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
