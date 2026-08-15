"""InternVL3-8B adapter -- transformers-native ``-hf`` variant (primary path).

``OpenGVLab/InternVL3-8B-hf`` exposes a standard ``AutoProcessor`` and chat
template, needs no ``trust_remote_code``, and is the variant vLLM (stage 0.1)
and both harnesses (stage 0.2) handle most cleanly.

Tiling is the budget knob here: 448px tiles at 256 tokens each after pixel
shuffle, plus a thumbnail tile when the grid has more than one. ``max_tiles``
therefore multiplies across a multi-image prompt, which is what the sweep
measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..budget import INTERNVL_TOKENS_PER_TILE
from ..config import DecodeConfig
from .base import GenOutput, VLMAdapter


class InternVL3HFAdapter(VLMAdapter):
    """InternVL3 via ``AutoModelForImageTextToText``."""

    def _load_impl(self, model_path: Path, budget: dict[str, Any]) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        extra = {"trust_remote_code": True} if self.model_cfg.trust_remote_code else {}
        self.processor = AutoProcessor.from_pretrained(str(model_path), **extra)

        # The tile ceiling lives on the image processor. Set it at load time so
        # every call inherits the configured budget rather than the default 12.
        self._apply_tile_budget(budget)

        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), **self._load_kwargs()
        )
        self.model.eval()

    def _apply_tile_budget(self, budget: dict[str, Any]) -> None:
        """Push ``max_tiles``/``use_thumbnail`` onto the image processor.

        Attribute names have moved between transformers releases, so we set
        whichever the installed version exposes and fail loudly if none are
        present -- silently running at the default tile count would produce a
        sweep whose rungs all mean the same thing.
        """
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("InternVL processor exposes no image_processor")

        max_tiles = int(budget.get("max_tiles", 6))
        use_thumbnail = bool(budget.get("use_thumbnail", True))

        applied = []
        for attr in ("max_num_tiles", "max_patches", "max_num"):
            if hasattr(image_processor, attr):
                setattr(image_processor, attr, max_tiles)
                applied.append(attr)
        for attr in ("use_thumbnail", "crop_to_patches"):
            if hasattr(image_processor, attr):
                setattr(image_processor, attr, use_thumbnail if attr == "use_thumbnail" else True)
                applied.append(attr)

        if not applied:
            raise RuntimeError(
                "could not set a tile budget on this InternVL image processor "
                f"(tried max_num_tiles/max_patches/max_num on "
                f"{type(image_processor).__name__}). The sweep would otherwise run "
                f"every rung at the default tile count. Check the installed "
                f"transformers version, or switch to the internvl3_native adapter."
            )
        self._tile_attrs = applied

    def _build_messages(self, images: list[Path], text: str) -> list[dict]:
        content: list[dict[str, Any]] = [
            {"type": "image", "url": str(path.resolve())} for path in images
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _generate_impl(
        self, images: list[Path], text: str, decode: DecodeConfig, budget: dict[str, Any]
    ) -> GenOutput:
        import torch  # noqa: PLC0415

        self._apply_tile_budget(budget)
        messages = self._build_messages(images, text)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Ground truth: pixel_values arrives as (total_tiles, C, H, W) across all
        # images, so its leading dimension is the true tile count.
        visual_tokens = 0
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            visual_tokens = int(pixel_values.shape[0]) * INTERNVL_TOKENS_PER_TILE

        prompt_tokens = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **decode.to_generate_kwargs())

        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        return GenOutput(
            text=output.strip(),
            visual_tokens=visual_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(trimmed.shape[-1]),
        )

    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
