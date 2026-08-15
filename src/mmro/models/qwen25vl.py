"""Qwen2.5-VL-7B-Instruct adapter.

The one thing this file must get right is the pixel budget. Preprocessing runs
through ``qwen_vl_utils.process_vision_info``, whose ``MAX_PIXELS`` default is
16,384 visual tokens for a single image -- see :mod:`mmro.budget` for the
arithmetic and why that ceiling does not fit a 24GB L4. We therefore always pass
``min_pixels``/``max_pixels`` explicitly, derived from the configured budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..budget import qwen_max_pixels
from ..config import DecodeConfig
from .base import GenOutput, VLMAdapter


class Qwen25VLAdapter(VLMAdapter):
    """Qwen2.5-VL via ``Qwen2_5_VLForConditionalGeneration``."""

    def _load_impl(self, model_path: Path, budget: dict[str, Any]) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

        min_pixels = int(budget.get("min_pixels", 3136))
        max_pixels = qwen_max_pixels(int(budget.get("max_visual_tokens_per_image", 1280)))

        # Bound on the processor as well as at call time: the processor applies
        # these during smart_resize, and leaving them unset here would let a
        # caller that forgets to pass them inherit the 16,384-token ceiling.
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **({"trust_remote_code": True} if self.model_cfg.trust_remote_code else {}),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **self._load_kwargs()
        )
        self.model.eval()

    def _build_messages(self, images: list[Path], text: str, budget: dict[str, Any]) -> list[dict]:
        """One user turn holding every image, then the text.

        Qwen2.5-VL handles multi-image natively this way; the per-image pixel
        bounds ride along on each image entry so process_vision_info honours the
        budget rather than its own defaults.
        """
        min_pixels = int(budget.get("min_pixels", 3136))
        max_pixels = qwen_max_pixels(int(budget.get("max_visual_tokens_per_image", 1280)))

        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "image": f"file://{path.resolve()}",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
            for path in images
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _generate_impl(
        self, images: list[Path], text: str, decode: DecodeConfig, budget: dict[str, Any]
    ) -> GenOutput:
        import torch  # noqa: PLC0415
        from qwen_vl_utils import process_vision_info  # noqa: PLC0415

        messages = self._build_messages(images, text, budget)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Ground truth, straight off the tensors the model will consume.
        # image_grid_thw rows are (t, h, w) in *patch* units; the 2x2 merge means
        # visual tokens are that product divided by merge_size^2.
        visual_tokens = 0
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
            visual_tokens = int(grid.prod(dim=-1).sum().item()) // (merge * merge)

        prompt_tokens = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **decode.to_generate_kwargs())

        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        output = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return GenOutput(
            text=output.strip(),
            visual_tokens=visual_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(trimmed.shape[-1]),
        )

    def lora_target_modules(self) -> list[str]:
        # Attention and MLP projections on the language tower. The vision tower
        # is excluded at stage 0; whether to tune it is a stage-1 decision, and
        # including it here would silently change what "LoRA" means later.
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
