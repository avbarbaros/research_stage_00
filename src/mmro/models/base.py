"""The adapter seam.

``generate(images, text, decode) -> str`` is the contract that matters. It is
what VLMEvalKit's ``BaseModel.generate_inner`` and lmms-eval's
``lmms.generate_until`` both collapse to, so writing stage 0's captioning
against it means stage 0.2 adds thin wrappers instead of reworking the core.

``generate_measured`` is the same code path with telemetry attached -- peak VRAM,
timings, and the token count the model *actually* used. Stage 0 calls the
measured form; the harnesses will call the bare one.
"""

from __future__ import annotations

import importlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, DecodeConfig, ModelConfig
from ..records import GenerationResult

# adapter name -> (module, class). Lazy so importing this module does not drag
# in transformers, and so a broken optional adapter cannot break the others.
ADAPTERS: dict[str, tuple[str, str]] = {
    "qwen25vl": ("mmro.models.qwen25vl", "Qwen25VLAdapter"),
    "internvl3_hf": ("mmro.models.internvl3_hf", "InternVL3HFAdapter"),
    "internvl3_native": ("mmro.models.internvl3_native", "InternVL3NativeAdapter"),
}


@dataclass
class GenOutput:
    """What an adapter's generation step reports back to the base class."""

    text: str
    visual_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class VLMAdapter(ABC):
    """Base class for a vision-language model under test."""

    name: str = ""

    def __init__(self, cfg: Config, model_cfg: ModelConfig):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.name = model_cfg.name
        self.model = None
        self.processor = None
        self.load_seconds = 0.0
        self._loaded = False

    # -- subclass hooks -----------------------------------------------------
    @abstractmethod
    def _load_impl(self, model_path: Path, budget: dict[str, Any]) -> None:
        """Load weights and processor from ``model_path``."""

    @abstractmethod
    def _generate_impl(
        self, images: list[Path], text: str, decode: DecodeConfig, budget: dict[str, Any]
    ) -> GenOutput:
        """Run one generation and report the tokens actually used."""

    @abstractmethod
    def lora_target_modules(self) -> list[str]:
        """LoRA target module names for stage-1 training (scaffold only)."""

    # -- shared machinery ---------------------------------------------------
    @property
    def budget(self) -> dict[str, Any]:
        return dict(self.model_cfg.budget)

    def load(self, budget: dict[str, Any] | None = None) -> None:
        """Load the model, timing it.

        ``budget`` may override the configured one: the sweep reloads at each
        rung because preprocessing limits are baked into the processor at
        construction time for both model families.
        """
        from .. import cache  # noqa: PLC0415 - avoids a cycle at import time

        model_path = cache.local_path(self.cfg, self.model_cfg)
        start = time.perf_counter()
        self._load_impl(model_path, budget if budget is not None else self.budget)
        self.load_seconds = time.perf_counter() - start
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def generate(
        self,
        images: list[Path],
        text: str,
        decode: DecodeConfig | None = None,
        budget: dict[str, Any] | None = None,
    ) -> str:
        """THE harness contract. Returns the generated text and nothing else."""
        self.ensure_loaded()
        decode = decode or self.cfg.decode
        return self._generate_impl(
            list(images), text, decode, budget if budget is not None else self.budget
        ).text

    def generate_measured(
        self,
        images: list[Path],
        text: str,
        decode: DecodeConfig | None = None,
        budget: dict[str, Any] | None = None,
        estimated_visual_tokens: int = 0,
        rung: int | None = None,
        image_provenance: list[str] | None = None,
        request_id: str | None = None,
    ) -> GenerationResult:
        """Same path as :meth:`generate`, with telemetry.

        An OOM is recorded and returned rather than raised: the sweep pushes
        each model until it fails, so failure is the measurement, not a crash.
        """
        from .. import vram  # noqa: PLC0415

        decode = decode or self.cfg.decode
        budget = budget if budget is not None else self.budget
        images = list(images)

        result = GenerationResult(
            model=self.name,
            request_id=request_id or uuid.uuid4().hex[:12],
            n_images=len(images),
            estimated_visual_tokens=estimated_visual_tokens,
            rung=rung,
            budget=dict(budget),
            decode=decode.to_generate_kwargs(),
            images=image_provenance or [p.name for p in images],
            load_seconds=self.load_seconds,
        )

        trace = vram.MemoryTrace()
        try:
            with vram.track(trace):
                output = self._generate_impl(images, text, decode, budget)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            result.error = f"{type(exc).__name__}: {exc}"
            result.peak_vram_gb = round(trace.peak_gb, 3)
            result.generate_seconds = round(trace.seconds, 3)
            return result

        result.generate_seconds = round(trace.seconds, 3)
        result.peak_vram_gb = round(trace.peak_gb, 3)

        if trace.oom:
            result.oom = True
            result.error = trace.error
            return result

        result.text = output.text
        result.visual_tokens = output.visual_tokens
        result.prompt_tokens = output.prompt_tokens
        result.completion_tokens = output.completion_tokens
        return result

    def unload(self) -> None:
        """Best-effort release.

        Not sufficient to swap models inside one process on a 24GB L4 --
        fragmentation defeats it -- which is why smoke and sweep run one model
        per subprocess. Useful between rungs of the same model.
        """
        from .. import vram  # noqa: PLC0415

        self.model = None
        self.processor = None
        self._loaded = False
        vram.empty_cache()

    # -- shared load helpers ------------------------------------------------
    def _torch_dtype(self):
        from ..env import torch_dtype  # noqa: PLC0415

        return torch_dtype(self.cfg)

    def _load_kwargs(self) -> dict[str, Any]:
        """``from_pretrained`` kwargs shared by every adapter."""
        kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(),
            "device_map": self.cfg.runtime.device_map,
            "attn_implementation": self.cfg.runtime.attn_implementation,
        }
        if self.model_cfg.trust_remote_code:
            kwargs["trust_remote_code"] = True
        return kwargs


def get_adapter_class(adapter: str) -> type[VLMAdapter]:
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}; valid: {sorted(ADAPTERS)}")
    module_name, class_name = ADAPTERS[adapter]
    return getattr(importlib.import_module(module_name), class_name)


def load_adapter(cfg: Config, model_name: str, load: bool = True) -> VLMAdapter:
    """Build (and optionally load) the adapter for a configured model."""
    model_cfg = cfg.model(model_name)
    adapter = get_adapter_class(model_cfg.adapter)(cfg, model_cfg)
    if load:
        adapter.load()
    return adapter
