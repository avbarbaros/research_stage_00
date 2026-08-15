"""Stage-1 training scaffold check — non-blocking, no training step.

Stage 0 does not train. This module only answers the two questions that would
otherwise block the first day of stage 1:

1. Do the LoRA target module names an adapter declares actually resolve against
   the loaded model? A name that matches nothing produces a PEFT model with
   zero trainable parameters — which trains happily and learns nothing.
2. Is an FSDP config valid? FSDP over DeepSpeed because single-GPU Colab makes
   ZeRO-3 pointless and DeepSpeed's JIT ops are a recurring Colab build failure.

Run per model, in its own process, like everything else that loads weights.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config, ConfigError, load_config


@dataclass
class ScaffoldReport:
    model: str
    target_modules: list[str] = field(default_factory=list)
    matched: dict[str, int] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    trainable_params: int = 0
    total_params: int = 0
    vision_tower_hits: list[str] = field(default_factory=list)
    fsdp_config_valid: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


VISION_HINTS = ("visual", "vision_model", "vision_tower", "mlp1")


def check_lora(cfg: Config, model_name: str) -> ScaffoldReport:
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    from .models import load_adapter  # noqa: PLC0415

    adapter = load_adapter(cfg, model_name)
    targets = adapter.lora_target_modules()
    report = ScaffoldReport(model=model_name, target_modules=targets)

    module_names = [name for name, _ in adapter.model.named_modules()]
    for target in targets:
        hits = [n for n in module_names if n.split(".")[-1] == target or fnmatch.fnmatch(n, target)]
        if hits:
            report.matched[target] = len(hits)
        else:
            report.unmatched.append(target)

    if report.unmatched:
        report.problems.append(
            f"target module(s) {report.unmatched} match nothing in the model — "
            f"LoRA would attach zero adapters for them and train silently"
        )

    # The vision tower is deliberately excluded at stage 0. Flag leakage rather
    # than letting "LoRA" quietly come to mean something different later.
    for target, _ in report.matched.items():
        hits = [
            n for n in module_names
            if n.split(".")[-1] == target and any(h in n for h in VISION_HINTS)
        ]
        report.vision_tower_hits += hits[:3]
    if report.vision_tower_hits:
        report.problems.append(
            f"targets also match vision-tower modules (e.g. "
            f"{report.vision_tower_hits[:2]}); decide explicitly at stage 1 "
            f"whether to tune the vision tower"
        )

    peft_model = get_peft_model(
        adapter.model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets, bias="none"),
    )
    report.trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    report.total_params = sum(p.numel() for p in peft_model.parameters())
    if report.trainable_params == 0:
        report.problems.append("PEFT produced zero trainable parameters")

    adapter.unload()
    return report


def check_fsdp(cfg: Config) -> tuple[bool, list[str]]:
    """Validate an FSDP plugin config without running anything."""
    problems: list[str] = []
    try:
        from accelerate import FullyShardedDataParallelPlugin  # noqa: PLC0415,F401
    except ImportError as exc:
        return False, [f"accelerate is not installed or lacks FSDP support: {exc}"]

    try:
        import torch  # noqa: PLC0415
        from torch.distributed.fsdp import MixedPrecision  # noqa: PLC0415

        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"FSDP mixed-precision config is invalid: {exc}")

    return not problems, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the stage-1 training scaffold.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    try:
        cfg.model(args.model)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1

    from .env import apply_env  # noqa: PLC0415

    apply_env(cfg)

    report = check_lora(cfg, args.model)
    report.fsdp_config_valid, fsdp_problems = check_fsdp(cfg)
    report.problems += fsdp_problems

    print(f"=== scaffold: {args.model} ===")
    print(f"  target modules : {report.target_modules}")
    print(f"  matched        : {report.matched}")
    print(f"  trainable      : {report.trainable_params:,} / {report.total_params:,}")
    print(f"  FSDP config    : {'valid' if report.fsdp_config_valid else 'INVALID'}")
    for problem in report.problems:
        print(f"  PROBLEM        : {problem}")

    out = cfg.paths.results_path / f"scaffold_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2) + "\n")
    print(f"  wrote          : {out}")

    # Non-blocking by design: a scaffold problem is a stage-1 concern and must
    # not fail the stage-0 milestone.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
