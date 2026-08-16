"""Environment bootstrap: HF cache location, dtype checks, version capture.

Every path comes from ``config.yaml``. This module mounts nothing and guesses
nothing -- if you are caching on Drive, mount it yourself and point
``paths.hf_home`` at the result. It will tell you when a configured directory is
missing or short on space, but it will not silently relocate 33GB of weights.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError


@dataclass
class EnvReport:
    hf_home: str
    cache_mode: str
    hf_home_exists: bool
    hf_home_free_gb: float
    python: str
    torch_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_gb: float
    compute_capability: str | None
    bf16_supported: bool
    dtype: str
    attn_implementation: str
    packages: dict[str, str | None]
    warnings: list[str]


def free_gb(path: Path) -> float:
    """Free space at the nearest existing ancestor of ``path``."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / (1024**3)
    except OSError:
        return 0.0


def apply_env(cfg: Config) -> Path:
    """Set the HF cache environment variables. Must run before transformers imports."""
    hf_home = cfg.paths.hf_home_resolved
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    # huggingface_hub replaced hf_transfer with Xet; setting the old variable now
    # only produces a deprecation warning on every import.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # Fragmentation is the practical VRAM limit during a sweep on a 24GB card;
    # expandable segments meaningfully reduce it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return hf_home


def check_cache_space(cfg: Config) -> list[str]:
    """Warn when the configured cache cannot hold the enabled models.

    Both models together are ~33GB. Drive's free tier is 15GB, which is the
    single most likely way for a fetch to die two thirds of the way through.
    """
    warnings: list[str] = []
    hf_home = cfg.paths.hf_home_resolved
    available = free_gb(hf_home)
    if available < cfg.runtime.min_free_drive_gb:
        warnings.append(
            f"{hf_home} has {available:.1f}GB free but runtime.min_free_drive_gb "
            f"is {cfg.runtime.min_free_drive_gb}GB (Qwen2.5-VL-7B and InternVL3-8B "
            f"are ~33GB together). Free up space, point paths.hf_home somewhere "
            f"larger, or lower min_free_drive_gb if you know better."
        )
    if cfg.paths.caching_on_drive and not hf_home.exists():
        warnings.append(
            f"paths.hf_home does not exist: {hf_home}. If this is on Google "
            f"Drive, mount it first -- nothing here mounts it for you."
        )
    return warnings


def seed_everything(seed: int) -> None:
    import random  # noqa: PLC0415

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def torch_dtype(cfg: Config):
    """``runtime.dtype`` as a torch dtype, validated against the GPU."""
    import torch  # noqa: PLC0415

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = mapping[cfg.runtime.dtype]
    if dtype is torch.bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            raise ConfigError(
                "runtime.dtype: bfloat16 is not supported by this GPU. "
                "Set runtime.dtype: float16 in config.yaml."
            )
    return dtype


def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    names = [
        "torch", "torchvision", "transformers", "accelerate", "peft",
        "qwen-vl-utils", "huggingface-hub", "safetensors", "pillow",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def probe(cfg: Config) -> EnvReport:
    """Inspect the environment and collect everything worth recording."""
    hf_home = apply_env(cfg)
    warnings = check_cache_space(cfg)

    torch_version = None
    cuda_available = False
    gpu_name = None
    gpu_total = 0.0
    capability = None
    bf16 = False
    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_total = props.total_memory / (1024**3)
            capability = f"{props.major}.{props.minor}"
            bf16 = torch.cuda.is_bf16_supported()
        else:
            warnings.append("no CUDA device visible; GPU steps will not run")
    except ImportError:
        warnings.append("torch is not installed")

    if cuda_available and cfg.runtime.dtype == "bfloat16" and not bf16:
        warnings.append(
            "runtime.dtype is bfloat16 but this GPU does not support it; "
            "set runtime.dtype: float16"
        )
    if cfg.runtime.attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401,PLC0415
        except ImportError:
            warnings.append(
                "runtime.attn_implementation is flash_attention_2 but flash-attn is "
                "not installed; install a PREBUILT wheel or set it back to sdpa "
                "(building from source on Colab costs ~30 minutes)"
            )

    return EnvReport(
        hf_home=str(hf_home),
        cache_mode=cfg.paths.cache_mode,
        hf_home_exists=hf_home.exists(),
        hf_home_free_gb=round(free_gb(hf_home), 1),
        python=sys.version.split()[0],
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_gb=round(gpu_total, 2),
        compute_capability=capability,
        bf16_supported=bf16,
        dtype=cfg.runtime.dtype,
        attn_implementation=cfg.runtime.attn_implementation,
        packages=_package_versions(),
        warnings=warnings,
    )


def write_env_lock(cfg: Config, report: EnvReport) -> Path:
    """Record resolved versions so a working combination is reproducible."""
    import dataclasses  # noqa: PLC0415

    path = cfg.paths.results_path / "env_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    from .config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Probe the environment and write env_lock.json.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="exit non-zero if no CUDA device is available",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    report = probe(cfg)
    path = write_env_lock(cfg, report)

    print(f"python           : {report.python}")
    print(f"torch            : {report.torch_version}")
    print(f"gpu              : {report.gpu_name or 'none'} ({report.gpu_total_gb} GB)")
    print(f"compute cap      : {report.compute_capability}")
    print(f"bf16 supported   : {report.bf16_supported}")
    print(f"cache mode       : {report.cache_mode}")
    print(f"HF_HOME          : {report.hf_home} ({report.hf_home_free_gb} GB free)")
    print(f"env lock         : {path}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")

    if args.require_gpu and not report.cuda_available:
        print("error: --require-gpu was set but no CUDA device is available")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
