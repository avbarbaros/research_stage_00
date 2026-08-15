"""Weight fetching and local staging.

Two separate concerns:

1. **Fetch** — ``snapshot_download`` into the HF cache (Drive or local). This is
   resumable: an interrupted 16GB pull must continue, not restart.
2. **Stage** — copy the snapshot to fast local disk before loading. Drive is a
   FUSE mount; mmap-ing multi-shard safetensors across it is slow and prone to
   stalling. We always load from local disk, even when caching on Drive.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, ModelConfig

# Weight shards plus everything the processor/tokenizer needs. Excluding the
# duplicate .bin/.pth copies roughly halves the download for repos that ship
# both formats.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",
    "*.py",           # required by trust_remote_code repos (InternVL3 native)
    "*.jinja",        # chat templates
]


@dataclass
class FetchResult:
    model: str
    repo_id: str
    snapshot_dir: str
    staged_dir: str
    n_shards: int = 0
    total_gb: float = 0.0
    staged: bool = False
    warnings: list[str] = field(default_factory=list)


def _shards(path: Path) -> list[Path]:
    return sorted(path.glob("*.safetensors"))


def _dir_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def verify_snapshot(path: Path) -> tuple[bool, list[str]]:
    """Check a snapshot looks complete.

    Guards against the failure mode that actually bites: a partial download that
    looks fine until ``from_pretrained`` fails minutes later, after the weights
    have already been staged.
    """
    problems: list[str] = []
    if not path.exists():
        return False, [f"snapshot directory does not exist: {path}"]

    if not (path / "config.json").exists():
        problems.append("config.json is missing")

    shards = _shards(path)
    if not shards:
        problems.append("no .safetensors shards found")

    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except json.JSONDecodeError as exc:
            problems.append(f"unreadable safetensors index: {exc}")
        else:
            expected = set(weight_map.values())
            present = {s.name for s in shards}
            missing = sorted(expected - present)
            if missing:
                problems.append(
                    f"{len(missing)} shard(s) named in the index are missing: "
                    f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
                )

    # An interrupted hf_hub download leaves .incomplete files behind.
    incomplete = list(path.rglob("*.incomplete"))
    if incomplete:
        problems.append(f"{len(incomplete)} incomplete download(s) present")

    return not problems, problems


def fetch(cfg: Config, model: ModelConfig, force: bool = False) -> FetchResult:
    """Download a model snapshot, then stage it to local disk."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot = Path(
        snapshot_download(
            repo_id=model.repo_id,
            allow_patterns=ALLOW_PATTERNS,
            resume_download=True,
            max_workers=4,
        )
    )

    ok, problems = verify_snapshot(snapshot)
    result = FetchResult(
        model=model.name,
        repo_id=model.repo_id,
        snapshot_dir=str(snapshot),
        staged_dir="",
        n_shards=len(_shards(snapshot)),
        total_gb=round(_dir_gb(snapshot), 2),
        warnings=[] if ok else problems,
    )
    if not ok:
        return result

    staged = stage_to_local(cfg, model, snapshot, force=force)
    result.staged_dir = str(staged)
    result.staged = True
    return result


def stage_to_local(
    cfg: Config, model: ModelConfig, snapshot: Path, force: bool = False
) -> Path:
    """Copy a snapshot to ``paths.stage_dir`` for fast loading.

    Skipped when the cache is already on local disk -- there is nothing to gain
    from a second copy of 16GB.
    """
    if not cfg.paths.caching_on_drive:
        return snapshot

    target = cfg.paths.stage_path / model.stage_name

    if target.exists() and not force:
        ok, _ = verify_snapshot(target)
        if ok:
            return target
        shutil.rmtree(target, ignore_errors=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        # rsync resumes a partial staging copy; a 16GB cp that dies at 90%
        # otherwise costs the whole transfer again.
        subprocess.run(
            ["rsync", "-a", "--partial", f"{snapshot}/", f"{target}/"],
            check=True,
        )
    else:
        shutil.copytree(snapshot, target, dirs_exist_ok=True)
    return target


def local_path(cfg: Config, model: ModelConfig) -> Path:
    """Where an adapter should load this model from.

    Prefers the staged local copy, falls back to the HF cache, and finally to
    the bare repo ID so ``from_pretrained`` can resolve it itself.
    """
    staged = cfg.paths.stage_path / model.stage_name
    if staged.exists() and verify_snapshot(staged)[0]:
        return staged

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        return Path(
            snapshot_download(
                repo_id=model.repo_id,
                allow_patterns=ALLOW_PATTERNS,
                local_files_only=True,
            )
        )
    except Exception:
        return Path(model.repo_id)


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    from .config import load_config  # noqa: PLC0415
    from .env import apply_env, check_cache_space  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Fetch and stage model weights.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", default=None, help="fetch one model (default: all enabled)")
    parser.add_argument("--force", action="store_true", help="re-stage even if present")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, cli_overrides=args.overrides)
    hf_home = apply_env(cfg)
    print(f"HF_HOME: {hf_home}  (cache_mode: {cfg.paths.cache_mode})")
    for warning in check_cache_space(cfg):
        print(f"WARNING: {warning}")

    targets = {args.model: cfg.model(args.model)} if args.model else cfg.enabled_models
    if not targets:
        print("no enabled models in config; nothing to fetch")
        return 1

    failed = False
    for name, model in targets.items():
        print(f"\n=== {name}  ({model.repo_id}) ===")
        result = fetch(cfg, model, force=args.force)
        print(f"  snapshot : {result.snapshot_dir}")
        print(f"  shards   : {result.n_shards}  ({result.total_gb} GB)")
        print(f"  staged   : {result.staged_dir or '(not staged)'}")
        for warning in result.warnings:
            print(f"  PROBLEM  : {warning}")
            failed = True

    if failed:
        print("\nsome snapshots are incomplete; re-run to resume the download")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
