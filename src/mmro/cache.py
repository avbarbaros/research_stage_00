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
            max_workers=4,
        )
    )  # resuming is the default now; passing resume_download only warns

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
        # -L (--copy-links) is load-bearing, not a nicety. A hub snapshot
        # directory is a farm of RELATIVE symlinks into ../../blobs/; plain
        # `rsync -a` copies the links themselves, producing a staged directory
        # of broken pointers and zero bytes of weights. Loading then silently
        # falls back to reading 16GB across Drive's FUSE mount at ~44 MB/s --
        # about six minutes per model instead of thirty seconds.
        # --partial resumes an interrupted copy rather than restarting 16GB.
        subprocess.run(
            ["rsync", "-aL", "--partial", f"{snapshot}/", f"{target}/"],
            check=True,
        )
    else:
        # symlinks=False (the default) likewise dereferences into real files.
        shutil.copytree(snapshot, target, dirs_exist_ok=True)

    ok, problems = verify_snapshot(target)
    if not ok:
        raise RuntimeError(
            f"staging to {target} produced an unusable copy: {'; '.join(problems)}. "
            f"Loading would fall back to the slow cache. Remove that directory and "
            f"re-run with --force."
        )
    return target


def resolve_load_path(cfg: Config, model: ModelConfig) -> tuple[Path, str, bool]:
    """Where an adapter will load this model from.

    Returns ``(path, description, is_fast)``. ``is_fast`` is False when we are
    about to read weights across Drive, which costs minutes per load -- and the
    sweep reloads for every cell, so it compounds badly.
    """
    staged = cfg.paths.stage_path / model.stage_name
    if staged.exists():
        ok, problems = verify_snapshot(staged)
        if ok:
            return staged, "staged local copy", True
        detail = "; ".join(problems)
    else:
        detail = "not staged"

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        cached = Path(
            snapshot_download(
                repo_id=model.repo_id,
                allow_patterns=ALLOW_PATTERNS,
                local_files_only=True,
            )
        )
        return cached, f"HF cache ({detail})", not cfg.paths.caching_on_drive
    except Exception:
        return Path(model.repo_id), f"repo ID, will download ({detail})", False


def local_path(cfg: Config, model: ModelConfig, announce: bool = True) -> Path:
    """Resolve the load path, saying out loud when it is the slow one.

    A silent fallback here is how six minutes of Drive I/O per load hides in
    plain sight: the model does load, the captions are fine, and nothing
    indicates that staging failed.
    """
    path, description, is_fast = resolve_load_path(cfg, model)
    if announce and not is_fast:
        print(
            f"WARNING: loading {model.name} from {description}: {path}\n"
            f"         This reads weights across a slow mount (expect minutes, "
            f"not seconds).\n"
            f"         Run `bash scripts/00_fetch_weights.sh --force` to stage to "
            f"{cfg.paths.stage_path} first -- it re-copies from the existing cache "
            f"and does not re-download."
        )
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    from .config import load_config  # noqa: PLC0415
    from .env import apply_env, check_cache_space  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Fetch and stage model weights.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", default=None, help="fetch one model (default: all enabled)")
    parser.add_argument("--force", action="store_true", help="re-stage even if present")
    parser.add_argument(
        "--status", action="store_true",
        help="report where each model would load from, and how fast, without fetching",
    )
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

    if args.status:
        slow = False
        for name, model in targets.items():
            path, description, is_fast = resolve_load_path(cfg, model)
            print(f"\n{name}:")
            print(f"  loads from : {path}")
            print(f"  source     : {description}")
            print(f"  speed      : {'fast (local disk)' if is_fast else 'SLOW'}")
            slow = slow or not is_fast
        if slow:
            print(
                "\nAt least one model would load across a slow mount. Every sweep "
                "cell reloads,\nso this multiplies: run "
                "`bash scripts/00_fetch_weights.sh --force` to stage locally."
            )
        return 1 if slow else 0

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
