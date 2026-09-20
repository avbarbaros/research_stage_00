"""Weight staging tests.

The regression these exist for: a Hugging Face snapshot directory is a farm of
*relative symlinks* into ``../../blobs/``. Staging it with plain ``rsync -a``
copies the links rather than the weights, producing a directory that looks
populated and contains nothing. Loading then falls back to reading 16GB across
Drive at ~44 MB/s -- six minutes per load, silently, with correct captions at
the end so nothing looks wrong.

So the fixtures below build a real symlink farm rather than plain files. A test
using ordinary files would pass against the broken implementation.
"""

from __future__ import annotations

import json
import shutil
import struct

import pytest
import yaml

from mmro.cache import check_shard, resolve_load_path, stage_to_local, verify_snapshot
from mmro.config import EXAMPLE_CONFIG_PATH, load_config


def safetensors_bytes(payload_size: int) -> bytes:
    """A minimal valid safetensors file: one uint8 tensor of ``payload_size``."""
    header = json.dumps(
        {"w": {"dtype": "U8", "shape": [payload_size], "data_offsets": [0, payload_size]}}
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * payload_size


def make_hub_snapshot(root, shards=2, blob_size=2048):
    """A cache laid out the way huggingface_hub actually lays one out."""
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    def link(blob_name, link_name, payload: bytes):
        (blobs / blob_name).write_bytes(payload)
        (snapshot / link_name).symlink_to(f"../../blobs/{blob_name}")

    link("cfg", "config.json", json.dumps({"model_type": "test"}).encode())
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    for i, name in enumerate(names):
        link(f"blob{i}", name, safetensors_bytes(blob_size))
    link(
        "idx",
        "model.safetensors.index.json",
        json.dumps({"weight_map": {f"layer.{i}": n for i, n in enumerate(names)}}).encode(),
    )
    return snapshot


@pytest.fixture
def cfg(tmp_path):
    data = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())
    data["paths"].update(
        cache_mode="drive",
        hf_home=str(tmp_path / "drive_cache"),
        local_hf_home=str(tmp_path / "local_cache"),
        stage_dir=str(tmp_path / "stage"),
        images_dir=str(tmp_path / "img"),
        results_dir=str(tmp_path / "res"),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


# --- the actual regression --------------------------------------------------
def test_staging_dereferences_symlinks(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    model = cfg.model("qwen2_5_vl_7b")

    staged = stage_to_local(cfg, model, snapshot)

    for entry in staged.iterdir():
        assert not entry.is_symlink(), (
            f"{entry.name} was staged as a symlink; it will dangle outside the "
            f"hub cache and loading will silently fall back to the slow mount"
        )
        assert entry.stat().st_size > 0


def test_staged_copy_holds_the_real_weights(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub", shards=3, blob_size=4096)
    staged = stage_to_local(cfg, model := cfg.model("qwen2_5_vl_7b"), snapshot)

    shards = sorted(staged.glob("*.safetensors"))
    assert len(shards) == 3
    assert all(s.read_bytes() == safetensors_bytes(4096) for s in shards)
    assert json.loads((staged / "config.json").read_text())["model_type"] == "test"
    assert staged.name == model.stage_name


def test_staged_copy_verifies_clean(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    staged = stage_to_local(cfg, cfg.model("qwen2_5_vl_7b"), snapshot)
    ok, problems = verify_snapshot(staged)
    assert ok, problems


def test_staging_is_idempotent(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    model = cfg.model("qwen2_5_vl_7b")
    first = stage_to_local(cfg, model, snapshot)
    second = stage_to_local(cfg, model, snapshot)
    assert first == second
    assert verify_snapshot(second)[0]


def test_a_broken_staged_copy_is_replaced_not_reused(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    model = cfg.model("qwen2_5_vl_7b")
    target = cfg.paths.stage_path / model.stage_name

    # Simulate the damage the old rsync -a caused.
    target.mkdir(parents=True)
    (target / "config.json").symlink_to("../../blobs/gone")

    staged = stage_to_local(cfg, model, snapshot)
    assert verify_snapshot(staged)[0]
    assert not (staged / "config.json").is_symlink()


def test_no_staging_when_cache_is_already_local(cfg, tmp_path):
    # A second 16GB copy of local files buys nothing.
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    data["paths"]["cache_mode"] = "local"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data))
    local_cfg = load_config(tmp_path / "config.yaml")

    snapshot = make_hub_snapshot(tmp_path / "hub")
    assert stage_to_local(local_cfg, local_cfg.model("qwen2_5_vl_7b"), snapshot) == snapshot


# --- verification catches partial snapshots ---------------------------------
def test_missing_shard_named_in_the_index_is_caught(tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub", shards=3)
    (snapshot / "model-00002-of-00003.safetensors").unlink()
    ok, problems = verify_snapshot(snapshot)
    assert not ok
    assert any("missing" in p for p in problems)


def test_incomplete_download_is_caught(tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    (snapshot / "model-00001-of-00002.safetensors.incomplete").write_bytes(b"partial")
    ok, problems = verify_snapshot(snapshot)
    assert not ok
    assert any("incomplete" in p for p in problems)


def test_missing_config_is_caught(tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    (snapshot / "config.json").unlink()
    ok, problems = verify_snapshot(snapshot)
    assert not ok
    assert any("config.json" in p for p in problems)


def test_absent_directory_is_caught(tmp_path):
    ok, problems = verify_snapshot(tmp_path / "nothing")
    assert not ok and problems


# --- corrupt shards: the "header too large" failure --------------------------
def test_valid_shard_passes(tmp_path):
    f = tmp_path / "ok.safetensors"
    f.write_bytes(safetensors_bytes(512))
    assert check_shard(f) is None


@pytest.mark.parametrize(
    "payload",
    [
        b"",                                                    # zero-byte file
        b"version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n",  # LFS pointer
        b"\xff" * 4096,                                          # garbage
        safetensors_bytes(512)[:-100],                          # truncated tensor data
    ],
    ids=["empty", "lfs-pointer", "garbage", "truncated"],
)
def test_corrupt_shard_is_caught(tmp_path, payload):
    f = tmp_path / "bad.safetensors"
    f.write_bytes(payload)
    assert check_shard(f)


def test_snapshot_with_a_corrupt_shard_fails_verification(tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    (tmp_path / "hub" / "blobs" / "blob1").write_bytes(b"\xff" * 4096)
    ok, problems = verify_snapshot(snapshot)
    assert not ok
    assert any("corrupt shard" in p for p in problems)


# --- load-path resolution must be honest about speed ------------------------
def test_staged_copy_resolves_as_fast(cfg, tmp_path):
    snapshot = make_hub_snapshot(tmp_path / "hub")
    model = cfg.model("qwen2_5_vl_7b")
    stage_to_local(cfg, model, snapshot)

    path, description, is_fast = resolve_load_path(cfg, model)
    assert is_fast
    assert "staged" in description
    assert path == cfg.paths.stage_path / model.stage_name


def test_unstaged_drive_cache_resolves_as_slow(cfg):
    # The exact situation that cost six minutes per load: nothing staged, so
    # weights come off the slow mount. It must not report itself as fine.
    _, description, is_fast = resolve_load_path(cfg, cfg.model("qwen2_5_vl_7b"))
    assert not is_fast
    assert "not staged" in description or "download" in description


def test_broken_staged_copy_resolves_as_slow(cfg, tmp_path):
    model = cfg.model("qwen2_5_vl_7b")
    target = cfg.paths.stage_path / model.stage_name
    target.mkdir(parents=True)
    (target / "config.json").symlink_to("../../blobs/gone")

    _, _, is_fast = resolve_load_path(cfg, model)
    assert not is_fast


def test_copytree_fallback_also_dereferences(cfg, tmp_path, monkeypatch):
    # Colab has rsync, but the fallback must not reintroduce the bug.
    monkeypatch.setattr(shutil, "which", lambda _: None)
    snapshot = make_hub_snapshot(tmp_path / "hub")
    staged = stage_to_local(cfg, cfg.model("qwen2_5_vl_7b"), snapshot)
    assert not any(e.is_symlink() for e in staged.iterdir())
    assert verify_snapshot(staged)[0]
