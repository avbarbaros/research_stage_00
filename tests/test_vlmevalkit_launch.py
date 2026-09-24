"""vlmevalkit_launch.py: turns model kwargs into a run.py --config JSON.

No VLMEvalKit installed: a stub ``vlmeval`` stands in for the two registries
the launcher reads, and ``os.execv`` is captured instead of replacing the test
process.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from mmro.harness import vlmevalkit_launch as launch


class MMMUDataset:
    @staticmethod
    def supported_datasets():
        return ["MMMU_DEV_VAL", "MMMU_TEST"]


@pytest.fixture
def fake_vlmeval(monkeypatch):
    pkg = types.ModuleType("vlmeval")
    dataset = types.ModuleType("vlmeval.dataset")
    dataset.DATASET_CLASSES = [MMMUDataset]
    dataset.MMMUDataset = MMMUDataset
    config = types.ModuleType("vlmeval.config")
    config.supported_VLM = {"Qwen2.5-VL-7B-Instruct": object()}
    pkg.dataset, pkg.config = dataset, config
    for name, mod in (("vlmeval", pkg), ("vlmeval.dataset", dataset),
                      ("vlmeval.config", config)):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def execv(monkeypatch):
    calls = []
    monkeypatch.setattr(launch.os, "execv", lambda exe, argv: calls.append(argv))
    # setenv first so teardown restores the original, even though the launcher
    # sets the variable itself.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD")
    return calls


def run(monkeypatch, tmp_path, kwargs, data="MMMU_DEV_VAL"):
    out = tmp_path / "cfg" / "c.json"
    monkeypatch.setattr(sys, "argv", [
        "vlmevalkit_launch.py", "/repo/run.py",
        "--model", "Qwen2.5-VL-7B-Instruct", "--data", data,
        "--model-kwargs", json.dumps(kwargs), "--config-out", str(out),
        "--work-dir", "/w", "--judge", "gpt-4o-mini",
    ])
    launch.main()
    return out


def test_kwargs_become_a_config_and_judge_args_pass_through(
    fake_vlmeval, execv, monkeypatch, tmp_path
):
    out = run(monkeypatch, tmp_path, {"use_vllm": True})
    assert json.loads(out.read_text()) == {
        "model": {"Qwen2.5-VL-7B-Instruct": {"use_vllm": True}},
        "data": {"MMMU_DEV_VAL": {"class": "MMMUDataset", "dataset": "MMMU_DEV_VAL"}},
    }
    argv = execv[0]
    assert argv[1:] == ["/repo/run.py", "--config", str(out),
                        "--work-dir", "/w", "--judge", "gpt-4o-mini"]
    assert "--use-vllm" not in argv
    assert launch.os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_no_kwargs_keeps_the_plain_cli_form(execv, monkeypatch, tmp_path):
    # No stub: the plain path must not even import vlmeval.
    out = run(monkeypatch, tmp_path, {})
    assert not out.exists()
    assert execv[0][1:6] == ["/repo/run.py", "--model", "Qwen2.5-VL-7B-Instruct",
                             "--data", "MMMU_DEV_VAL"]
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in launch.os.environ


def test_unknown_dataset_fails_before_launch(fake_vlmeval, execv, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="no VLMEvalKit dataset class"):
        run(monkeypatch, tmp_path, {"use_vllm": True}, data="NOPE")
    assert execv == []
