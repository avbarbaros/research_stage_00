"""Stage-0.2 seam tests — milestone criterion 6.

Proves, on CPU with no weights, that the architecture is actually ready for
VLMEvalKit and lmms-eval rather than merely claimed to be:

* every adapter exposes the bare ``generate(images, text, decode) -> str``
  contract both harnesses reduce to, and it really is the same code path the
  instrumented stage-0 form uses;
* a canonical task name resolves to each harness's own task ID from config
  alone;
* the adapter package stays importable without the harnesses' conflicting
  dependencies.

A fake adapter stands in for the real models: the point is the seam, and it must
be testable without 16.5GB of weights.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mmro.config import ConfigError, EXAMPLE_CONFIG_PATH, load_config
from mmro.harness import BACKENDS, get_backend_class, resolve_task_id
from mmro.harness.base import HarnessBackend, task_matrix
from mmro.models import ADAPTERS, GenOutput, get_adapter_class
from mmro.models.base import VLMAdapter
from mmro.records import GenerationResult


@pytest.fixture
def cfg():
    return load_config(EXAMPLE_CONFIG_PATH)


class FakeAdapter(VLMAdapter):
    """Minimal adapter: records what it was asked, returns a canned caption."""

    def __init__(self, cfg, model_cfg):
        super().__init__(cfg, model_cfg)
        self.calls: list[tuple[list[Path], str, dict]] = []

    def _load_impl(self, model_path, budget):
        self.model = object()

    def _generate_impl(self, images, text, decode, budget):
        self.calls.append((list(images), text, dict(budget)))
        return GenOutput(
            text=f"a caption about {len(images)} image(s)",
            visual_tokens=100 * len(images),
            prompt_tokens=42,
            completion_tokens=7,
        )

    def lora_target_modules(self):
        return ["q_proj", "v_proj"]


@pytest.fixture
def fake(cfg):
    adapter = FakeAdapter(cfg, cfg.model("qwen2_5_vl_7b"))
    adapter.load()
    return adapter


# ---------------------------------------------------------------------------
# The harness contract
# ---------------------------------------------------------------------------
def test_generate_returns_a_bare_string(fake):
    # This exact shape is what VLMEvalKit's generate_inner and lmms-eval's
    # generate_until both need.
    out = fake.generate([Path("a.jpg")], "Describe this image.")
    assert isinstance(out, str)
    assert out


def test_every_registered_adapter_implements_the_contract():
    for name in ADAPTERS:
        cls = get_adapter_class(name)
        assert issubclass(cls, VLMAdapter)
        assert not inspect.isabstract(cls), f"{name} leaves abstract methods unimplemented"


def test_generate_signature_is_harness_compatible():
    sig = inspect.signature(VLMAdapter.generate)
    assert list(sig.parameters)[:3] == ["self", "images", "text"]
    # decode and budget must stay optional: the harness wrappers call with
    # positional images and text only.
    assert sig.parameters["decode"].default is None
    assert sig.parameters["budget"].default is None


def test_generate_and_generate_measured_share_one_code_path(fake):
    bare = fake.generate([Path("a.jpg")], "prompt")
    measured = fake.generate_measured([Path("a.jpg")], "prompt")
    assert measured.text == bare
    # Both went through _generate_impl with identical arguments -- so what stage
    # 0 measures is what the harnesses will run.
    assert fake.calls[0] == fake.calls[1]


def test_generate_measured_returns_a_full_record(fake):
    result = fake.generate_measured(
        [Path("a.jpg"), Path("b.jpg")],
        "compare",
        estimated_visual_tokens=200,
        rung=1280,
        image_provenance=["a.jpg:100x100:abc", "b.jpg:100x100:def"],
    )
    assert isinstance(result, GenerationResult)
    assert result.ok
    assert result.n_images == 2
    assert result.visual_tokens == 200
    assert result.estimated_visual_tokens == 200
    assert result.token_estimate_matches is True
    assert result.rung == 1280
    assert result.decode == {"do_sample": False, "max_new_tokens": 256}


def test_estimate_mismatch_is_visible_in_the_record(fake):
    result = fake.generate_measured([Path("a.jpg")], "x", estimated_visual_tokens=999)
    assert result.token_estimate_matches is False


def test_budget_override_reaches_the_adapter(fake):
    fake.generate([Path("a.jpg")], "x", budget={"max_visual_tokens_per_image": 4096})
    assert fake.calls[-1][2] == {"max_visual_tokens_per_image": 4096}


def test_adapter_errors_are_recorded_not_raised(cfg):
    class Broken(FakeAdapter):
        def _generate_impl(self, images, text, decode, budget):
            raise RuntimeError("processor exploded")

    adapter = Broken(cfg, cfg.model("qwen2_5_vl_7b"))
    adapter.load()
    result = adapter.generate_measured([Path("a.jpg")], "x")
    assert not result.ok
    assert "processor exploded" in result.error
    assert result.oom is False


def test_lora_targets_declared_for_every_adapter(cfg):
    # Stage-1 scaffold: an adapter with no declared targets would produce a PEFT
    # model with zero trainable parameters.
    for name, model_cfg in cfg.models.items():
        cls = get_adapter_class(model_cfg.adapter)
        targets = cls(cfg, model_cfg).lora_target_modules()
        assert targets and all(isinstance(t, str) for t in targets)


# ---------------------------------------------------------------------------
# Task-ID resolution across harnesses
# ---------------------------------------------------------------------------
def test_canonical_task_resolves_to_both_harness_ids(cfg):
    # The headline criterion: one config row, two harness-specific IDs.
    assert resolve_task_id(cfg, "mathvista_testmini", "vlmevalkit") == "MathVista_MINI"
    assert resolve_task_id(cfg, "mathvista_testmini", "lmms_eval") == "mathvista_testmini"
    assert resolve_task_id(cfg, "mmmu_val", "vlmevalkit") == "MMMU_DEV_VAL"
    assert resolve_task_id(cfg, "mmmu_val", "lmms_eval") == "mmmu_val"


def test_every_configured_task_resolves_for_every_enabled_harness(cfg):
    for row, harness in task_matrix(cfg):
        assert row.id_for(harness)


def test_unknown_task_names_the_configured_ones(cfg):
    with pytest.raises(ConfigError, match="no task named"):
        resolve_task_id(cfg, "mmbench", "vlmevalkit")


def test_unknown_harness_rejected(cfg):
    with pytest.raises(ConfigError, match="invalid value"):
        resolve_task_id(cfg, "mmmu_val", "lm_eval_harness")


def test_missing_task_id_for_a_harness_is_an_error(tmp_path):
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())
    data["eval"]["tasks"] = [{"name": "mmmu_val", "vlmevalkit": "MMMU_DEV_VAL"}]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))

    partial = load_config(path)
    assert resolve_task_id(partial, "mmmu_val", "vlmevalkit") == "MMMU_DEV_VAL"
    # Silently skipping the missing side would compare one harness against
    # nothing and look like agreement.
    with pytest.raises(ConfigError, match="no task ID for harness"):
        resolve_task_id(partial, "mmmu_val", "lmms_eval")


# ---------------------------------------------------------------------------
# Harness backends
# ---------------------------------------------------------------------------
def test_both_backends_are_registered_and_constructible(cfg):
    assert set(BACKENDS) == {"vlmevalkit", "lmms_eval"}
    for name in BACKENDS:
        backend = get_backend_class(name)(cfg)
        assert isinstance(backend, HarnessBackend)
        assert backend.name == name


def test_backends_refuse_cleanly_when_the_venv_is_missing(cfg):
    # Implemented as of stage 0.2. With no venv built they must fail fast with
    # an actionable message rather than shelling out to a missing interpreter.
    for name in BACKENDS:
        backend = get_backend_class(name)(cfg)
        with pytest.raises(RuntimeError, match="venv is missing"):
            backend.run("qwen2_5_vl_7b", cfg.eval.task("mmmu_val"))


def test_backends_report_a_missing_venv(cfg):
    # The venvs do not exist at stage 0, and the message must point at the
    # script that creates them.
    problems = get_backend_class("vlmevalkit")(cfg).check_available()
    assert any("venv is missing" in p for p in problems)


def test_backends_use_separate_venvs(cfg):
    a = get_backend_class("vlmevalkit")(cfg).python()
    b = get_backend_class("lmms_eval")(cfg).python()
    assert a != b


def test_both_backends_echo_identical_decoding(cfg):
    # Comparing scores produced under different decoding settings would measure
    # the decoding, not the harnesses.
    a = get_backend_class("vlmevalkit")(cfg).decode_echo()
    b = get_backend_class("lmms_eval")(cfg).decode_echo()
    assert a == b
    assert a["do_sample"] is False


# ---------------------------------------------------------------------------
# Dependency isolation
# ---------------------------------------------------------------------------
def test_model_package_imports_without_harness_dependencies():
    # Both harnesses import mmro.models from their own venvs, where the base
    # environment's extras are absent.
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", "import mmro.models, mmro.config, mmro.records"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr


def test_adapter_modules_do_not_import_torch_at_module_level():
    # Heavy imports must stay inside functions so the registry can be inspected
    # -- and a harness venv can import this package -- without loading torch.
    import ast  # noqa: PLC0415

    src = Path(__file__).resolve().parents[1] / "src" / "mmro"
    for module, _ in ADAPTERS.values():
        path = src / f"{module.split('.')[-1]}.py"
        if not path.exists():
            path = src / "models" / f"{module.split('.')[-1]}.py"
        tree = ast.parse(path.read_text())
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {
            alias.name.split(".")[0]
            for node in top_level
            for alias in getattr(node, "names", [])
        } | {
            (node.module or "").split(".")[0]
            for node in top_level
            if isinstance(node, ast.ImportFrom)
        }
        assert "torch" not in names, f"{path.name} imports torch at module level"
        assert "transformers" not in names, f"{path.name} imports transformers at module level"
