"""Stage-0.2 eval tests (no GPU, no harness installed).

Covers the parts that decide whether a benchmark run is trustworthy:

* the per-task visual budget, which is the most likely silent cause of a
  below-baseline score;
* result parsing for both harnesses, using fixtures shaped like their real
  output;
* the gate itself, and the diagnosis it offers when it fails.
"""

from __future__ import annotations

import json

import pytest
import yaml

from mmro.config import ConfigError, EXAMPLE_CONFIG_PATH, load_config
from mmro.eval import check_gate, diagnose
from mmro.harness import get_backend_class
from mmro.harness.base import budget_for_task
from mmro.records import ScoreRecord


@pytest.fixture
def cfg(tmp_path):
    data = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())
    data["paths"].update(
        cache_mode="local", local_hf_home=str(tmp_path / "h"), stage_dir=str(tmp_path / "s"),
        images_dir=str(tmp_path / "i"), results_dir=str(tmp_path / "r"),
    )
    data["eval"]["enabled"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def score(task="mathvista_testmini", harness="vlmevalkit", value=68.0, metric="accuracy"):
    return ScoreRecord(
        harness=harness, model="qwen2_5_vl_7b", task=task,
        harness_task_id="X", metric=metric, score=value,
    )


# ---------------------------------------------------------------------------
# Per-task budget — the quiet accuracy killer
# ---------------------------------------------------------------------------
def test_single_image_task_gets_the_whole_token_ceiling(cfg):
    # MathVista is single-image, so its one image gets the full budget and runs
    # effectively at native resolution -- as the published evaluation did.
    budget = budget_for_task(cfg, "qwen2_5_vl_7b", cfg.eval.task("mathvista_testmini"))
    assert budget["max_visual_tokens_per_image"] == 20000


def test_multi_image_task_divides_the_ceiling(cfg):
    # MMMU allows 7 images, so each gets a seventh. Handing it the single-image
    # budget instead would OOM; handing MathVista this one would downscale every
    # chart for nothing.
    budget = budget_for_task(cfg, "qwen2_5_vl_7b", cfg.eval.task("mmmu_val"))
    assert budget["max_visual_tokens_per_image"] == 20000 // 7


def test_budgets_differ_between_the_two_tasks(cfg):
    a = budget_for_task(cfg, "qwen2_5_vl_7b", cfg.eval.task("mathvista_testmini"))
    b = budget_for_task(cfg, "qwen2_5_vl_7b", cfg.eval.task("mmmu_val"))
    assert a["max_visual_tokens_per_image"] > b["max_visual_tokens_per_image"]


def test_internvl_budget_converts_tokens_to_tiles(cfg):
    # InternVL is quantized to 256-token tiles, one of which is the thumbnail.
    budget = budget_for_task(cfg, "internvl3_8b_hf", cfg.eval.task("mmmu_val"))
    per_image = 14336 // 7
    assert budget["max_tiles"] == per_image // 256 - 1
    assert (budget["max_tiles"] + 1) * 256 * 7 <= 14336   # inside the cliff


def test_budget_falls_back_when_no_ceiling_is_configured(tmp_path):
    data = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())
    del data["models"]["qwen2_5_vl_7b"]["budget"]["safe_total_visual_tokens"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    c = load_config(path)
    budget = budget_for_task(c, "qwen2_5_vl_7b", c.eval.task("mmmu_val"))
    assert budget["max_visual_tokens_per_image"] == 2560   # the configured value
    assert "safe_total_visual_tokens" not in budget


# ---------------------------------------------------------------------------
# Command templating
# ---------------------------------------------------------------------------
def test_command_renders_with_task_and_model(cfg):
    backend = get_backend_class("vlmevalkit")(cfg)
    argv = cfg.eval.harnesses["vlmevalkit"].render_command(
        **backend.command_fields("qwen2_5_vl_7b", cfg.eval.task("mmmu_val"))
    )
    assert "MMMU_DEV_VAL" in argv
    assert "Qwen2.5-VL-7B-Instruct" in argv


def test_each_harness_gets_its_own_task_id(cfg):
    task = cfg.eval.task("mathvista_testmini")
    a = get_backend_class("vlmevalkit")(cfg).command_fields("qwen2_5_vl_7b", task)
    b = get_backend_class("lmms_eval")(cfg).command_fields("qwen2_5_vl_7b", task)
    assert a["task_id"] == "MathVista_MINI"
    assert b["task_id"] == "mathvista_testmini"


def test_unknown_placeholder_is_a_config_error(cfg):
    cfg.eval.harnesses["vlmevalkit"].command = "{python} --nope {not_a_field}"
    with pytest.raises(ConfigError, match="unknown placeholder"):
        cfg.eval.harnesses["vlmevalkit"].render_command(python="p")


# ---------------------------------------------------------------------------
# Result parsing — fixtures shaped like the real thing
# ---------------------------------------------------------------------------
def test_vlmevalkit_parses_an_accuracy_csv(cfg, tmp_path):
    work = tmp_path / "vlmevalkit"
    work.mkdir()
    (work / "Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_acc.csv").write_text(
        "split,Overall,Art\nvalidation,58.4,61.0\n"
    )
    backend = get_backend_class("vlmevalkit")(cfg)
    got = backend.parse_results(work, "qwen2_5_vl_7b", cfg.eval.task("mmmu_val"), "MMMU_DEV_VAL")
    assert len(got) == 1
    assert got[0].score == pytest.approx(58.4)
    assert got[0].metric == "accuracy"
    assert got[0].harness_task_id == "MMMU_DEV_VAL"


def test_vlmevalkit_converts_a_fraction_to_percent(cfg, tmp_path):
    work = tmp_path / "v"
    work.mkdir()
    (work / "m_MathVista_MINI_acc.csv").write_text("split,Overall\ntestmini,0.682\n")
    backend = get_backend_class("vlmevalkit")(cfg)
    got = backend.parse_results(
        work, "qwen2_5_vl_7b", cfg.eval.task("mathvista_testmini"), "MathVista_MINI"
    )
    assert got[0].score == pytest.approx(68.2)


def test_vlmevalkit_ignores_other_tasks_files(cfg, tmp_path):
    work = tmp_path / "v"
    work.mkdir()
    (work / "m_MathVista_MINI_acc.csv").write_text("split,Overall\ntestmini,68.2\n")
    backend = get_backend_class("vlmevalkit")(cfg)
    assert backend.parse_results(
        work, "qwen2_5_vl_7b", cfg.eval.task("mmmu_val"), "MMMU_DEV_VAL"
    ) == []


def test_lmms_eval_parses_results_json(cfg, tmp_path):
    work = tmp_path / "lmms"
    work.mkdir()
    (work / "results_2026.json").write_text(json.dumps({
        "results": {"mmmu_val": {"alias": "mmmu_val", "mmmu_acc,none": 0.584}},
        "n-samples": {"mmmu_val": {"original": 900, "effective": 900}},
    }))
    backend = get_backend_class("lmms_eval")(cfg)
    got = backend.parse_results(work, "qwen2_5_vl_7b", cfg.eval.task("mmmu_val"), "mmmu_val")
    assert got[0].score == pytest.approx(58.4)
    assert got[0].n_samples == 900


def test_lmms_eval_strips_the_filter_suffix(cfg, tmp_path):
    work = tmp_path / "l"
    work.mkdir()
    (work / "results.json").write_text(json.dumps(
        {"results": {"mathvista_testmini": {"exact_match,none": 0.681, "alias": "x"}}}
    ))
    backend = get_backend_class("lmms_eval")(cfg)
    got = backend.parse_results(
        work, "qwen2_5_vl_7b", cfg.eval.task("mathvista_testmini"), "mathvista_testmini"
    )
    assert [r.metric for r in got] == ["exact_match"]


def test_parsed_scores_carry_decode_and_budget(cfg, tmp_path):
    # The whole point of the ScoreRecord fields: prove both harnesses ran under
    # identical conditions before comparing them.
    work = tmp_path / "v"
    work.mkdir()
    (work / "m_MathVista_MINI_acc.csv").write_text("split,Overall\ntestmini,68.2\n")
    backend = get_backend_class("vlmevalkit")(cfg)
    got = backend.parse_results(
        work, "qwen2_5_vl_7b", cfg.eval.task("mathvista_testmini"), "MathVista_MINI"
    )[0]
    assert got.decode["do_sample"] is False
    assert got.budget["max_visual_tokens_per_image"] == 20000


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_gate_passes_within_tolerance(cfg):
    gate = check_gate(cfg, [score(value=68.2 - 1.5)])
    assert gate[0].passed is True
    assert gate[0].baseline == pytest.approx(68.2)


def test_gate_fails_outside_tolerance(cfg):
    gate = check_gate(cfg, [score(value=68.2 - 6.0)])
    assert gate[0].passed is False
    assert gate[0].delta == pytest.approx(-6.0)


def test_gate_is_symmetric(cfg):
    # Scoring far ABOVE published is equally suspect -- usually a split or
    # subset mismatch, not a better model.
    assert check_gate(cfg, [score(value=68.2 + 6.0)])[0].passed is False


def test_gate_tolerance_comes_from_config(cfg):
    cfg.eval.gate_tolerance = 10.0
    assert check_gate(cfg, [score(value=68.2 - 6.0)])[0].passed is True


def test_gate_skips_non_headline_metrics(cfg):
    assert check_gate(cfg, [score(metric="art_and_design")]) == []


def test_gate_reports_missing_baseline_rather_than_passing(cfg):
    cfg.model("qwen2_5_vl_7b").baselines["mathvista_testmini"] = None
    g = check_gate(cfg, [score()])[0]
    assert g.passed is None and g.delta is None


def test_gate_ignores_unknown_tasks(cfg):
    assert check_gate(cfg, [score(task="mmbench")]) == []


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def test_no_diagnosis_when_the_gate_passes(cfg):
    assert diagnose(cfg, check_gate(cfg, [score(value=68.0)])) == []


def test_judge_is_named_first_when_unset(cfg):
    # The most common cause of a low score, and it is pure configuration.
    cfg.eval.judge.backend = "none"
    notes = diagnose(cfg, check_gate(cfg, [score(value=60.0)]))
    assert "judge" in notes[0].lower()


def test_low_scores_point_at_resolution_and_split(cfg):
    notes = " ".join(diagnose(cfg, check_gate(cfg, [score(value=60.0)])))
    assert "max_images" in notes
    assert "MMMU_DEV_VAL" in notes


def test_subset_runs_are_called_out(cfg):
    cfg.eval.limit = 25
    notes = " ".join(diagnose(cfg, check_gate(cfg, [score(value=60.0)])))
    assert "subset" in notes


def test_a_huge_miss_suggests_the_wrong_thing_ran(cfg):
    notes = " ".join(diagnose(cfg, check_gate(cfg, [score(value=20.0)])))
    assert "wrong checkpoint" in notes or "wrong thing ran" in notes


# ---------------------------------------------------------------------------
# Harness environment fields (stage-0.2 setup)
# ---------------------------------------------------------------------------
def test_both_harnesses_declare_an_import_probe(cfg):
    # Both import large registries at package-import time, so one missing
    # optional dep stops a run before it reaches the task. The probe is what
    # surfaces that at setup instead of at launch.
    for name, h in cfg.eval.harnesses.items():
        assert h.import_probe, f"{name} has no import_probe"


def test_lmms_eval_probe_imports_the_model_module_not_just_the_package(cfg):
    # `import lmms_eval` alone succeeds; it is the qwen2_5_vl model module that
    # pulls decord and fails, so the probe has to reach that far.
    probe = cfg.eval.harnesses["lmms_eval"].import_probe
    assert "models" in probe and "qwen2_5_vl" in probe


def test_python_bin_defaults_to_empty(cfg):
    # Empty means "whatever `python` is"; it exists as an escape hatch for
    # dependencies with no wheel for the default interpreter.
    for h in cfg.eval.harnesses.values():
        assert h.python_bin == ""


def test_python_bin_is_settable(tmp_path):
    import yaml as _yaml
    from mmro.config import EXAMPLE_CONFIG_PATH as EX, load_config as lc
    data = _yaml.safe_load(EX.read_text())
    data["eval"]["harnesses"]["lmms_eval"]["python_bin"] = "python3.11"
    p = tmp_path / "c.yaml"; p.write_text(_yaml.safe_dump(data))
    assert lc(p).eval.harnesses["lmms_eval"].python_bin == "python3.11"


# ---------------------------------------------------------------------------
# Judge wiring
# ---------------------------------------------------------------------------
def _fields(cfg, harness):
    return get_backend_class(harness)(cfg).command_fields(
        "qwen2_5_vl_7b", cfg.eval.task("mmmu_val")
    )


def test_api_judge_reaches_the_vlmevalkit_command(cfg):
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    argv = cfg.eval.harnesses["vlmevalkit"].render_command(**_fields(cfg, "vlmevalkit"))
    assert argv[argv.index("--judge") + 1] == "gpt-4o-mini"


def test_no_judge_adds_no_flag(cfg):
    cfg.eval.judge.backend = "none"
    argv = cfg.eval.harnesses["vlmevalkit"].render_command(**_fields(cfg, "vlmevalkit"))
    assert "--judge" not in argv


def test_api_judge_without_a_key_is_a_hard_problem(cfg, monkeypatch):
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    cfg.eval.judge.api_key_env = "MMRO_TEST_KEY"
    monkeypatch.delenv("MMRO_TEST_KEY", raising=False)
    assert any("no key" in p for p in get_backend_class("lmms_eval")(cfg).judge_problems())


def test_key_never_appears_in_argv_but_reaches_the_environment(cfg, monkeypatch):
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    cfg.eval.judge.api_key_env = "MMRO_TEST_KEY"
    monkeypatch.setenv("MMRO_TEST_KEY", "sk-secret")
    for name in ("vlmevalkit", "lmms_eval"):
        backend = get_backend_class(name)(cfg)
        argv = cfg.eval.harnesses[name].render_command(**_fields(cfg, name))
        assert "sk-secret" not in " ".join(argv)
        assert backend.judge_env()["OPENAI_API_KEY"] == "sk-secret"
    assert get_backend_class("lmms_eval")(cfg).judge_env()["MODEL_VERSION"] == "gpt-4o-mini"


def test_vlmevalkit_reads_its_key_from_dot_env(cfg, monkeypatch, tmp_path):
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg.eval.harnesses["vlmevalkit"].repo_dir = str(tmp_path)
    backend = get_backend_class("vlmevalkit")(cfg)
    assert not backend.judge_key_present()
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-abc\n")
    assert backend.judge_key_present()


def test_model_path_is_the_verified_local_copy_once_resolved(cfg, tmp_path):
    # Handing lmms-eval the bare repo ID makes it re-resolve into the HF cache,
    # which is where a corrupt shard hides ("header too large").
    backend = get_backend_class("lmms_eval")(cfg)
    task = cfg.eval.task("mmmu_val")
    assert backend.command_fields("qwen2_5_vl_7b", task)["model_path"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    backend.model_paths["qwen2_5_vl_7b"] = tmp_path / "staged"
    assert backend.command_fields("qwen2_5_vl_7b", task)["model_path"] == str(tmp_path / "staged")


# ---------------------------------------------------------------------------
# Judge requirement is per task and harness
# ---------------------------------------------------------------------------
def test_judge_free_task_needs_no_key(cfg, monkeypatch):
    # lmms-eval scores mmmu_val by rule-based parsing, so an api judge with no
    # key must not block it.
    monkeypatch.delenv("MMRO_TEST_KEY", raising=False)
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    cfg.eval.judge.api_key_env = "MMRO_TEST_KEY"
    backend = get_backend_class("lmms_eval")(cfg)
    assert backend.judge_problems(cfg.eval.task("mmmu_val")) == []


def test_judged_tasks_still_require_a_key(cfg, monkeypatch):
    monkeypatch.delenv("MMRO_TEST_KEY", raising=False)
    cfg.eval.judge.backend, cfg.eval.judge.model = "api", "gpt-4o-mini"
    cfg.eval.judge.api_key_env = "MMRO_TEST_KEY"
    lmms = get_backend_class("lmms_eval")(cfg)
    vlm = get_backend_class("vlmevalkit")(cfg)
    assert lmms.judge_problems(cfg.eval.task("mathvista_testmini"))  # judged
    assert vlm.judge_problems(cfg.eval.task("mmmu_val"))  # only lmms_eval is judge-free
    assert lmms.judge_problems()  # no task given: stay strict


def test_judge_free_is_validated(cfg):
    from mmro.config import TaskRow

    with pytest.raises(ConfigError, match="judge_free"):
        TaskRow(name="t", lmms_eval="t", judge_free=["nope"])


def test_diagnose_skips_the_judge_note_for_judge_free_failures(cfg):
    cfg.eval.judge.backend, cfg.eval.judge.model = "none", None
    notes = " ".join(diagnose(cfg, check_gate(cfg, [score(task="mmmu_val", harness="lmms_eval", value=40.0)])))
    assert "judge.backend is 'none'" not in notes


# ---------------------------------------------------------------------------
# Per-model harness names
# ---------------------------------------------------------------------------
def test_each_model_gets_its_own_harness_name(cfg):
    task = cfg.eval.task("mmmu_val")
    fields = lambda h, m: get_backend_class(h)(cfg).command_fields(m, task)["model_id"]
    assert fields("lmms_eval", "qwen2_5_vl_7b") == "qwen2_5_vl"
    assert fields("lmms_eval", "internvl3_8b_hf") == "internvl_hf"
    assert fields("vlmevalkit", "internvl3_8b_hf") == "InternVL3-8B"


def test_missing_harness_name_is_an_error_not_a_guess(cfg):
    cfg.model("internvl3_8b_hf").harness_ids.pop("lmms_eval")
    backend = get_backend_class("lmms_eval")(cfg)
    with pytest.raises(ConfigError, match="harness_ids.lmms_eval"):
        backend.command_fields("internvl3_8b_hf", cfg.eval.task("mmmu_val"))


def test_baseline_is_looked_up_per_model(cfg):
    # The same task must gate each model against its own published number.
    cfg.model("internvl3_8b_hf").baselines["mmmu_val"] = 62.0
    q = ScoreRecord(harness="lmms_eval", model="qwen2_5_vl_7b", task="mmmu_val",
                    harness_task_id="x", metric="accuracy", score=58.0)
    i = ScoreRecord(harness="lmms_eval", model="internvl3_8b_hf", task="mmmu_val",
                    harness_task_id="x", metric="accuracy", score=58.0)
    gq, gi = check_gate(cfg, [q, i])
    assert gq.baseline == pytest.approx(58.6) and gq.passed is True
    assert gi.baseline == pytest.approx(62.0) and gi.passed is False


def test_model_without_a_baseline_is_not_gated_against_another_models(cfg):
    i = ScoreRecord(harness="lmms_eval", model="internvl3_8b_hf", task="mmmu_val",
                    harness_task_id="x", metric="accuracy", score=62.7)
    g = check_gate(cfg, [i])[0]
    assert g.baseline is None and g.passed is None
