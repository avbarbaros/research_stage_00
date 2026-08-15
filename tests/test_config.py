"""Config layer tests.

The bias throughout: a mistake must fail loudly at startup. A typo that
silently reverts to a default would invalidate a whole sweep, and we would only
find out by disbelieving the results.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from mmro.config import (
    Config,
    ConfigError,
    DecodeConfig,
    EXAMPLE_CONFIG_PATH,
    load_config,
)


def write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def base_data():
    return yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text())


# --- the shipped example must itself be valid ------------------------------
def test_example_config_loads():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    assert set(cfg.models) == {"qwen2_5_vl_7b", "internvl3_8b_hf", "internvl3_8b_native"}
    assert cfg.enabled_models.keys() == {"qwen2_5_vl_7b", "internvl3_8b_hf"}
    assert cfg.model("internvl3_8b_native").enabled is False


def test_example_config_decode_is_greedy():
    # Cross-harness score comparison at stage 0.2 is only meaningful under
    # deterministic decoding.
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    assert cfg.decode.do_sample is False
    assert "temperature" not in cfg.decode.to_generate_kwargs()


# --- unknown keys are hard errors ------------------------------------------
def test_unknown_top_level_section_rejected(tmp_path, base_data):
    base_data["typo_section"] = {}
    with pytest.raises(ConfigError, match="unknown top-level section"):
        load_config(write(tmp_path, base_data))


def test_unknown_key_in_section_rejected(tmp_path, base_data):
    base_data["runtime"]["dtpye"] = "bfloat16"
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, base_data))


def test_unknown_budget_key_rejected(tmp_path, base_data):
    # The exact failure that motivates strictness: max_pixels is not the knob
    # this adapter reads, so it would be silently ignored.
    base_data["models"]["qwen2_5_vl_7b"]["budget"]["max_pixels"] = 999
    with pytest.raises(ConfigError, match="unknown key.*max_pixels"):
        load_config(write(tmp_path, base_data))


def test_budget_key_from_wrong_adapter_rejected(tmp_path, base_data):
    base_data["models"]["qwen2_5_vl_7b"]["budget"]["max_tiles"] = 6
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, base_data))


def test_unknown_adapter_rejected(tmp_path, base_data):
    base_data["models"]["qwen2_5_vl_7b"]["adapter"] = "not_an_adapter"
    with pytest.raises(ConfigError, match="unknown adapter"):
        load_config(write(tmp_path, base_data))


def test_unknown_harness_rejected(tmp_path, base_data):
    base_data["eval"]["harnesses"]["lmms_evel"] = {"enabled": True}
    with pytest.raises(ConfigError, match="unknown harness"):
        load_config(write(tmp_path, base_data))


# --- enum and type validation ----------------------------------------------
@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("runtime", "dtype", "float8", "invalid value"),
        ("runtime", "attn_implementation", "flash_attention_3", "invalid value"),
        ("paths", "cache_mode", "gdrive", "invalid value"),
        ("runtime", "seed", -1, "expected >= 0"),
        ("decode", "max_new_tokens", 0, "expected >= 1"),
        ("decode", "do_sample", "yes", "expected true or false"),
    ],
)
def test_invalid_values_rejected(tmp_path, base_data, section, key, value, match):
    base_data[section][key] = value
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, base_data))


def test_ladder_must_be_descending(tmp_path, base_data):
    # The sweep walks high -> low and stops at the first survivor. Out of order,
    # the reported "boundary" would not be the maximum.
    base_data["sweep"]["qwen_visual_token_ladder"] = [320, 1280, 640]
    with pytest.raises(ConfigError, match="strictly descending"):
        load_config(write(tmp_path, base_data))


def test_judge_backend_requires_model(tmp_path, base_data):
    base_data["eval"]["judge"]["backend"] = "api"
    with pytest.raises(ConfigError, match="eval.judge.model"):
        load_config(write(tmp_path, base_data))


def test_duplicate_task_names_rejected(tmp_path, base_data):
    base_data["eval"]["tasks"].append({"name": "mmmu_val", "vlmevalkit": "X", "lmms_eval": "y"})
    with pytest.raises(ConfigError, match="duplicate task name"):
        load_config(write(tmp_path, base_data))


def test_empty_models_rejected(tmp_path, base_data):
    base_data["models"] = {}
    with pytest.raises(ConfigError, match="at least one model"):
        load_config(write(tmp_path, base_data))


# --- override precedence ----------------------------------------------------
def test_env_override_applies(tmp_path, base_data):
    cfg = load_config(
        write(tmp_path, base_data), environ={"MMRO_PATHS__CACHE_MODE": "local"}
    )
    assert cfg.paths.cache_mode == "local"


def test_env_override_parses_types(tmp_path, base_data):
    cfg = load_config(
        write(tmp_path, base_data), environ={"MMRO_DECODE__MAX_NEW_TOKENS": "32"}
    )
    assert cfg.decode.max_new_tokens == 32


def test_cli_override_beats_env(tmp_path, base_data):
    cfg = load_config(
        write(tmp_path, base_data),
        cli_overrides=["paths.cache_mode=local"],
        environ={"MMRO_PATHS__CACHE_MODE": "drive"},
    )
    assert cfg.paths.cache_mode == "local"


def test_override_to_nonexistent_key_rejected(tmp_path, base_data):
    # A mistyped override that silently did nothing is worse than a crash: the
    # run would look configured and behave otherwise.
    with pytest.raises(ConfigError, match="no such config key"):
        load_config(write(tmp_path, base_data), cli_overrides=["paths.cach_mode=local"])


def test_override_still_validated(tmp_path, base_data):
    with pytest.raises(ConfigError, match="invalid value"):
        load_config(write(tmp_path, base_data), cli_overrides=["runtime.dtype=float8"])


def test_malformed_cli_override_rejected(tmp_path, base_data):
    with pytest.raises(ConfigError, match="dotted.key=value"):
        load_config(write(tmp_path, base_data), cli_overrides=["paths.cache_mode"])


# --- path resolution --------------------------------------------------------
def test_relative_paths_resolve_against_repo_root(tmp_path, base_data):
    cfg = load_config(write(tmp_path, base_data))
    assert cfg.paths.images_path.is_absolute()
    assert cfg.paths.images_path.name == "heldout"


def test_absolute_paths_pass_through(tmp_path, base_data):
    base_data["paths"]["results_dir"] = "/tmp/mmro_results"
    cfg = load_config(write(tmp_path, base_data))
    assert str(cfg.paths.results_path) == "/tmp/mmro_results"


def test_hf_home_follows_cache_mode(tmp_path, base_data):
    base_data["paths"]["cache_mode"] = "local"
    cfg = load_config(write(tmp_path, base_data))
    assert str(cfg.paths.hf_home_resolved) == base_data["paths"]["local_hf_home"]
    assert cfg.paths.caching_on_drive is False


@pytest.mark.parametrize(
    "key", ["hf_home", "local_hf_home", "stage_dir", "images_dir", "results_dir"]
)
def test_paths_have_no_defaults(tmp_path, base_data, key):
    # Paths are environment-specific. Defaulting one would mean a missing key
    # silently sends 33GB of weights somewhere the user did not choose.
    del base_data["paths"][key]
    with pytest.raises(ConfigError, match=f"missing required key '{key}'"):
        load_config(write(tmp_path, base_data))


def test_paths_section_cannot_be_omitted_entirely(tmp_path, base_data):
    del base_data["paths"]
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(write(tmp_path, base_data))


def test_stage_name_is_filesystem_safe():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    assert cfg.model("qwen2_5_vl_7b").stage_name == "Qwen__Qwen2.5-VL-7B-Instruct"
    assert "/" not in cfg.model("internvl3_8b_hf").stage_name


# --- decode sections --------------------------------------------------------
def test_decode_for_sections_override_only_max_new_tokens():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    assert cfg.decode_for("smoke").max_new_tokens == 256
    assert cfg.decode_for("sweep").max_new_tokens == 64
    assert cfg.decode_for("eval").max_new_tokens == 512
    # do_sample stays global so every harness provably decodes identically.
    assert all(cfg.decode_for(s).do_sample is False for s in ("smoke", "sweep", "eval"))


def test_decode_for_unknown_section_rejected():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    with pytest.raises(ConfigError, match="unknown section"):
        cfg.decode_for("training")


def test_generate_kwargs_omit_temperature_when_greedy():
    assert "temperature" not in DecodeConfig(do_sample=False).to_generate_kwargs()
    assert "temperature" in DecodeConfig(do_sample=True, temperature=0.7).to_generate_kwargs()


# --- lookups ----------------------------------------------------------------
def test_unknown_model_lookup_lists_options():
    cfg = load_config(EXAMPLE_CONFIG_PATH)
    with pytest.raises(ConfigError, match="no model named"):
        cfg.model("qwen3")


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_reports_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent("""
        paths:
          cache_mode: [unclosed
    """))
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)
