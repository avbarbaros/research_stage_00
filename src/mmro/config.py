"""Configuration loading, validation, and override precedence.

This is the only module that reads ``config.yaml``. Every other module receives
a typed :class:`Config` object. Nothing operational is hardcoded elsewhere in
``src/`` — if a researcher might reasonably change a value between runs, it
lives in the YAML and arrives here.

Precedence (highest first):

1. CLI flag           ``--set paths.cache_mode=local``
2. Environment var    ``MMRO_PATHS__CACHE_MODE=local``  (``__`` separates levels)
3. ``config.yaml``
4. Built-in defaults declared on the dataclasses below

Unknown keys and bad enum values are **hard errors**. A typo that silently
reverted to a default would invalidate a whole sweep or benchmark run without
anyone noticing, which is far more expensive than a failed startup.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config.example.yaml"

ENV_PREFIX = "MMRO_"
ENV_LEVEL_SEP = "__"


class ConfigError(Exception):
    """Raised for any malformed, unknown, or invalid configuration value."""


# ---------------------------------------------------------------------------
# Budget schemas
#
# Budget keys are adapter-specific, so they cannot live on a single dataclass.
# This table is a *schema*, not a tunable — researchers change budget values in
# YAML, never these key names — so it is legitimately source, not config.
# ---------------------------------------------------------------------------
BUDGET_KEYS: dict[str, set[str]] = {
    "qwen25vl": {"max_visual_tokens_per_image", "min_pixels", "safe_total_visual_tokens"},
    "internvl3_hf": {"max_tiles", "use_thumbnail", "safe_total_visual_tokens"},
    "internvl3_native": {"max_tiles", "use_thumbnail", "safe_total_visual_tokens"},
}

VALID_DTYPES = {"bfloat16", "float16", "float32"}
VALID_ATTN = {"sdpa", "flash_attention_2", "eager"}
VALID_CACHE_MODES = {"drive", "local"}
VALID_JUDGE_BACKENDS = {"none", "api", "local"}
KNOWN_HARNESSES = {"vlmevalkit", "lmms_eval"}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    """Where everything lives. Every path is stated in config.yaml.

    No defaults: these are environment-specific, so guessing one would mean a
    missing key silently sends 33GB of weights somewhere the user did not
    choose. A missing path is a startup error instead.
    """

    hf_home: str
    local_hf_home: str
    stage_dir: str
    images_dir: str
    results_dir: str
    cache_mode: str = "drive"

    def __post_init__(self) -> None:
        _check_enum("paths.cache_mode", self.cache_mode, VALID_CACHE_MODES)

    @property
    def hf_home_resolved(self) -> Path:
        """The HF cache root actually in effect, per ``cache_mode``."""
        raw = self.hf_home if self.cache_mode == "drive" else self.local_hf_home
        return _resolve(raw)

    @property
    def stage_path(self) -> Path:
        return _resolve(self.stage_dir)

    @property
    def images_path(self) -> Path:
        return _resolve(self.images_dir)

    @property
    def results_path(self) -> Path:
        return _resolve(self.results_dir)

    @property
    def caching_on_drive(self) -> bool:
        return self.cache_mode == "drive"


@dataclass
class RuntimeConfig:
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str = "cuda:0"
    seed: int = 0
    min_free_drive_gb: int = 50

    def __post_init__(self) -> None:
        _check_enum("runtime.dtype", self.dtype, VALID_DTYPES)
        _check_enum("runtime.attn_implementation", self.attn_implementation, VALID_ATTN)
        _check_int("runtime.seed", self.seed, minimum=0)
        _check_int("runtime.min_free_drive_gb", self.min_free_drive_gb, minimum=0)


@dataclass
class DecodeConfig:
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        _check_bool("decode.do_sample", self.do_sample)
        _check_int("decode.max_new_tokens", self.max_new_tokens, minimum=1)
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ConfigError(f"decode.temperature: expected a number, got {self.temperature!r}")

    def merged(self, **overrides: Any) -> "DecodeConfig":
        """A copy with non-None overrides applied. Used by sweep/eval sections."""
        values = dataclasses.asdict(self)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return DecodeConfig(**values)

    def to_generate_kwargs(self) -> dict[str, Any]:
        """``model.generate`` kwargs.

        ``temperature`` is omitted under greedy decoding: passing it alongside
        ``do_sample=False`` makes transformers emit a warning and is meaningless.
        """
        kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
        return kwargs


@dataclass
class ModelConfig:
    adapter: str
    repo_id: str
    enabled: bool = True
    trust_remote_code: bool = False
    budget: dict[str, Any] = field(default_factory=dict)

    # Populated by the loader; not a YAML key.
    name: str = ""

    def __post_init__(self) -> None:
        if self.adapter not in BUDGET_KEYS:
            raise ConfigError(
                f"models.{self.name}.adapter: unknown adapter {self.adapter!r}; "
                f"valid adapters: {sorted(BUDGET_KEYS)}"
            )
        _check_bool(f"models.{self.name}.enabled", self.enabled)
        _check_bool(f"models.{self.name}.trust_remote_code", self.trust_remote_code)

        allowed = BUDGET_KEYS[self.adapter]
        unknown = set(self.budget) - allowed
        if unknown:
            raise ConfigError(
                f"models.{self.name}.budget: unknown key(s) {sorted(unknown)} for "
                f"adapter {self.adapter!r}; valid keys: {sorted(allowed)}"
            )

    @property
    def stage_name(self) -> str:
        """Directory name under ``paths.stage_dir`` for this model's weights."""
        return self.repo_id.replace("/", "__")


@dataclass
class ImagesConfig:
    """Constraints on the held-out set.

    Stage 0 is a high-resolution test, so the resolution floor is a real
    parameter rather than a buried constant: below it, a sweep still runs but
    measures the images rather than the model.
    """

    min_long_side: int = 2048

    def __post_init__(self) -> None:
        _check_int("images.min_long_side", self.min_long_side, minimum=1)


@dataclass
class SmokeConfig:
    prompt: str = "Describe this image in detail."
    multi_image_prompt: str = "Compare these images. What differs between them?"
    image_counts: list[int] = field(default_factory=lambda: [1, 3])

    def __post_init__(self) -> None:
        _check_int_list("smoke.image_counts", self.image_counts, minimum=1)


@dataclass
class SweepConfig:
    image_counts: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    max_new_tokens: int | None = 64
    qwen_visual_token_ladder: list[int] = field(
        default_factory=lambda: [4096, 2560, 1280, 640, 320]
    )
    internvl_tile_ladder: list[int] = field(default_factory=lambda: [12, 8, 6, 4, 2, 1])

    def __post_init__(self) -> None:
        _check_int_list("sweep.image_counts", self.image_counts, minimum=1)
        _check_int_list("sweep.qwen_visual_token_ladder", self.qwen_visual_token_ladder, minimum=1)
        _check_int_list("sweep.internvl_tile_ladder", self.internvl_tile_ladder, minimum=1)
        if self.max_new_tokens is not None:
            _check_int("sweep.max_new_tokens", self.max_new_tokens, minimum=1)
        # The sweep walks each ladder high -> low and stops at the first rung
        # that survives. Out-of-order rungs would report a boundary that is not
        # actually the maximum, so the ordering is enforced rather than sorted
        # silently -- a mis-ordered ladder is a config mistake worth surfacing.
        _check_descending("sweep.qwen_visual_token_ladder", self.qwen_visual_token_ladder)
        _check_descending("sweep.internvl_tile_ladder", self.internvl_tile_ladder)


@dataclass
class HarnessConfig:
    enabled: bool = True
    venv: str = ""
    work_dir: str = ""
    repo: str = ""          # git URL, cloned by scripts/03_setup_harness.sh
    repo_dir: str = ""      # where the clone lives
    model_id: str = ""      # the harness's OWN name for the model
    command: str = ""       # command template; see `render_command`
    # A statement that imports what the harness needs to run our tasks. The
    # setup script executes it after installing and reports (or auto-installs)
    # whatever is missing, so dependency gaps surface in one pass instead of one
    # failed benchmark launch at a time.
    import_probe: str = ""
    # Interpreter to build the venv with. Empty means "whatever `python` is".
    # Set it when a harness dependency has no wheel for the default Python --
    # e.g. decord has none for 3.12+, so lmms-eval may need python3.11.
    python_bin: str = ""

    name: str = ""  # populated by the loader

    def __post_init__(self) -> None:
        _check_bool(f"eval.harnesses.{self.name}.enabled", self.enabled)

    @property
    def venv_path(self) -> Path:
        return _resolve(self.venv)

    @property
    def python(self) -> Path:
        """Interpreter inside this harness's isolated venv."""
        return self.venv_path / "bin" / "python"

    @property
    def work_path(self) -> Path:
        return _resolve(self.work_dir)

    @property
    def repo_path(self) -> Path:
        return _resolve(self.repo_dir)

    def render_command(self, **fields: Any) -> list[str]:
        """Fill the command template.

        The invocation lives in config because both harnesses change their CLI
        between releases. When one does, you edit a string here rather than
        patching a backend -- the same rule the rest of the project follows.
        """
        if not self.command:
            raise ConfigError(
                f"eval.harnesses.{self.name}.command: not set; nothing to run"
            )
        try:
            rendered = self.command.format(**fields)
        except KeyError as exc:
            raise ConfigError(
                f"eval.harnesses.{self.name}.command: unknown placeholder {exc}; "
                f"available: {sorted(fields)}"
            ) from exc
        return shlex.split(rendered)


@dataclass
class TaskRow:
    """One benchmark, with the task ID each harness uses for it.

    The two harnesses name the same benchmark differently; carrying both IDs on
    one row is what makes the cross-harness score comparison meaningful.
    """

    name: str
    vlmevalkit: str = ""
    lmms_eval: str = ""

    # The most images a single question in this benchmark can carry. It sets the
    # per-image visual budget: a mostly-single-image task must NOT inherit the
    # 7-image worst-case budget, or the model sees a needlessly downscaled image
    # and scores below its published number for a reason that is our doing.
    max_images: int = 1

    # Published score for the reference model, as a percentage. The gate compares
    # our result against this. Null means "no published number to check against".
    baseline: float | None = None

    # Harnesses whose scoring of THIS task is rule-based, i.e. never calls an LLM
    # judge. Only those runs may proceed with no judge configured; everything
    # else still fails fast without one. Empty by default: assuming a task needs
    # no judge when it does is the expensive mistake (hours of inference, then a
    # score tens of points low), so this must be opted into per harness.
    judge_free: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_int(f"eval.tasks[{self.name}].max_images", self.max_images, minimum=1)
        if not isinstance(self.judge_free, list):
            raise ConfigError(
                f"eval.tasks[{self.name}].judge_free: expected a list of harness "
                f"names, got {self.judge_free!r}"
            )
        for harness in self.judge_free:
            _check_enum(f"eval.tasks[{self.name}].judge_free", harness, KNOWN_HARNESSES)
        if self.baseline is not None:
            if not isinstance(self.baseline, (int, float)) or isinstance(self.baseline, bool):
                raise ConfigError(
                    f"eval.tasks[{self.name}].baseline: expected a number, "
                    f"got {self.baseline!r}"
                )

    def needs_judge(self, harness: str) -> bool:
        """Whether ``harness`` scores this task with an LLM judge."""
        return harness not in self.judge_free

    def id_for(self, harness: str) -> str:
        _check_enum("harness", harness, KNOWN_HARNESSES)
        task_id = getattr(self, harness)
        if not task_id:
            raise ConfigError(
                f"eval.tasks[{self.name!r}]: no task ID for harness {harness!r}; "
                f"add a '{harness}:' key to that row or disable the harness"
            )
        return task_id


@dataclass
class JudgeConfig:
    backend: str = "none"
    model: str | None = None
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        _check_enum("eval.judge.backend", self.backend, VALID_JUDGE_BACKENDS)
        if self.backend != "none" and not self.model:
            raise ConfigError(
                f"eval.judge.model: required when backend is {self.backend!r}"
            )
        if self.backend == "local":
            # Accepted by the enum so the message below can be specific, but not
            # wired to anything yet. Letting it through would pass `--judge` to
            # the harness with no endpoint, which silently falls back to OpenAI
            # or to rule-based matching and under-scores without an error.
            raise ConfigError(
                "eval.judge.backend: 'local' is not implemented yet. Use 'api', "
                "or 'none' for runs whose tasks are judge-free "
                "(see eval.tasks[].judge_free)"
            )
        if self.backend == "api" and not self.api_key_env:
            self.api_key_env = "OPENAI_API_KEY"


@dataclass
class EvalConfig:
    enabled: bool = False
    max_new_tokens: int | None = 512
    harnesses: dict[str, HarnessConfig] = field(default_factory=dict)
    tasks: list[TaskRow] = field(default_factory=list)
    limit: int | None = None
    # Absolute percentage points a result may differ from `baseline` and still
    # pass. Stage 0.2's gate.
    gate_tolerance: float = 2.0
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def __post_init__(self) -> None:
        _check_bool("eval.enabled", self.enabled)
        if self.gate_tolerance < 0:
            raise ConfigError("eval.gate_tolerance: must be >= 0")
        if self.max_new_tokens is not None:
            _check_int("eval.max_new_tokens", self.max_new_tokens, minimum=1)
        if self.limit is not None:
            _check_int("eval.limit", self.limit, minimum=1)

    @property
    def enabled_harnesses(self) -> dict[str, HarnessConfig]:
        return {n: h for n, h in self.harnesses.items() if h.enabled}

    def task(self, name: str) -> TaskRow:
        for row in self.tasks:
            if row.name == name:
                return row
        raise ConfigError(
            f"eval.tasks: no task named {name!r}; "
            f"configured tasks: {[r.name for r in self.tasks]}"
        )


@dataclass
class Config:
    paths: PathsConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    source_path: Path | None = None

    # -- model access -------------------------------------------------------
    @property
    def enabled_models(self) -> dict[str, ModelConfig]:
        return {n: m for n, m in self.models.items() if m.enabled}

    def model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise ConfigError(
                f"models: no model named {name!r}; configured: {sorted(self.models)}"
            )
        return self.models[name]

    # -- decoding -----------------------------------------------------------
    def decode_for(self, section: str) -> DecodeConfig:
        """Decoding settings for ``smoke`` | ``sweep`` | ``eval``.

        Sections override only ``max_new_tokens``; ``do_sample`` stays global so
        every harness and every model provably decodes the same way. That is a
        precondition for comparing scores across harnesses at stage 0.2.
        """
        if section == "smoke":
            return self.decode
        if section == "sweep":
            return self.decode.merged(max_new_tokens=self.sweep.max_new_tokens)
        if section == "eval":
            return self.decode.merged(max_new_tokens=self.eval.max_new_tokens)
        raise ConfigError(
            f"decode_for: unknown section {section!r}; expected smoke|sweep|eval"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_enum(where: str, value: Any, valid: set[str]) -> None:
    if value not in valid:
        raise ConfigError(f"{where}: invalid value {value!r}; expected one of {sorted(valid)}")


def _check_bool(where: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false, got {value!r}")


def _check_int(where: str, value: Any, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: expected >= {minimum}, got {value}")


def _check_int_list(where: str, value: Any, minimum: int | None = None) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list of integers, got {value!r}")
    for i, item in enumerate(value):
        _check_int(f"{where}[{i}]", item, minimum=minimum)


def _check_descending(where: str, value: list[int]) -> None:
    if any(b >= a for a, b in zip(value, value[1:])):
        raise ConfigError(
            f"{where}: must be strictly descending (the sweep walks it high to low "
            f"and stops at the first rung that survives), got {value}"
        )


def _resolve(raw: str) -> Path:
    """Absolute paths pass through; relative ones resolve against the repo root."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _build(cls: type, data: Any, where: str, **extra: Any):
    """Instantiate a dataclass from a mapping, rejecting unknown/missing keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    settable = set(fields) - set(extra)
    unknown = set(data) - settable
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(settable)}"
        )

    kwargs: dict[str, Any] = dict(extra)
    for name, f in fields.items():
        if name in data:
            kwargs[name] = data[name]
        elif name not in kwargs and f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{where}: missing required key {name!r}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def _walk_env_overrides(environ: dict[str, str]) -> Iterator[tuple[list[str], Any]]:
    for key, raw in sorted(environ.items()):
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split(ENV_LEVEL_SEP) if p]
        if not path:
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        yield path, value


def _parse_cli_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ConfigError(
            f"--set {item!r}: expected dotted.key=value, e.g. paths.cache_mode=local"
        )
    key, _, raw = item.partition("=")
    path = [p for p in key.strip().split(".") if p]
    if not path:
        raise ConfigError(f"--set {item!r}: empty key")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return path, value


def _apply_override(data: dict[str, Any], path: list[str], value: Any, origin: str) -> None:
    """Set ``path`` within ``data``.

    Overrides may only reach keys that already exist in the merged YAML. Creating
    a new key here would defeat the unknown-key check -- a mistyped env var would
    look like it worked while changing nothing the code reads.
    """
    node: Any = data
    for i, part in enumerate(path[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(
                f"{origin}: no such config section {'.'.join(path[:i + 1])!r}"
            )
        node = node[part]
    leaf = path[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise ConfigError(f"{origin}: no such config key {'.'.join(path)!r}")
    node[leaf] = value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_config(
    path: str | Path | None = None,
    cli_overrides: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load, override, and validate the configuration.

    Falls back to ``config.example.yaml`` when ``config.yaml`` is absent, so the
    repo is runnable on checkout (and CPU tests need no setup step).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if path is not None:
            raise ConfigError(f"config file not found: {config_path}")
        config_path = EXAMPLE_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a top-level mapping")

    for env_path, value in _walk_env_overrides(
        os.environ.copy() if environ is None else environ
    ):
        _apply_override(raw, env_path, value, origin=f"{ENV_PREFIX}* env override")
    for item in cli_overrides or []:
        cli_path, value = _parse_cli_override(item)
        _apply_override(raw, cli_path, value, origin=f"--set {item}")

    top_level = {f.name for f in dataclasses.fields(Config)} - {"source_path"}
    unknown = set(raw) - top_level
    if unknown:
        raise ConfigError(
            f"{config_path}: unknown top-level section(s) {sorted(unknown)}; "
            f"valid sections: {sorted(top_level)}"
        )

    models: dict[str, ModelConfig] = {}
    for name, spec in (raw.get("models") or {}).items():
        models[name] = _build(ModelConfig, spec, f"models.{name}", name=name)
    if not models:
        raise ConfigError(f"{config_path}: models: at least one model must be configured")

    return Config(
        paths=_build(PathsConfig, raw.get("paths", {}), "paths"),
        runtime=_build(RuntimeConfig, raw.get("runtime", {}), "runtime"),
        decode=_build(DecodeConfig, raw.get("decode", {}), "decode"),
        models=models,
        images=_build(ImagesConfig, raw.get("images", {}), "images"),
        smoke=_build(SmokeConfig, raw.get("smoke", {}), "smoke"),
        sweep=_build(SweepConfig, raw.get("sweep", {}), "sweep"),
        eval=_build_eval(raw.get("eval", {})),
        source_path=config_path,
    )


def _build_eval(raw: Any) -> EvalConfig:
    if not isinstance(raw, dict):
        raise ConfigError("eval: expected a mapping")

    harnesses: dict[str, HarnessConfig] = {}
    for name, spec in (raw.get("harnesses") or {}).items():
        if name not in KNOWN_HARNESSES:
            raise ConfigError(
                f"eval.harnesses.{name}: unknown harness; "
                f"valid harnesses: {sorted(KNOWN_HARNESSES)}"
            )
        harnesses[name] = _build(HarnessConfig, spec, f"eval.harnesses.{name}", name=name)

    tasks: list[TaskRow] = []
    for i, spec in enumerate(raw.get("tasks") or []):
        tasks.append(_build(TaskRow, spec, f"eval.tasks[{i}]"))
    duplicates = {t.name for t in tasks if [x.name for x in tasks].count(t.name) > 1}
    if duplicates:
        raise ConfigError(f"eval.tasks: duplicate task name(s) {sorted(duplicates)}")

    rest = {k: v for k, v in raw.items() if k not in {"harnesses", "tasks", "judge"}}
    return _build(
        EvalConfig,
        rest,
        "eval",
        harnesses=harnesses,
        tasks=tasks,
        judge=_build(JudgeConfig, raw.get("judge", {}), "eval.judge"),
    )


# ---------------------------------------------------------------------------
# CLI: python -m mmro.config --validate
# ---------------------------------------------------------------------------
def _summarize(cfg: Config) -> str:
    lines = [
        f"config source     : {cfg.source_path}",
        f"cache mode        : {cfg.paths.cache_mode}",
        f"HF_HOME (resolved): {cfg.paths.hf_home_resolved}",
        f"stage dir         : {cfg.paths.stage_path}",
        f"images dir        : {cfg.paths.images_path}",
        f"results dir       : {cfg.paths.results_path}",
        f"dtype / attn      : {cfg.runtime.dtype} / {cfg.runtime.attn_implementation}",
        f"decode            : {cfg.decode.to_generate_kwargs()}",
        "",
        "models:",
    ]
    for name, m in cfg.models.items():
        mark = "on " if m.enabled else "off"
        lines.append(f"  [{mark}] {name:<20} {m.repo_id:<32} adapter={m.adapter} budget={m.budget}")
    lines += ["", f"eval (stage 0.2)  : enabled={cfg.eval.enabled}"]
    for name, h in cfg.eval.harnesses.items():
        mark = "on " if h.enabled else "off"
        lines.append(f"  [{mark}] {name:<12} venv={h.venv}")
    for row in cfg.eval.tasks:
        lines.append(
            f"  task {row.name:<20} vlmevalkit={row.vlmevalkit!r} lmms_eval={row.lmms_eval!r}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and print the resolved config.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--validate", action="store_true", help="validate and print a summary")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set paths.cache_mode=local",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, cli_overrides=args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    print(_summarize(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
