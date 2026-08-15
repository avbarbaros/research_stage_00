"""Result schemas shared by stage 0 and stage 0.2.

Defined once, up front, so smoke/sweep output and benchmark output share a
format and ``report.py`` never needs rewriting when the harnesses land.

Everything is written as JSONL: each run appends, nothing rewrites, and a
crashed sweep still leaves every completed cell on disk.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class GenerationResult:
    """One generation, with the telemetry stage 0 exists to collect."""

    model: str
    request_id: str
    text: str = ""

    n_images: int = 0
    # Read back from the model's own tensors (image_grid_thw for Qwen, tile
    # count for InternVL) -- never estimated. budget.py's estimate is recorded
    # alongside so the two can be compared; a mismatch means our budgeting model
    # is wrong, which is exactly the bug we cannot carry into a benchmark run.
    visual_tokens: int = 0
    estimated_visual_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    peak_vram_gb: float = 0.0
    load_seconds: float = 0.0
    generate_seconds: float = 0.0

    # The budget rung this run was attempted at (sweep ladder value).
    rung: int | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    decode: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)

    oom: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.oom and self.error is None and bool(self.text.strip())

    @property
    def token_estimate_matches(self) -> bool | None:
        """Whether budget.py's estimate matched the measured token count.

        ``None`` when there is nothing to compare (a failed or un-estimated run).
        """
        if not self.ok or not self.estimated_visual_tokens:
            return None
        return self.visual_tokens == self.estimated_visual_tokens


@dataclass
class ScoreRecord:
    """One benchmark score, normalized across harnesses (stage 0.2).

    Both VLMEvalKit and lmms-eval reduce to this shape so ``report.py`` can join
    on ``(model, task, metric)`` and show the two side by side.
    """

    harness: str            # "vlmevalkit" | "lmms_eval"
    model: str
    task: str               # our canonical name, e.g. "mathvista_testmini"
    harness_task_id: str    # what that harness actually ran
    metric: str
    score: float
    n_samples: int = 0
    # Echoed back to prove both harnesses decoded identically. Comparing scores
    # produced under different decoding settings would be meaningless.
    decode: dict[str, Any] = field(default_factory=dict)
    raw_path: str = ""


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------
def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls: type[T]) -> list[T]:
    """Read records, skipping blank lines.

    Unknown fields are dropped rather than raising: an older results file should
    stay readable after the schema grows a field, since re-running a sweep costs
    GPU hours. Config is strict about unknown keys; results are not.
    """
    if not path.exists():
        return []
    valid = {f.name for f in dataclasses.fields(cls)}
    out: list[T] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        out.append(cls(**{k: v for k, v in data.items() if k in valid}))
    return out


def iter_results(results_dir: Path, pattern: str, cls: type[T]) -> Iterator[T]:
    for path in sorted(results_dir.glob(pattern)):
        yield from read_jsonl(path, cls)
