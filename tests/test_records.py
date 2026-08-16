"""Record schema and JSONL round-trip tests.

The schema is shared by stage 0 and stage 0.2, so it has to survive being
appended to across crashed runs and read back after the schema grows.
"""

from __future__ import annotations

from mmro.records import (
    GenerationResult,
    ScoreRecord,
    append_jsonl,
    iter_results,
    read_jsonl,
    write_jsonl,
)


def test_ok_requires_text_and_no_failure():
    assert GenerationResult(model="m", request_id="r", text="a caption").ok
    assert not GenerationResult(model="m", request_id="r", text="").ok
    assert not GenerationResult(model="m", request_id="r", text="x", oom=True).ok
    assert not GenerationResult(model="m", request_id="r", text="x", error="boom").ok
    # Whitespace is not a caption.
    assert not GenerationResult(model="m", request_id="r", text="   \n ").ok


def test_token_estimate_comparison():
    matched = GenerationResult(
        model="m", request_id="r", text="x", visual_tokens=100, estimated_visual_tokens=100
    )
    assert matched.token_estimate_matches is True

    mismatched = GenerationResult(
        model="m", request_id="r", text="x", visual_tokens=100, estimated_visual_tokens=90
    )
    assert mismatched.token_estimate_matches is False

    # Nothing to compare: no estimate, or the run failed.
    assert GenerationResult(model="m", request_id="r", text="x").token_estimate_matches is None
    assert GenerationResult(
        model="m", request_id="r", oom=True, estimated_visual_tokens=90
    ).token_estimate_matches is None


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "smoke_m.jsonl"
    records = [
        GenerationResult(model="m", request_id="a", text="one", n_images=1, visual_tokens=100),
        GenerationResult(model="m", request_id="b", text="two", n_images=3, visual_tokens=300),
    ]
    write_jsonl(path, records)
    assert read_jsonl(path, GenerationResult) == records


def test_append_accumulates_across_calls(tmp_path):
    # A crashed sweep must still leave every completed cell on disk.
    path = tmp_path / "sweep_m.jsonl"
    for i in range(3):
        append_jsonl(path, GenerationResult(model="m", request_id=str(i), text="x"))
    assert len(read_jsonl(path, GenerationResult)) == 3


def test_append_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "out.jsonl"
    append_jsonl(path, GenerationResult(model="m", request_id="a", text="x"))
    assert path.exists()


def test_reading_a_missing_file_yields_nothing(tmp_path):
    assert read_jsonl(tmp_path / "absent.jsonl", GenerationResult) == []


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [GenerationResult(model="m", request_id="a", text="x")])
    path.write_text(path.read_text() + "\n\n")
    assert len(read_jsonl(path, GenerationResult)) == 1


def test_unknown_fields_are_dropped_not_fatal(tmp_path):
    # Results outlive schema changes: re-running a sweep costs GPU hours, so an
    # older file stays readable after the schema grows a field.
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"model": "m", "request_id": "a", "text": "x", "some_future_field": 1}\n'
    )
    records = read_jsonl(path, GenerationResult)
    assert len(records) == 1 and records[0].model == "m"


def test_unicode_survives_the_round_trip(tmp_path):
    path = tmp_path / "out.jsonl"
    caption = "a café sign reading « ouvert » — 日本語も"
    write_jsonl(path, [GenerationResult(model="m", request_id="a", text=caption)])
    assert read_jsonl(path, GenerationResult)[0].text == caption


def test_newlines_in_captions_do_not_break_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [GenerationResult(model="m", request_id="a", text="line1\nline2")])
    assert path.read_text().count("\n") == 1
    assert read_jsonl(path, GenerationResult)[0].text == "line1\nline2"


def test_iter_results_spans_files_in_sorted_order(tmp_path):
    write_jsonl(tmp_path / "smoke_b.jsonl", [GenerationResult(model="b", request_id="1")])
    write_jsonl(tmp_path / "smoke_a.jsonl", [GenerationResult(model="a", request_id="1")])
    write_jsonl(tmp_path / "sweep_c.jsonl", [GenerationResult(model="c", request_id="1")])

    models = [r.model for r in iter_results(tmp_path, "smoke_*.jsonl", GenerationResult)]
    assert models == ["a", "b"]


def test_score_records_round_trip(tmp_path):
    path = tmp_path / "scores_vlmevalkit.jsonl"
    record = ScoreRecord(
        harness="vlmevalkit",
        model="qwen2_5_vl_7b",
        task="mathvista_testmini",
        harness_task_id="MathVista_MINI",
        metric="accuracy",
        score=0.581,
        n_samples=1000,
        decode={"do_sample": False, "max_new_tokens": 512},
    )
    write_jsonl(path, [record])
    restored = read_jsonl(path, ScoreRecord)[0]
    assert restored == record
    # The decode echo has to survive: it is what proves both harnesses ran the
    # same way.
    assert restored.decode["do_sample"] is False


def test_score_records_carry_the_visual_budget(tmp_path):
    # Both harnesses must be pinned to the sweep's measured budget; the record
    # has to prove it, the same way it proves decoding matched.
    path = tmp_path / "scores_vlmevalkit.jsonl"
    record = ScoreRecord(
        harness="vlmevalkit", model="qwen2_5_vl_7b", task="mmmu_val",
        harness_task_id="MMMU_DEV_VAL", metric="accuracy", score=0.52,
        decode={"do_sample": False, "max_new_tokens": 512},
        budget={"max_visual_tokens_per_image": 8192},
    )
    write_jsonl(path, [record])
    assert read_jsonl(path, ScoreRecord)[0].budget == {"max_visual_tokens_per_image": 8192}


def test_conditions_detect_a_mismatched_visual_budget():
    common = dict(model="m", task="t", harness_task_id="T", metric="acc", score=0.5,
                  decode={"do_sample": False})
    a = ScoreRecord(harness="vlmevalkit", budget={"max_tiles": 12}, **common)
    b = ScoreRecord(harness="lmms_eval", budget={"max_tiles": 6}, **common)
    # Same decoding, different resolution -> not comparable, and the score gap
    # would look like a harness difference.
    assert a.conditions[0] == b.conditions[0]
    assert a.conditions != b.conditions


def test_conditions_match_when_both_are_pinned():
    common = dict(model="m", task="t", harness_task_id="T", metric="acc", score=0.5,
                  decode={"do_sample": False}, budget={"max_tiles": 12})
    assert (ScoreRecord(harness="vlmevalkit", **common).conditions
            == ScoreRecord(harness="lmms_eval", **common).conditions)
