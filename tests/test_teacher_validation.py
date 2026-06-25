from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.teacher_validation import validate_teacher_output_file, validate_teacher_row


def _valid_answer() -> str:
    return '{"elements":[{"text":"Home","type":"tab","focused":true}]}'


def _mismatching_decode(_tokens: list[int]) -> str:
    return '{"elements":[{"text":"Search","type":"tab","focused":true}]}'


def _failing_decode(_tokens: list[int]) -> str:
    raise ValueError("teacher_answer is not valid JSON")


def _logits(length: int) -> dict:
    return {
        "indices": [[[0] for _ in range(length)]],
        "values": [[[1.0] for _ in range(length)]],
        "shape": [1, length, 8],
        "vocab_size": 8,
    }


def _teacher_logits_metadata(tokens: list[int]) -> dict:
    return {
        "teacher_logits_format": "adaptive_topk",
        "teacher_logits_vocab_size": 8,
        "teacher_logits_aligned_to_answer": True,
        "teacher_logits_token_identity_match": True,
        "teacher_logits_answer_token_ids": tokens,
    }


def _row(*, tokens: list[int] | None = None) -> dict:
    actual_tokens = tokens or [101, 202, 303]
    return {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": actual_tokens,
        "teacher_logits": _logits(len(actual_tokens)),
        **_teacher_logits_metadata(actual_tokens),
    }


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_teacher_forced_canonical_row_passes_with_decode_mismatch():
    valid, reason = validate_teacher_row(
        _row(),
        require_teacher_logits=True,
        decode_tokens=_mismatching_decode,
    )

    assert valid is True
    assert reason is None


def test_teacher_forced_canonical_row_passes_with_decode_error_when_logits_validate():
    valid, reason = validate_teacher_row(
        _row(),
        require_teacher_logits=True,
        decode_tokens=_failing_decode,
    )

    assert valid is True
    assert reason is None


def test_row_fails_when_teacher_logits_answer_token_ids_do_not_match_teacher_tokens():
    row = _row()
    row["teacher_logits_answer_token_ids"] = [999, 202, 303]

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_mismatching_decode,
    )

    assert valid is False
    assert "token identity mismatch" in str(reason)


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (
            lambda row: row.pop("teacher_logits_token_identity_match"),
            "teacher_logits_token_identity_match is not true",
        ),
        (
            lambda row: row.__setitem__("teacher_logits_token_identity_match", False),
            "teacher_logits_token_identity_match is not true",
        ),
    ],
)
def test_row_fails_when_token_identity_match_flag_missing_or_false(mutator, expected_reason):
    row = _row()
    mutator(row)

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_mismatching_decode,
    )

    assert valid is False
    assert expected_reason in str(reason)


def test_row_fails_when_logits_shape_length_does_not_match_teacher_tokens():
    row = _row()
    row["teacher_logits"]["shape"] = [1, 2, 8]

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_mismatching_decode,
    )

    assert valid is False
    assert "length mismatch with teacher_tokens" in str(reason)


def test_validate_teacher_output_file_reports_token_identity_match_rows(tmp_path: Path):
    valid_row = _row(tokens=[1, 2, 3])
    invalid_row = _row(tokens=[4, 5, 6])
    invalid_row["teacher_logits_answer_token_ids"] = [4, 5, 999]

    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [valid_row, invalid_row]),
        decode_tokens=_mismatching_decode,
        require_teacher_logits=True,
    )

    assert summary["token_identity_match_rows"] == 1
    assert summary["token_identity_mismatch_rows"] == 1
    assert summary["answer_token_match_rows"] == 0
    assert summary["answer_token_mismatch_rows"] == 1
    assert summary["invalid_rows"] == 1


def test_validate_teacher_output_file_keeps_decode_error_diagnostic_non_fatal_when_logits_validate(tmp_path: Path):
    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [_row(tokens=[11, 22, 33])]),
        decode_tokens=_failing_decode,
        require_teacher_logits=True,
    )

    assert summary["valid_json_rows"] == 1
    assert summary["schema_valid_rows"] == 1
    assert summary["rows_with_teacher_logits"] == 1
    assert summary["valid_teacher_logits_rows"] == 1
    assert summary["logits_length_match_rows"] == 1
    assert summary["token_identity_match_rows"] == 1
    assert summary["answer_token_mismatch_rows"] == 0
    assert summary["invalid_rows"] == 0


def test_validate_teacher_output_file_fails_on_decode_error_without_required_logits(tmp_path: Path):
    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [_row(tokens=[11, 22, 33])]),
        decode_tokens=_failing_decode,
        require_teacher_logits=False,
    )

    assert summary["valid_json_rows"] == 1
    assert summary["schema_valid_rows"] == 1
    assert summary["invalid_rows"] == 1
    assert summary["bad_rows"][0]["reason"] == "teacher_answer is not valid JSON"


def test_validate_teacher_output_file_uses_configurable_logits_field(tmp_path: Path):
    row = _row(tokens=[7, 8, 9])
    row["cached_teacher_logits"] = row.pop("teacher_logits")
    row["cached_teacher_logits_format"] = row.pop("teacher_logits_format")
    row["cached_teacher_logits_vocab_size"] = row.pop("teacher_logits_vocab_size")
    row["cached_teacher_logits_aligned_to_answer"] = row.pop("teacher_logits_aligned_to_answer")
    row["cached_teacher_logits_token_identity_match"] = row.pop("teacher_logits_token_identity_match")
    row["cached_teacher_logits_answer_token_ids"] = row.pop("teacher_logits_answer_token_ids")

    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [row]),
        decode_tokens=_failing_decode,
        require_teacher_logits=True,
        logits_field="cached_teacher_logits",
    )

    assert summary["rows_with_teacher_logits"] == 1
    assert summary["valid_teacher_logits_rows"] == 1
    assert summary["token_identity_match_rows"] == 1
    assert summary["invalid_rows"] == 0
