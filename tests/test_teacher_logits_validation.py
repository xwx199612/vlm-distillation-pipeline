from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vlm_distill.config_schema import load_config, resolve_label_path
import vlm_distill.cli as cli
from vlm_distill.teacher_validation import validate_teacher_output_file, validate_teacher_row


def _answer() -> str:
    return '{"elements":[{"text":"Home","type":"tab","focused":true}]}'


def _valid_answer() -> str:
    return '{"elements":[{"text":"Home","type":"tab","focused":true}]}'


def _matching_decode(_tokens: list[int]) -> str:
    return _valid_answer()


def _logits(length: int) -> dict:
    return {
        "indices": [[[0] for _ in range(length)]],
        "values": [[[1.0] for _ in range(length)]],
        "shape": [1, length, 8],
        "vocab_size": 8,
    }


def _write_row(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_valid_unified_teacher_row_passes(tmp_path: Path):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
        "teacher_logits": _logits(3),
        "teacher_logits_format": "adaptive_topk",
        "teacher_logits_vocab_size": 8,
        "teacher_logits_aligned_to_answer": True,
    }

    summary = validate_teacher_output_file(
        _write_row(tmp_path, row),
        decode_tokens=_matching_decode,
        require_teacher_logits=True,
    )

    assert summary["valid_json_rows"] == 1
    assert summary["schema_valid_rows"] == 1
    assert summary["answer_token_match_rows"] == 1
    assert summary["rows_with_teacher_logits"] == 1
    assert summary["valid_teacher_logits_rows"] == 1
    assert summary["logits_length_match_rows"] == 1
    assert summary["invalid_rows"] == 0


def test_string_list_row_fails(tmp_path: Path):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": json.dumps({"elements": ["Search", "Home"]}),
        "teacher_tokens": [1, 2, 3],
    }

    valid, reason = validate_teacher_row(row, decode_tokens=_matching_decode)

    assert valid is False
    assert "string-list item" in str(reason)

    summary = validate_teacher_output_file(_write_row(tmp_path, row), decode_tokens=_matching_decode)
    assert summary["string_list_rows"] == 1
    assert summary["invalid_rows"] == 1


def test_token_mismatch_fails(tmp_path: Path):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
    }

    valid, reason = validate_teacher_row(
        row,
        decode_tokens=lambda _tokens: '{"elements":[{"text":"Search","type":"tab","focused":true}]}',
    )

    assert valid is False
    assert "do not match" in str(reason)


def test_missing_logits_fails_when_required():
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
    }

    valid, reason = validate_teacher_row(row, require_teacher_logits=True, decode_tokens=_matching_decode)

    assert valid is False
    assert "teacher_logits" in str(reason)


def test_missing_logits_passes_when_not_required():
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
    }

    valid, reason = validate_teacher_row(row, require_teacher_logits=False, decode_tokens=_matching_decode)

    assert valid is True
    assert reason is None


def test_full_sequence_logits_fail(tmp_path: Path):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
        "teacher_logits": _logits(4),
        "teacher_logits_format": "adaptive_topk",
        "teacher_logits_vocab_size": 8,
        "teacher_logits_aligned_to_answer": True,
    }

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_matching_decode,
    )

    assert valid is False
    assert "length mismatch" in str(reason)

    summary = validate_teacher_output_file(
        _write_row(tmp_path, row),
        decode_tokens=_matching_decode,
        require_teacher_logits=True,
    )
    assert summary["full_sequence_logits_rows"] == 1
    assert summary["logits_length_mismatch_rows"] == 1


@pytest.mark.parametrize(
    "mutator, expected_reason",
    [
        (lambda payload: payload.pop("indices"), "missing indices"),
        (lambda payload: payload.pop("values"), "missing indices"),
        (
            lambda payload: payload.update({"values": [[[1.0, 2.0]]]}),
            "indices/values sequence length mismatch",
        ),
        (
            lambda payload: payload["indices"][0][0].__setitem__(0, 9),
            "token index out of range",
        ),
    ],
)
def test_invalid_logits_payload_fails(mutator, expected_reason):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
        "teacher_logits": _logits(3),
        "teacher_logits_format": "adaptive_topk",
        "teacher_logits_vocab_size": 8,
        "teacher_logits_aligned_to_answer": True,
    }
    mutator(row["teacher_logits"])

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_matching_decode,
    )

    assert valid is False
    assert expected_reason in str(reason)


@pytest.mark.parametrize(
    "field, value, expected_reason",
    [
        ("teacher_logits_format", None, "teacher_logits_format is missing"),
        ("teacher_logits_vocab_size", None, "teacher_logits_vocab_size is missing"),
        ("teacher_logits_aligned_to_answer", False, "teacher_logits_aligned_to_answer is not true"),
    ],
)
def test_missing_logits_metadata_fails(field, value, expected_reason):
    row = {
        "id": "sample-1",
        "image": "screen.png",
        "query": "List the visible UI elements.",
        "teacher_answer": _valid_answer(),
        "teacher_tokens": [1, 2, 3],
        "teacher_logits": _logits(3),
        "teacher_logits_format": "adaptive_topk",
        "teacher_logits_vocab_size": 8,
        "teacher_logits_aligned_to_answer": True,
    }
    if value is None:
        row.pop(field)
    else:
        row[field] = value

    valid, reason = validate_teacher_row(
        row,
        require_teacher_logits=True,
        decode_tokens=_matching_decode,
    )

    assert valid is False
    assert expected_reason in str(reason)


def test_validate_teacher_cli_works(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config_path = Path("configs/parsing_switch_kd.yaml")
    monkeypatch.setenv("VLM_DISTILL_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli.teacher_validation, "build_teacher_token_decoder", lambda _config: _matching_decode)

    config = load_config(config_path)
    label_path = resolve_label_path(config.data)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image": "screen.png",
                "query": "List the visible UI elements.",
                "teacher_answer": _valid_answer(),
                "teacher_tokens": [1, 2, 3],
                "teacher_logits": _logits(3),
                "teacher_logits_format": "adaptive_topk",
                "teacher_logits_vocab_size": 8,
                "teacher_logits_aligned_to_answer": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["vlm-distill", "validate-teacher", "--config", str(config_path)],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "OK validated teacher output path=" in output
    assert "invalid_rows=0" in output


def test_deprecated_validate_labels_alias_is_rejected(monkeypatch, tmp_path: Path):
    config_path = Path("configs/parsing_switch_kd.yaml")
    monkeypatch.setenv("VLM_DISTILL_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["vlm-distill", "validate-labels", "--config", str(config_path)],
    )

    with pytest.raises(SystemExit, match="validate-labels is deprecated. Use validate-teacher."):
        cli.main()
