from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .data_manifest import read_jsonl
from .stage_teacher_precompute import (
    _canonicalize_teacher_answer,
    _parse_json_object,
    _strip_special_tokens,
)
from .token_alignment import build_token_mismatch_details, coerce_token_ids


DecodeTokens = Callable[[list[int]], str]

ALLOWED_ELEMENT_TYPES = {
    "tab",
    "button",
    "app_icon",
    "app_tile",
    "menu_item",
    "tile",
    "toggle",
    "input",
    "icon",
    "link",
    "other",
    "unknown",
}


@dataclass(frozen=True)
class _TeacherRowReport:
    valid: bool
    reason: str | None
    decode_error: str | None
    valid_json: bool
    schema_valid: bool
    string_list_row: bool
    answer_token_match: bool
    token_identity_match: bool
    teacher_logits_present: bool
    valid_teacher_logits: bool
    logits_length_match: bool
    full_sequence_logits: bool
    vocab_mismatch: bool


def validate_teacher_output_file(
    path: Path,
    *,
    max_samples: int | None = None,
    decode_tokens: DecodeTokens | None = None,
    require_teacher_logits: bool = False,
    bad_limit: int = 5,
    logits_field: str = "teacher_logits",
) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    summary: dict[str, Any] = {
        "path": str(path),
        "total_rows": len(rows),
        "valid_json_rows": 0,
        "schema_valid_rows": 0,
        "string_list_rows": 0,
        "answer_token_match_rows": 0,
        "answer_token_mismatch_rows": 0,
        "token_identity_match_rows": 0,
        "token_identity_mismatch_rows": 0,
        "rows_with_teacher_logits": 0,
        "valid_teacher_logits_rows": 0,
        "logits_length_match_rows": 0,
        "logits_length_mismatch_rows": 0,
        "full_sequence_logits_rows": 0,
        "vocab_mismatch_rows": 0,
        "invalid_rows": 0,
        "bad_rows": [],
    }

    bad_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        report = _analyze_teacher_row(
            row,
            decode_tokens=decode_tokens,
            require_teacher_logits=require_teacher_logits,
            logits_field=logits_field,
        )

        if report.valid_json:
            summary["valid_json_rows"] += 1
        if report.schema_valid:
            summary["schema_valid_rows"] += 1
        if report.string_list_row:
            summary["string_list_rows"] += 1
        if report.answer_token_match:
            summary["answer_token_match_rows"] += 1
        elif _should_count_token_mismatch(
            report,
            decode_tokens=decode_tokens,
            require_teacher_logits=require_teacher_logits,
        ):
            summary["answer_token_mismatch_rows"] += 1
        if report.token_identity_match:
            summary["token_identity_match_rows"] += 1
        elif _should_count_token_identity_mismatch(report, require_teacher_logits=require_teacher_logits):
            summary["token_identity_mismatch_rows"] += 1
        if report.teacher_logits_present:
            summary["rows_with_teacher_logits"] += 1
        if report.valid_teacher_logits:
            summary["valid_teacher_logits_rows"] += 1
        if report.logits_length_match:
            summary["logits_length_match_rows"] += 1
        elif report.teacher_logits_present or require_teacher_logits:
            summary["logits_length_mismatch_rows"] += 1
        if report.full_sequence_logits:
            summary["full_sequence_logits_rows"] += 1
        if report.vocab_mismatch:
            summary["vocab_mismatch_rows"] += 1
        if not report.valid:
            summary["invalid_rows"] += 1
            _add_bad_row(bad_rows, row_id, report.reason or "invalid teacher row", bad_limit)

    summary["bad_rows"] = bad_rows
    return summary


def validate_teacher_row(
    row: dict[str, Any],
    *,
    require_teacher_logits: bool = False,
    decode_tokens: DecodeTokens | None = None,
    logits_field: str = "teacher_logits",
) -> tuple[bool, str | None]:
    report = _analyze_teacher_row(
        row,
        decode_tokens=decode_tokens,
        require_teacher_logits=require_teacher_logits,
        logits_field=logits_field,
    )
    return report.valid, report.reason


def build_teacher_token_decoder(config) -> DecodeTokens | None:
    try:
        from transformers import AutoProcessor

        from .model_loading import resolve_model_path
    except ImportError:
        return None

    try:
        processor = AutoProcessor.from_pretrained(
            resolve_model_path(config.teacher.model_name),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    decoder = tokenizer if tokenizer is not None else processor

    def decode(token_ids: list[int]) -> str:
        return decoder.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return decode


def _analyze_teacher_row(
    row: dict[str, Any],
    *,
    decode_tokens: DecodeTokens | None,
    require_teacher_logits: bool,
    logits_field: str = "teacher_logits",
) -> _TeacherRowReport:
    row_id = row.get("id")
    if row_id is None or str(row_id).strip() == "":
        return _report(False, "id is missing")
    if not _has_text(row.get("image")):
        return _report(False, "image is missing")
    if not _has_text(row.get("query")):
        return _report(False, "query is missing")
    answer = row.get("teacher_answer")
    if not isinstance(answer, str) or not answer.strip():
        return _report(False, "teacher_answer is missing or not a string")
    tokens = _extract_teacher_tokens(row)
    if not tokens:
        return _report(False, "teacher_tokens missing or empty")

    raw_answer = str(answer)
    parsed = _parse_json_object(raw_answer)
    if parsed is None:
        return _report(False, "teacher_answer is not valid JSON")
    valid_json = True

    schema_valid, schema_reason, string_list_row = _validate_teacher_answer_schema(parsed)
    if not schema_valid:
        return _report(
            False,
            schema_reason or "teacher_answer schema is invalid",
            valid_json=valid_json,
            schema_valid=False,
            string_list_row=string_list_row,
        )

    answer_token_match = False
    logits_present = row.get(logits_field) is not None
    logits_report = _validate_teacher_logits_payload(
        row,
        answer_len=len(tokens),
        logits_field=logits_field,
        require_teacher_logits=require_teacher_logits,
    )
    if not logits_report.valid:
        return _report(
            False,
            logits_report.reason,
            valid_json=valid_json,
            schema_valid=True,
            string_list_row=string_list_row,
            answer_token_match=answer_token_match,
            token_identity_match=logits_report.token_identity_match,
            teacher_logits_present=logits_present,
            valid_teacher_logits=logits_report.valid_teacher_logits,
            logits_length_match=logits_report.logits_length_match,
            full_sequence_logits=logits_report.full_sequence_logits,
            vocab_mismatch=logits_report.vocab_mismatch,
        )

    decode_error: str | None = None
    if decode_tokens is not None:
        try:
            decoded = _strip_special_tokens(decode_tokens(tokens))
            canonical_answer = _canonicalize_teacher_answer(_strip_special_tokens(raw_answer))
            canonical_decoded = _canonicalize_teacher_answer(decoded)
            answer_token_match = canonical_answer == canonical_decoded
            if not answer_token_match and not require_teacher_logits:
                return _report(
                    False,
                    "decoded teacher_tokens do not match teacher_answer",
                    valid_json=valid_json,
                    schema_valid=True,
                    string_list_row=string_list_row,
                    answer_token_match=False,
                    token_identity_match=logits_report.token_identity_match,
                    teacher_logits_present=logits_present,
                    valid_teacher_logits=logits_report.valid_teacher_logits,
                    logits_length_match=logits_report.logits_length_match,
                    full_sequence_logits=logits_report.full_sequence_logits,
                    vocab_mismatch=logits_report.vocab_mismatch,
                )
        except Exception as exc:  # noqa: BLE001
            decode_error = str(exc)
            if not require_teacher_logits:
                return _report(
                    False,
                    decode_error,
                    decode_error=decode_error,
                    valid_json=valid_json,
                    schema_valid=True,
                    string_list_row=string_list_row,
                    answer_token_match=False,
                    token_identity_match=logits_report.token_identity_match,
                    teacher_logits_present=logits_present,
                    valid_teacher_logits=logits_report.valid_teacher_logits,
                    logits_length_match=logits_report.logits_length_match,
                    full_sequence_logits=logits_report.full_sequence_logits,
                    vocab_mismatch=logits_report.vocab_mismatch,
                )

    return _report(
        True,
        None,
        decode_error=decode_error,
        valid_json=valid_json,
        schema_valid=True,
        string_list_row=string_list_row,
        answer_token_match=answer_token_match,
        token_identity_match=logits_report.token_identity_match,
        teacher_logits_present=logits_present,
        valid_teacher_logits=logits_report.valid_teacher_logits,
        logits_length_match=logits_report.logits_length_match,
        full_sequence_logits=logits_report.full_sequence_logits,
        vocab_mismatch=logits_report.vocab_mismatch,
    )


@dataclass(frozen=True)
class _LogitsReport:
    valid: bool
    reason: str | None
    token_identity_match: bool
    valid_teacher_logits: bool
    logits_length_match: bool
    full_sequence_logits: bool
    vocab_mismatch: bool


def _validate_teacher_logits_payload(
    row: dict[str, Any],
    *,
    answer_len: int,
    logits_field: str,
    require_teacher_logits: bool,
) -> _LogitsReport:
    payload = row.get(logits_field)
    if payload is None:
        if require_teacher_logits:
            return _logits_report(False, f"{logits_field} missing", False, False, False, False, False)
        return _logits_report(True, None, False, False, False, False, False)
    if not isinstance(payload, dict):
        return _logits_report(False, f"{logits_field} missing or not a dict", False, False, False, False, False)

    metadata_prefix = logits_field
    if not _has_text(row.get(f"{metadata_prefix}_format")):
        return _logits_report(False, f"{metadata_prefix}_format is missing", False, False, False, False, False)
    if row.get(f"{metadata_prefix}_aligned_to_answer") is not True:
        return _logits_report(
            False,
            f"{metadata_prefix}_aligned_to_answer is not true",
            False,
            False,
            False,
            False,
            False,
        )
    if row.get(f"{metadata_prefix}_token_identity_match") is not True:
        return _logits_report(
            False,
            f"{metadata_prefix}_token_identity_match is not true",
            False,
            False,
            False,
            False,
            False,
        )
    if row.get(f"{metadata_prefix}_vocab_size") is None:
        return _logits_report(False, f"{metadata_prefix}_vocab_size is missing", False, False, False, False, False)
    answer_token_ids = row.get(f"{metadata_prefix}_answer_token_ids")
    if answer_token_ids is None:
        return _logits_report(False, f"{metadata_prefix}_answer_token_ids is missing", False, False, False, False, False)
    answer_token_ids = coerce_token_ids(answer_token_ids)
    teacher_tokens = _extract_teacher_tokens(row)
    if answer_token_ids != teacher_tokens:
        return _logits_report(
            False,
            f"{metadata_prefix} token identity mismatch: "
            f"{build_token_mismatch_details(expected=teacher_tokens, actual=answer_token_ids, actual_field_name='actual_answer_token_id')}",
            False,
            False,
            False,
            False,
            False,
        )
    token_identity_match = True

    required_keys = ("indices", "values", "vocab_size", "shape")
    if not all(key in payload for key in required_keys):
        return _logits_report(
            False,
            f"{logits_field} missing indices, values, vocab_size, or shape",
            token_identity_match,
            False,
            False,
            False,
            False,
        )

    if not isinstance(payload["shape"], list) or len(payload["shape"]) < 2:
        return _logits_report(False, f"{logits_field}.shape invalid", token_identity_match, False, False, False, False)
    shape = payload["shape"]
    if len(shape) < 3:
        return _logits_report(False, f"{logits_field}.shape invalid", token_identity_match, False, False, False, False)
    try:
        shape_seq_len = int(shape[1])
    except (TypeError, ValueError):
        return _logits_report(False, f"{logits_field}.shape is not numeric", token_identity_match, False, False, False, False)
    if shape_seq_len != answer_len:
        reason = f"{logits_field} length mismatch with teacher_tokens"
        return _logits_report(False, reason, token_identity_match, False, False, shape_seq_len > answer_len, False)

    if shape_seq_len > answer_len:
        return _logits_report(False, f"{logits_field} full-sequence logits are not allowed", token_identity_match, False, False, True, False)

    try:
        payload_vocab_size = int(payload["vocab_size"])
        row_vocab_size = int(row[f"{metadata_prefix}_vocab_size"])
    except (TypeError, ValueError, KeyError):
        return _logits_report(False, f"{logits_field} vocab_size is invalid", token_identity_match, False, False, False, False)
    if payload_vocab_size != row_vocab_size:
        return _logits_report(False, f"{logits_field} vocab_size mismatch", token_identity_match, False, False, False, True)

    indices = payload["indices"]
    values = payload["values"]
    if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
        return _logits_report(False, f"{logits_field} indices/values batch shape mismatch", token_identity_match, False, False, False, False)
    if len(indices) != 1:
        return _logits_report(False, f"{logits_field} batch dimension must be 1", token_identity_match, False, False, False, False)
    if not indices or not values:
        return _logits_report(False, f"{logits_field} indices/values are empty", token_identity_match, False, False, False, False)

    seq_indices = indices[0]
    seq_values = values[0]
    if not isinstance(seq_indices, list) or not isinstance(seq_values, list):
        return _logits_report(False, f"{logits_field} indices/values are not lists", token_identity_match, False, False, False, False)
    if len(seq_indices) != len(seq_values):
        return _logits_report(False, f"{logits_field} indices/values sequence length mismatch", token_identity_match, False, False, False, False)
    if len(seq_indices) == 0:
        return _logits_report(False, f"{logits_field} sequence length is zero", token_identity_match, False, False, False, False)
    if len(seq_indices) != answer_len:
        reason = f"{logits_field} length mismatch with teacher_tokens"
        return _logits_report(False, reason, token_identity_match, False, False, len(seq_indices) > answer_len, False)

    vocab_size = payload_vocab_size
    token_k = payload.get("token_k")
    token_k_rows = None
    if token_k is not None:
        if not isinstance(token_k, list) or len(token_k) != 1:
            return _logits_report(False, f"{logits_field}.token_k shape mismatch", token_identity_match, False, False, False, False)
        token_k_rows = token_k[0]
        if not isinstance(token_k_rows, list) or len(token_k_rows) != len(seq_indices):
            return _logits_report(False, f"{logits_field}.token_k length mismatch", token_identity_match, False, False, False, False)

    for position, (position_indices, position_values) in enumerate(zip(seq_indices, seq_values), start=0):
        if not isinstance(position_indices, list) or not isinstance(position_values, list):
            return _logits_report(False, f"{logits_field} position {position} is not a list", token_identity_match, False, False, False, False)
        if len(position_indices) != len(position_values):
            return _logits_report(
                False,
                f"{logits_field} indices/values top-k length mismatch at position {position}",
                token_identity_match,
                False,
                False,
                False,
                False,
            )
        if len(position_indices) <= 0:
            return _logits_report(False, f"{logits_field} top-k length is zero at position {position}", token_identity_match, False, False, False, False)

        expected_k = len(position_indices)
        if token_k_rows is not None:
            token_k_value = token_k_rows[position]
            try:
                token_k_int = int(token_k_value)
            except (TypeError, ValueError):
                return _logits_report(False, f"{logits_field}.token_k is not an integer at position {position}", token_identity_match, False, False, False, False)
            if token_k_int <= 0 or token_k_int > expected_k:
                return _logits_report(
                    False,
                    f"{logits_field}.token_k incompatible with active top-k at position {position}",
                    token_identity_match,
                    False,
                    False,
                    False,
                    False,
                )

        for token_index in position_indices:
            try:
                index_value = int(token_index)
            except (TypeError, ValueError):
                return _logits_report(False, f"{logits_field} contains a non-integer token index", token_identity_match, False, False, False, False)
            if index_value < 0 or index_value >= vocab_size:
                return _logits_report(False, f"{logits_field} token index out of range", token_identity_match, False, False, False, True)

    valid_teacher_logits = True
    logits_length_match = len(seq_indices) == answer_len
    return _logits_report(True, None, token_identity_match, valid_teacher_logits, logits_length_match, False, False)


def _validate_teacher_answer_schema(parsed: dict[str, Any]) -> tuple[bool, str | None, bool]:
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return False, "teacher_answer.elements is not a list", False

    string_list_row = False
    for index, element in enumerate(elements):
        if isinstance(element, str):
            string_list_row = True
            return False, f"teacher_answer.elements[{index}] is a string-list item", string_list_row
        if not isinstance(element, dict):
            return False, f"teacher_answer.elements[{index}] is not an object", string_list_row
        missing = {"text", "type", "focused"} - set(element)
        if missing:
            return False, f"teacher_answer.elements[{index}] missing {sorted(missing)}", string_list_row
        if not isinstance(element.get("text"), str):
            return False, f"teacher_answer.elements[{index}].text is not a string", string_list_row
        if not isinstance(element.get("type"), str):
            return False, f"teacher_answer.elements[{index}].type is not a string", string_list_row
        if element["type"] not in ALLOWED_ELEMENT_TYPES:
            ##return False, f"teacher_answer.elements[{index}].type is invalid", string_list_row
            return False, (
            f"teacher_answer.elements[{index}].type is invalid: "
            f"{element.get('type')!r}"
            ),string_list_row
        if not isinstance(element.get("focused"), bool):
            return False, f"teacher_answer.elements[{index}].focused is not a boolean", string_list_row
    return True, None, string_list_row


def _extract_teacher_tokens(row: dict[str, Any]) -> list[int]:
    tokens = row.get("teacher_tokens")
    if not isinstance(tokens, list):
        return []
    extracted: list[int] = []
    for token in tokens:
        try:
            extracted.append(int(token))
        except (TypeError, ValueError):
            return []
    return extracted


def _should_count_token_mismatch(
    report: _TeacherRowReport,
    *,
    decode_tokens: DecodeTokens | None,
    require_teacher_logits: bool,
) -> bool:
    if decode_tokens is None or report.answer_token_match:
        return False
    if require_teacher_logits and report.token_identity_match:
        return False
    return True


def _should_count_token_identity_mismatch(
    report: _TeacherRowReport,
    *,
    require_teacher_logits: bool,
) -> bool:
    return require_teacher_logits and (report.teacher_logits_present or report.reason is not None) and not report.token_identity_match


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _report(
    valid: bool,
    reason: str | None,
    *,
    decode_error: str | None = None,
    valid_json: bool = False,
    schema_valid: bool = False,
    string_list_row: bool = False,
    answer_token_match: bool = False,
    token_identity_match: bool = False,
    teacher_logits_present: bool = False,
    valid_teacher_logits: bool = False,
    logits_length_match: bool = False,
    full_sequence_logits: bool = False,
    vocab_mismatch: bool = False,
) -> _TeacherRowReport:
    return _TeacherRowReport(
        valid=valid,
        reason=reason,
        decode_error=decode_error,
        valid_json=valid_json,
        schema_valid=schema_valid,
        string_list_row=string_list_row,
        answer_token_match=answer_token_match,
        token_identity_match=token_identity_match,
        teacher_logits_present=teacher_logits_present,
        valid_teacher_logits=valid_teacher_logits,
        logits_length_match=logits_length_match,
        full_sequence_logits=full_sequence_logits,
        vocab_mismatch=vocab_mismatch,
    )


def _logits_report(
    valid: bool,
    reason: str | None,
    token_identity_match: bool,
    valid_teacher_logits: bool,
    logits_length_match: bool,
    full_sequence_logits: bool,
    vocab_mismatch: bool,
) -> _LogitsReport:
    return _LogitsReport(
        valid=valid,
        reason=reason,
        token_identity_match=token_identity_match,
        valid_teacher_logits=valid_teacher_logits,
        logits_length_match=logits_length_match,
        full_sequence_logits=full_sequence_logits,
        vocab_mismatch=vocab_mismatch,
    )


def _add_bad_row(
    bad_rows: list[dict[str, str]],
    row_id: str,
    reason: str,
    bad_limit: int,
) -> None:
    if len(bad_rows) < bad_limit:
        bad_rows.append({"id": row_id, "reason": reason})
