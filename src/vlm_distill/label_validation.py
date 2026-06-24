from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

from .data_manifest import read_jsonl
from .stage_teacher_precompute import (
    _canonicalize_teacher_answer,
    _parse_json_object,
    _strip_special_tokens,
    _validate_parsing_teacher_answer,
)


DecodeTokens = Callable[[list[int]], str]


def validate_label_rows(
    path: Path,
    *,
    max_samples: int | None = None,
    decode_tokens: DecodeTokens | None = None,
    require_logits: bool = False,
    bad_limit: int = 5,
) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    valid_json_rows = 0
    schema_valid_rows = 0
    string_list_rows = 0
    answer_token_mismatch_rows = 0
    valid_teacher_logits_rows = 0
    answer_logits_length_mismatch_rows = 0
    valid_teacher_answer_rows = 0
    bad_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        valid, reason = validate_teacher_row(
            row,
            require_logits=require_logits,
            decode_tokens=decode_tokens,
        )
        answer = row.get("teacher_answer")
        if isinstance(answer, str):
            valid_teacher_answer_rows += 1
            parsed = _parse_json_object(answer)
            if parsed is not None:
                valid_json_rows += 1
                elements = parsed.get("elements")
                if isinstance(elements, list) and any(isinstance(element, str) for element in elements):
                    string_list_rows += 1
            schema_valid, _schema_reason = _validate_parsing_teacher_answer(answer)
            if schema_valid:
                schema_valid_rows += 1
        tokens = _extract_teacher_tokens(row)
        if tokens and decode_tokens is not None and isinstance(answer, str):
            try:
                decoded = _strip_special_tokens(decode_tokens(tokens))
                if _canonicalize_teacher_answer(answer) != _canonicalize_teacher_answer(decoded):
                    answer_token_mismatch_rows += 1
            except ValueError:
                answer_token_mismatch_rows += 1
        logits_valid, logits_reason = _validate_logits_alignment(row, "teacher_logits")
        if logits_valid:
            valid_teacher_logits_rows += 1
        elif row.get("teacher_logits") is not None or require_logits:
            if logits_reason and "length" in logits_reason:
                answer_logits_length_mismatch_rows += 1
        if not valid:
            _add_bad_row(bad_rows, row_id, reason or "invalid teacher row", bad_limit)

    return {
        "total_rows": len(rows),
        "valid_teacher_answer_rows": valid_teacher_answer_rows,
        "valid_json_rows": valid_json_rows,
        "schema_valid_rows": schema_valid_rows,
        "string_list_rows": string_list_rows,
        "answer_token_mismatch_rows": answer_token_mismatch_rows,
        "valid_teacher_logits_rows": valid_teacher_logits_rows,
        "answer_logits_length_mismatch_rows": answer_logits_length_mismatch_rows,
        "bad_rows": bad_rows,
    }


def validate_teacher_row(
    row: dict[str, Any],
    *,
    require_logits: bool,
    decode_tokens: DecodeTokens | None = None,
    logits_field: str = "teacher_logits",
) -> tuple[bool, str | None]:
    answer = row.get("teacher_answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "teacher_answer is missing or not a string"
    if _parse_json_object(answer) is None:
        return False, "teacher_answer is not valid JSON"
    schema_valid, schema_reason = _validate_parsing_teacher_answer(answer)
    if not schema_valid:
        return False, schema_reason or "schema invalid"
    tokens = _extract_teacher_tokens(row)
    if not tokens:
        return False, "teacher_tokens missing or empty"
    if decode_tokens is not None:
        try:
            decoded = _strip_special_tokens(decode_tokens(tokens))
            if _canonicalize_teacher_answer(answer) != _canonicalize_teacher_answer(decoded):
                return False, "decoded teacher_tokens do not match teacher_answer"
        except ValueError as exc:
            return False, str(exc)
    if require_logits:
        valid_logits, reason = _validate_logits_alignment(row, logits_field)
        if not valid_logits:
            return False, reason
    return True, None


def _validate_logits_alignment(row: dict[str, Any], field_name: str) -> tuple[bool, str | None]:
    payload = row.get(field_name)
    if not isinstance(payload, dict):
        return False, f"{field_name} missing or not a dict"
    if not all(key in payload for key in ("indices", "values", "vocab_size", "shape")):
        return False, f"{field_name} missing indices, values, vocab_size, or shape"
    shape = payload.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return False, f"{field_name}.shape invalid"
    indices = payload.get("indices")
    values = payload.get("values")
    if _nested_shape(indices) != _nested_shape(values) or not _nested_shape(indices):
        return False, f"{field_name} indices/values shape mismatch"
    tokens = _extract_teacher_tokens(row)
    seq_len = int(shape[1])
    if seq_len != len(tokens):
        return False, f"{field_name} length mismatch with teacher_tokens"
    if not bool(row.get(f"{field_name}_aligned_to_answer", False)):
        return False, f"{field_name}_aligned_to_answer is not true"
    return True, None


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


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return ()
    first_shape = _nested_shape(value[0])
    for item in value[1:]:
        if _nested_shape(item) != first_shape:
            return ()
    return (len(value), *first_shape)


def _add_bad_row(
    bad_rows: list[dict[str, str]],
    row_id: str,
    reason: str,
    bad_limit: int,
) -> None:
    if len(bad_rows) < bad_limit:
        bad_rows.append({"id": row_id, "reason": reason})
