from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

from .data_manifest import read_jsonl
from .stage_answer_labeling import (
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
    bad_limit: int = 5,
) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    valid_json_rows = 0
    schema_valid_rows = 0
    string_list_rows = 0
    answer_token_mismatch_rows = 0
    bad_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        answer = row.get("teacher_answer")
        if not isinstance(answer, str):
            _add_bad_row(bad_rows, row_id, "teacher_answer is missing or not a string", bad_limit)
            continue

        parsed = _parse_json_object(answer)
        if parsed is None:
            _add_bad_row(bad_rows, row_id, "teacher_answer is not valid JSON", bad_limit)
            continue
        valid_json_rows += 1

        elements = parsed.get("elements")
        if isinstance(elements, list) and any(isinstance(element, str) for element in elements):
            string_list_rows += 1

        schema_valid, schema_reason = _validate_parsing_teacher_answer(answer)
        if schema_valid:
            schema_valid_rows += 1
        else:
            _add_bad_row(bad_rows, row_id, schema_reason or "schema invalid", bad_limit)

        tokens = _extract_teacher_tokens(row)
        if tokens and decode_tokens is not None:
            decoded = _strip_special_tokens(decode_tokens(tokens))
            try:
                answer_canonical = _canonicalize_teacher_answer(answer)
                decoded_canonical = _canonicalize_teacher_answer(decoded)
            except ValueError as exc:
                answer_token_mismatch_rows += 1
                _add_bad_row(bad_rows, row_id, str(exc), bad_limit)
                continue

            if decoded_canonical != answer_canonical:
                answer_token_mismatch_rows += 1
                _add_bad_row(
                    bad_rows,
                    row_id,
                    "decoded teacher_tokens do not match teacher_answer",
                    bad_limit,
                )

    return {
        "total_rows": len(rows),
        "valid_json_rows": valid_json_rows,
        "schema_valid_rows": schema_valid_rows,
        "string_list_rows": string_list_rows,
        "answer_token_mismatch_rows": answer_token_mismatch_rows,
        "bad_rows": bad_rows,
    }


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


def _add_bad_row(
    bad_rows: list[dict[str, str]],
    row_id: str,
    reason: str,
    bad_limit: int,
) -> None:
    if len(bad_rows) < bad_limit:
        bad_rows.append({"id": row_id, "reason": reason})
