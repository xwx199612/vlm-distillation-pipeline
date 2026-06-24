from __future__ import annotations

from pathlib import Path
from typing import Any

from .teacher_validation import (
    build_teacher_token_decoder,
    validate_teacher_output_file,
    validate_teacher_row as _validate_teacher_row,
)


def validate_label_rows(
    path: Path,
    *,
    max_samples: int | None = None,
    decode_tokens=None,
    require_logits: bool = False,
    bad_limit: int = 5,
) -> dict[str, Any]:
    return validate_teacher_output_file(
        path,
        max_samples=max_samples,
        decode_tokens=decode_tokens,
        require_teacher_logits=require_logits,
        bad_limit=bad_limit,
    )


def validate_teacher_row(
    row: dict[str, Any],
    *,
    require_logits: bool = False,
    decode_tokens=None,
    logits_field: str = "teacher_logits",
) -> tuple[bool, str | None]:
    return _validate_teacher_row(
        row,
        require_teacher_logits=require_logits,
        decode_tokens=decode_tokens,
        logits_field=logits_field,
    )


__all__ = [
    "build_teacher_token_decoder",
    "validate_label_rows",
    "validate_teacher_output_file",
    "validate_teacher_row",
]
