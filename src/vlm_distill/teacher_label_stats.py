from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_manifest import read_jsonl
from .stage_teacher_precompute import _parse_json_object


SCHEMA_WORD_ELEMENTS = {
    "text",
    "type",
    "focused",
    "true",
    "false",
    "elements",
}


def summarize_teacher_label_file(path: Path, *, max_samples: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    total_elements = 0
    unknown_type_elements = 0
    empty_elements = 0
    schema_word_elements = 0

    for row in rows:
        answer = row.get("teacher_answer")
        parsed = _parse_teacher_answer(answer)
        elements = parsed.get("elements") if isinstance(parsed, dict) else None
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            total_elements += 1
            text = str(element.get("text") or "").strip()
            element_type = str(element.get("type") or "").strip().lower()
            if element_type == "unknown":
                unknown_type_elements += 1
            if not text:
                empty_elements += 1
            if text.lower() in SCHEMA_WORD_ELEMENTS:
                schema_word_elements += 1

    return {
        "path": str(path),
        "total_samples": len(rows),
        "total_elements": total_elements,
        "unknown_type_ratio": _safe_ratio(unknown_type_elements, total_elements),
        "empty_elements_ratio": _safe_ratio(empty_elements, total_elements),
        "schema_word_element_count": schema_word_elements,
    }


def _parse_teacher_answer(answer: object) -> dict[str, object] | None:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        return _parse_json_object(answer)
    return None


def _safe_ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def format_teacher_label_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"path={summary['path']}",
            f"total_samples={summary['total_samples']}",
            f"total_elements={summary['total_elements']}",
            f"unknown_type_ratio={summary['unknown_type_ratio']:.4f}",
            f"empty_elements_ratio={summary['empty_elements_ratio']:.4f}",
            f"schema_word_element_count={summary['schema_word_element_count']}",
        ]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python -m vlm_distill.teacher_label_stats")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    print(format_teacher_label_summary(summarize_teacher_label_file(args.path, max_samples=args.max_samples)))
