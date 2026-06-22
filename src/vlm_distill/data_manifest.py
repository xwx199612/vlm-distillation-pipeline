from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any


@dataclass(frozen=True)
class VlmSample:
    id: str
    image: str
    task: str = "vqa"
    query: str | None = None
    target_label: str | None = None
    target_type: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_manifest(path: Path, image_root: Path = Path("."), max_samples: int | None = None) -> list[VlmSample]:
    rows = read_jsonl(path, max_samples=max_samples)
    samples: list[VlmSample] = []
    required = {"id", "image", "task"}

    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing required fields: {sorted(missing)}")

        image_path = image_root / row["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"{path}:{index} image not found: {image_path}")

        task = str(row["task"])
        target_label = row.get("target_label")
        if task == "grounding" and not target_label:
            raise ValueError(f"{path}:{index} grounding task requires target_label")

        known_keys = {
            "id",
            "image",
            "task",
            "query",
            "target_label",
            "target_type",
            "answer",
            "metadata",
        }
        metadata: dict[str, Any] = {}
        existing_metadata = row.get("metadata")
        if isinstance(existing_metadata, dict):
            metadata.update(existing_metadata)
        for key, value in row.items():
            if key not in known_keys:
                metadata[key] = value

        samples.append(
            VlmSample(
                id=str(row["id"]),
                image=str(row["image"]),
                task=task,
                query=str(row["query"]) if row.get("query") is not None else None,
                target_label=str(target_label) if target_label is not None else None,
                target_type=row.get("target_type"),
                answer=row.get("answer"),
                metadata=metadata,
            )
        )

    return samples


def summarize_label_rows(path: Path, max_samples: int | None = None) -> dict[str, int]:
    rows = read_jsonl(path, max_samples=max_samples)
    teacher_answer_rows = 0
    non_empty_teacher_answer_rows = 0

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        if "teacher_answer" not in row:
            continue
        teacher_answer_rows += 1
        value = row.get("teacher_answer")
        if isinstance(value, str):
            if value.strip():
                non_empty_teacher_answer_rows += 1
        elif value is not None:
            non_empty_teacher_answer_rows += 1

    return {
        "total_rows": len(rows),
        "teacher_answer_rows": teacher_answer_rows,
        "non_empty_teacher_answer_rows": non_empty_teacher_answer_rows,
    }
