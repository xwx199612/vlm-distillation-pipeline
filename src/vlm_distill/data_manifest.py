from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any


@dataclass(frozen=True)
class VlmSample:
    id: str
    image: str
    query: str | None = None
    answer: str | None = None
    task: str = "vqa"
    target_label: str | None = None
    bbox: list[float] | None = None
    elements: list[dict[str, Any]] | None = None


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
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
    required = {"id", "image"}

    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing required fields: {sorted(missing)}")

        image_path = image_root / row["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"{path}:{index} image not found: {image_path}")

        task = str(row.get("task", "vqa"))
        query = row.get("query")

        if not query:
            if task == "screen_parsing":
                query = "List all visible UI icons, buttons, menu items, and actionable elements."
            elif task == "grounding" and row.get("target_label"):
                query = f"Locate the {row['target_label']} in the image."
            else:
                raise ValueError(
                    f"{path}:{index} requires query, "
                    "or target_label for grounding."
                )

        if task == "grounding" and not row.get("target_label"):
            raise ValueError(f"{path}:{index} grounding task requires target_label")

        samples.append(
            VlmSample(
                id=str(row["id"]),
                image=str(row["image"]),
                query=str(query) if query is not None else None,
                answer=row.get("answer"),
                task=task,
                target_label=row.get("target_label"),
                bbox=row.get("bbox"),
                elements=row.get("elements"),
            )
        )

    return samples