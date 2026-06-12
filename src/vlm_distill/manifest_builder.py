from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_schema import PipelineConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_IMAGE_DIR = Path("data/images")
DEFAULT_OUTPUT_DIR = Path("outputs")

TASK_DEFAULTS = {
    "screen_parsing": {
        "query": (
            "List all visible UI icons, buttons, menu items, text labels, "
            "and actionable elements on this screen."
        ),
    },
    "grounding": {
        "source_filename": "screen_parsing_teacher_labels.jsonl",
    },
}


def create_manifest_from_config(
    config: PipelineConfig,
    task: str,
    recursive: bool = False,
) -> Path:
    if task == "screen_parsing":
        return create_screen_parsing_manifest(
            image_dir=config.data.image_dir or DEFAULT_IMAGE_DIR,
            output_path=config.data.manifest_path,
            recursive=recursive,
        )

    if task == "grounding":
        output_dir = config.data.output_dir or DEFAULT_OUTPUT_DIR

        source_path = output_dir / TASK_DEFAULTS["grounding"]["source_filename"]

        return create_grounding_manifest(
            source_path=source_path,
            output_path=config.data.manifest_path,
        )

    raise ValueError(
        f"Unsupported task: {task}. "
        f"Available tasks: {sorted(TASK_DEFAULTS)}"
    )


def create_screen_parsing_manifest(
    image_dir: Path,
    output_path: Path,
    recursive: bool = False,
) -> Path:
    query = TASK_DEFAULTS["screen_parsing"]["query"]

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {image_dir}")

    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()

    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(images, start=1):
            row = {
                "id": f"screen_parsing-{index:06d}",
                "image": str(image_path).replace("\\", "/"),
                "task": "screen_parsing",
                "query": query,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created screen_parsing manifest: {output_path}")
    print(f"Image dir: {image_dir}")
    print(f"Samples: {len(images)}")

    return output_path


def create_grounding_manifest(
    source_path: Path,
    output_path: Path,
) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(
            f"screen_parsing teacher label file not found: {source_path}\n"
            "Run screen parsing label generation first."
        )

    source_rows = _read_jsonl(source_path)
    grounding_rows: list[dict[str, Any]] = []

    for row in source_rows:
        elements = _extract_elements(row)

        for element_index, element in enumerate(elements, start=1):
            label = _element_label(element)
            if not label:
                continue

            grounding_rows.append(
                {
                    "id": f"{row['id']}-grounding-{element_index:03d}",
                    "image": row["image"],
                    "task": "grounding",
                    "target_label": label,
                    "source_screen_parsing_id": row["id"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in grounding_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created grounding manifest: {output_path}")
    print(f"Source: {source_path}")
    print(f"Samples: {len(grounding_rows)}")

    return output_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc

    return rows


def _extract_elements(row: dict[str, Any]) -> list[Any]:
    parsed = _parse_json_like(row.get("student_target"))

    if parsed is None:
        parsed = _parse_json_like(row.get("teacher_answer"))

    if not isinstance(parsed, dict):
        return []

    for key in ("elements", "selectable_elements"):
        elements = parsed.get(key, [])
        if isinstance(elements, list):
            return elements

    return []


def _parse_json_like(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    end = value.rfind("}")

    if start >= 0 and end > start:
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def _element_label(element: Any) -> str | None:
    if isinstance(element, str):
        label = element.strip()
        return label or None

    if isinstance(element, dict):
        label = (
            element.get("label")
            or element.get("text")
            or element.get("name")
            or element.get("title")
        )

        if label:
            label = str(label).strip()
            return label or None

    return None
