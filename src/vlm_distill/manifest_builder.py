from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_IMAGE_DIR = Path(r"D:\TV_data\test_data")

TASK_DEFAULTS = {
    "screen_parsing": {
        "output": Path("data/screen_parsing_test.jsonl"),
        "instruction": "List all visible UI icons, buttons, menu items, text labels, and actionable elements on this screen.",
    },
    "grounding": {
        "output": Path("data/grounding_test.jsonl"),
        "instruction_template": "Locate the {target_label} on this screen.",
        "source": Path("outputs/screen_parsing_teacher_labels.jsonl"),
    },
}

"""
    vlm-distill create-manifest --task screen_parsing
    vlm-distill label --config configs/screen_parsing_test.yaml

    vlm-distill create-manifest --task grounding
    vlm-distill label --config configs/grounding_test.yaml
"""

def create_manifest(
    task: str,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_path: Path | None = None,
    instruction: str | None = None,
    recursive: bool = False,
) -> Path:
    if task == "screen_parsing":
        return create_screen_parsing_manifest(
            image_dir=image_dir,
            output_path=output_path,
            instruction=instruction,
            recursive=recursive,
        )

    if task == "grounding":
        return create_grounding_manifest(
            source_path=TASK_DEFAULTS["grounding"]["source"],
            output_path=output_path,
        )

    raise ValueError(
        f"Unsupported task: {task}. "
        f"Available tasks: {sorted(TASK_DEFAULTS)}"
    )


def create_screen_parsing_manifest(
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_path: Path | None = None,
    instruction: str | None = None,
    recursive: bool = False,
) -> Path:
    defaults = TASK_DEFAULTS["screen_parsing"]
    output_path = output_path or defaults["output"]
    instruction = instruction or defaults["instruction"]

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

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
                "instruction": instruction,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created screen_parsing manifest: {output_path}")
    print(f"Samples: {len(images)}")
    return output_path


def create_grounding_manifest(
    source_path: Path,
    output_path: Path | None = None,
) -> Path:
    defaults = TASK_DEFAULTS["grounding"]
    output_path = output_path or defaults["output"]
    instruction_template = defaults["instruction_template"]

    if not source_path.exists():
        raise FileNotFoundError(
            f"screen_parsing label file not found: {source_path}\n"
            "Run this first:\n"
            "vlm-distill label --config configs/screen_parsing_test.yaml"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_path)
    grounding_rows: list[dict[str, Any]] = []

    for row in rows:
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
                    "instruction": instruction_template.format(target_label=label),
                    "source_screen_parsing_id": row["id"],
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in grounding_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created grounding manifest: {output_path}")
    print(f"Source: {source_path}")
    print(f"Samples: {len(grounding_rows)}")
    return output_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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

    elements = parsed.get("elements", [])
    return elements if isinstance(elements, list) else []


def _parse_json_like(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(value[start : end + 1])
            except json.JSONDecodeError:
                return None

    return None


def _element_label(element: Any) -> str | None:
    if isinstance(element, str):
        return element.strip() or None

    if isinstance(element, dict):
        label = element.get("label") or element.get("text") or element.get("name")
        if label:
            return str(label).strip()

    return None