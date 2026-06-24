from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_distill.compare_outputs import build_teacher_student_unique_rows
from vlm_distill.config_schema import load_config, resolve_label_path, resolve_prediction_path
from vlm_distill.data_manifest import read_jsonl, validate_manifest, write_jsonl
from vlm_distill.stage_teacher_precompute import create_distillation_dataset
from vlm_distill.stage_student_prediction import create_student_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a parsing manifest from a new image folder, run teacher labeling, "
            "run merged-student prediction, then compare unique UI elements."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    workflow = raw.get("workflow") or {}
    offline = bool(workflow.get("offline", False))
    image_dir = Path(raw["data"]["image_dir"])
    manifest_path = Path(raw["data"]["manifest_path"])
    comparison_output_path = Path(workflow["comparison_output_path"])
    teacher_name = str(workflow.get("teacher_name", "teacher"))
    student_name = str(workflow.get("student_name", "student"))
    recursive = bool(workflow.get("recursive", False))
    keep_empty = not bool(workflow.get("drop_empty", False))
    query = str(
        workflow.get("query")
        or "List all visible interactive UI elements on this screen."
    )

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = load_config(args.config)

    create_parsing_manifest_with_query(
        image_dir=image_dir,
        output_path=manifest_path,
        query=query,
        recursive=recursive,
    )
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    label_path = create_distillation_dataset(config, samples)
    prediction_path = create_student_predictions(config, samples)

    teacher_rows = read_jsonl(resolve_label_path(config.data))
    student_rows = read_jsonl(resolve_prediction_path(config.data))
    comparison_rows = build_teacher_student_unique_rows(
        teacher_rows=teacher_rows,
        student_rows=student_rows,
        teacher_name=teacher_name,
        student_name=student_name,
        keep_empty=keep_empty,
    )
    write_jsonl(comparison_output_path, comparison_rows)

    print(f"Manifest written: {manifest_path}")
    print(f"Teacher labels written: {label_path}")
    print(f"Student predictions written: {prediction_path}")
    print(f"Unique comparison written: {comparison_output_path}")
    if offline:
        print("Offline mode: enabled")


def create_parsing_manifest_with_query(
    *,
    image_dir: Path,
    output_path: Path,
    query: str,
    recursive: bool = False,
) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in image_exts
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(images, start=1):
            row = {
                "id": f"parsing-{index:06d}",
                "image": str(image_path).replace("\\", "/"),
                "task": "parsing",
                "query": query,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    main()
