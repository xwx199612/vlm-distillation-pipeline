from __future__ import annotations

import argparse
from pathlib import Path

from .config_schema import load_config
from .data_manifest import validate_manifest
from .stage_evaluation import evaluate
from .stage_answer_labeling import create_distillation_dataset
from .stage_teacher_logits import create_teacher_logits_dataset
from .stage_student_training import train_student
from .stage_visual_switch_logits import create_visual_switch_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="vlm-distill")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-data", "label", "teacher-logits", "switch-logits", "train", "evaluate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "validate-data":
        samples = validate_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        print(f"OK validated {len(samples)} samples")
        return

    if args.command == "label":
        samples = validate_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        output_path = create_distillation_dataset(config, samples)
        print(f"OK distillation dataset written: {output_path}")
        return

    if args.command == "teacher-logits":
        output_path = create_teacher_logits_dataset(config)
        print(f"OK teacher logits written: {output_path}")
        return

    if args.command == "switch-logits":
        output_path = create_visual_switch_dataset(config)
        print(f"OK visual-switch logits written: {output_path}")
        return

    if args.command == "train":
        artifact = train_student(config)
        print(f"OK student artifact written: {artifact}")
        return

    if args.command == "evaluate":
        report_path = evaluate(config)
        print(f"OK eval report written: {report_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
