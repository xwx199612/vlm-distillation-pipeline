from __future__ import annotations

import argparse
from pathlib import Path

from .config_schema import load_config, resolve_label_path, resolve_prediction_path
from .data_manifest import validate_manifest
from .hf_runtime import configure_hf_offline_mode
from .label_validation import build_teacher_token_decoder, validate_label_rows
from .manifest_builder import create_manifest_from_config, infer_manifest_task_from_config_path
from .stage_evaluation import evaluate
from .stage_answer_labeling import create_distillation_dataset
from .stage_prediction_evaluation import evaluate_predictions
from .stage_student_prediction import create_student_predictions
from .stage_teacher_logits import create_teacher_logits_dataset
from .stage_student_training import train_student
from .stage_visual_switch_logits import create_visual_switch_dataset


def main() -> None:
    configure_hf_offline_mode()

    parser = argparse.ArgumentParser(prog="vlm-distill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_manifest_parser = subparsers.add_parser("create-manifest")
    create_manifest_parser.add_argument("--config", type=Path, required=True)
    create_manifest_parser.add_argument("--recursive", action="store_true")

    for command in (
        "validate-manifest",
        "validate-labels",
        "label",
        "predict",
        "teacher-logits",
        "switch-logits",
        "train",
        "evaluate",
        "evaluate-predictions",
    ):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create-manifest":
        config = load_config(args.config)
        task = infer_manifest_task_from_config_path(args.config)

        output_path = create_manifest_from_config(
            config=config,
            task=task,
            recursive=args.recursive,
        )
        print(f"OK manifest written: {output_path}")
        return

    config = load_config(args.config)

    if args.command == "validate-manifest":
        samples = validate_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        print(f"OK validated manifest samples={len(samples)} path={config.data.manifest_path}")
        return

    if args.command == "validate-labels":
        decoder = build_teacher_token_decoder(config)
        summary = validate_label_rows(
            resolve_label_path(config.data),
            max_samples=config.data.max_samples,
            decode_tokens=decoder,
        )
        print(
            "OK validated labels "
            f"path={resolve_label_path(config.data)} "
            f"total_rows={summary['total_rows']} "
            f"valid_json_rows={summary['valid_json_rows']} "
            f"schema_valid_rows={summary['schema_valid_rows']} "
            f"string_list_rows={summary['string_list_rows']} "
            f"answer_token_mismatch_rows={summary['answer_token_mismatch_rows']}"
        )
        if decoder is None:
            print("teacher_tokens decode check skipped: teacher tokenizer unavailable")
        if summary["bad_rows"]:
            print("first_bad_rows:")
            for bad_row in summary["bad_rows"]:
                print(f"  id={bad_row['id']} reason={bad_row['reason']}")
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

    if args.command == "predict":
        samples = validate_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        output_path = create_student_predictions(config, samples)
        print(f"OK student predictions written: {output_path}")
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

    if args.command == "evaluate-predictions":
        report_path = evaluate_predictions(config)
        print(
            "OK prediction eval report written: "
            f"{report_path} predictions={resolve_prediction_path(config.data)} "
            f"targets={resolve_label_path(config.data) if config.data.eval_path is None else config.data.eval_path}"
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
