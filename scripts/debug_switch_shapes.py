from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from vlm_distill.config_schema import format_prompt, load_config
from vlm_distill.data_manifest import validate_manifest
from vlm_distill.stage_visual_switch_logits import (
    VisualSwitchDistiller,
    _ensure_batch_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Switch-KD visual/text tensor shapes for one sample."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a Switch-KD YAML config.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index from the manifest to inspect.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=None,
    )
    if not samples:
        raise ValueError(f"No samples found in manifest: {config.data.manifest_path}")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range for {len(samples)} manifest rows."
        )

    sample = samples[args.sample_index]
    distiller = VisualSwitchDistiller(config)
    distiller.load()
    components = distiller._components()

    image_path = config.data.image_root / sample.image
    image = Image.open(image_path).convert("RGB")
    prompt = format_prompt(
        config.distillation.prompt_template,
        query=sample.query,
        target_label=sample.target_label,
        target_type=sample.target_type,
        task=sample.task,
    )

    student_inputs = distiller._student_image_inputs(image)
    visual_outputs = distiller._student_visual_outputs(student_inputs)
    projector_outputs = distiller._student_projector_outputs(visual_outputs, student_inputs)
    teacher_inputs = distiller._teacher_text_inputs(prompt)
    text_embeds = components.teacher_token_embedding(teacher_inputs["input_ids"])

    projected_visual = projector_outputs.to(text_embeds.device, dtype=text_embeds.dtype)
    projected_visual = _ensure_batch_sequence(projected_visual)

    visual_dim = int(projected_visual.shape[-1])
    teacher_dim = int(text_embeds.shape[-1])

    print(f"sample_id={sample.id}")
    print(f"image={sample.image}")
    print(f"student_vision_output_shape={tuple(visual_outputs.shape)}")
    print(f"student_projector_output_shape={tuple(projector_outputs.shape)}")
    print(f"teacher_embedding_shape={tuple(text_embeds.shape)}")
    print(f"teacher_embedding_dim={teacher_dim}")
    print(f"projector_output_dim={visual_dim}")
    print("align_triggered=false")
    print(f"projected_visual_shape={tuple(projected_visual.shape)}")


if __name__ == "__main__":
    main()
