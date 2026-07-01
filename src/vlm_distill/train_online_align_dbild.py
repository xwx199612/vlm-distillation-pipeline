from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_schema import load_config, format_prompt, resolve_label_path
from .data_manifest import read_jsonl
from .device_utils import batch_to_device, resolve_requested_device_map, resolve_training_device_map, select_model_input_device
from .loss_switch_kd import _causal_lm_loss, full_dynamic_bidirectional_logits_difference
from .model_loading import apply_attn_implementation, resolve_model_path
from .stage_student_training import VlmTrainingDataset
from .vlm_batching import build_supervision_mask, build_vlm_data_collator, encode_vlm_training_sample, load_training_image


VISION_FREEZE_KEYWORDS = (
    "visual.blocks",
    "vision_model.encoder",
    "vision_tower",
    "patch_embed",
    "visual.patch_embed",
    "visual.rotary_pos_emb",
    "visual.window_index",
)

TRAINABLE_KEYWORDS = (
    "merger",
    "projector",
    "connector",
    "language_model",
    "model.layers",
    "lm_head",
    "lora",
)


@dataclass(frozen=True)
class TrainableSummary:
    count: int
    total: int
    ratio: float
    names: list[str]


class OnlineAlignDataset(VlmTrainingDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(super().__getitem__(index))
        row = self.rows[index]
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=row.get("query") or metadata.get("query"),
            target_label=row.get("target_label") or metadata.get("target_label"),
            target_type=row.get("target_type") or metadata.get("target_type"),
            task=row.get("task", "vqa"),
        )
        item["sample_id"] = str(row["id"])
        item["image_path"] = str(row["image"])
        item["teacher_prompt"] = prompt
        item["teacher_answer"] = str(row["teacher_answer"])
        return item


class OnlineAlignCollator:
    def __init__(self, processor):
        self.base_collator = build_vlm_data_collator(processor, logits_fields=())
        self.metadata_keys = ("sample_id", "image_path", "teacher_prompt", "teacher_answer")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        cloned = [dict(feature) for feature in features]
        metadata = {key: [feature.pop(key) for feature in cloned] for key in self.metadata_keys}
        batch = self.base_collator(cloned)
        batch.update(metadata)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online teacher-student full-logits DBiLD training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config.training.max_steps.")
    return parser.parse_args()


def _resolve_torch_dtype(name: str | None):
    import torch

    if name is None:
        return torch.bfloat16
    normalized = str(name).strip().lower()
    mapping = {
        "auto": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return mapping[normalized]


def _build_model_kwargs(
    *,
    quantization: str,
    device_map: str | None,
    attn_implementation: str | None,
    torch_dtype_name: str | None,
    role: str,
) -> tuple[dict[str, Any], str | None]:
    import torch

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if role == "teacher":
        resolved_device_map = resolve_requested_device_map(device_map, quantization=quantization, role=role)
    else:
        resolved_device_map = resolve_training_device_map(device_map, quantization=quantization, role=role)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    apply_attn_implementation(model_kwargs, attn_implementation)

    if quantization == "none":
        dtype = _resolve_torch_dtype(torch_dtype_name) if torch_dtype_name is not None else torch.bfloat16
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig

        compute_dtype = _resolve_torch_dtype(torch_dtype_name)
        if compute_dtype is None:
            compute_dtype = torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization!r}")
    return model_kwargs, resolved_device_map


def _load_teacher(config):
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    from transformers import AutoProcessor

    teacher_model_path = resolve_model_path(config.teacher.model_name)
    processor = AutoProcessor.from_pretrained(
        teacher_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_kwargs, resolved_device_map = _build_model_kwargs(
        quantization=config.teacher.quantization,
        device_map=config.teacher.device_map,
        attn_implementation=config.teacher.attn_implementation,
        torch_dtype_name=config.teacher.torch_dtype,
        role="teacher",
    )
    model = AutoModelForVLM.from_pretrained(teacher_model_path, **model_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_device = select_model_input_device(model, label="teacher")
    return model, processor, teacher_model_path, resolved_device_map, input_device


def _load_student(config):
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    from transformers import AutoProcessor

    student_model_path = resolve_model_path(config.student.model_name)
    processor = AutoProcessor.from_pretrained(
        student_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_kwargs, resolved_device_map = _build_model_kwargs(
        quantization=config.student.quantization,
        device_map=config.student.device_map,
        attn_implementation=config.student.attn_implementation,
        torch_dtype_name="bfloat16",
        role="student",
    )
    model = AutoModelForVLM.from_pretrained(student_model_path, **model_kwargs)
    return model, processor, student_model_path, resolved_device_map


def freeze_student_vision_keep_merger_lm_trainable(model) -> TrainableSummary:
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in VISION_FREEZE_KEYWORDS):
            parameter.requires_grad_(False)
        if any(keyword in lowered for keyword in TRAINABLE_KEYWORDS):
            parameter.requires_grad_(True)

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for _, parameter in model.named_parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    ratio = float(trainable_count / total_count) if total_count else 0.0
    print(f"trainable_param_count={trainable_count}")
    print(f"trainable_param_ratio={ratio:.6f}")
    print("first_trainable_parameter_names=", trainable_names[:30])
    return TrainableSummary(
        count=int(trainable_count),
        total=int(total_count),
        ratio=ratio,
        names=trainable_names[:30],
    )


def _autocast_context(mixed_precision: str):
    import torch

    if not torch.cuda.is_available() or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision!r}")


def _build_teacher_inputs(batch, teacher_processor, config):
    import torch

    if len(batch["teacher_answer"]) != 1:
        raise ValueError("Online align training currently supports only batch_size == 1.")
    image = load_training_image(
        config.data.image_root,
        batch["image_path"][0],
        resize_mode=config.training.image_resize,
    )
    encoded = encode_vlm_training_sample(
        teacher_processor,
        image=image,
        prompt=batch["teacher_prompt"][0],
        target=batch["teacher_answer"][0],
        max_length=config.training.max_length,
        mask_prompt_labels=True,
        canonical_answer_span=True,
    )
    teacher_inputs = {
        key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim >= 1 else value
        for key, value in encoded.model_inputs.items()
        if key != "labels"
    }
    return teacher_inputs


def align_logits_to_supervised_positions(teacher_logits, student_logits, labels):
    import torch

    if teacher_logits.ndim != 3 or student_logits.ndim != 3:
        raise ValueError("teacher_logits and student_logits must have shape [batch, seq, vocab].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq].")
    if teacher_logits.shape[0] != student_logits.shape[0] or student_logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Batch size mismatch among teacher_logits, student_logits, and labels: "
            f"{tuple(teacher_logits.shape)}, {tuple(student_logits.shape)}, {tuple(labels.shape)}."
        )
    if teacher_logits.shape[-1] != student_logits.shape[-1]:
        raise ValueError(
            "Teacher/student vocab size mismatch: "
            f"{teacher_logits.shape[-1]} vs {student_logits.shape[-1]}."
        )
    if labels.shape[0] != 1:
        raise ValueError(
            "align_logits_to_supervised_positions currently requires batch_size == 1 "
            "to avoid ambiguous per-sample answer-length alignment."
        )

    supervised_mask = labels != -100
    supervised_count = int(supervised_mask[0].sum().item())
    if supervised_count <= 0:
        raise ValueError("No supervised answer tokens found in labels.")
    if supervised_count > int(student_logits.shape[1]) or supervised_count > int(teacher_logits.shape[1]):
        raise ValueError(
            "Reliable suffix alignment is not possible because supervised_count exceeds sequence length. "
            f"supervised_count={supervised_count}, teacher_seq={teacher_logits.shape[1]}, student_seq={student_logits.shape[1]}."
        )

    student_answer_logits = student_logits[supervised_mask].view(1, supervised_count, student_logits.shape[-1])
    teacher_answer_logits = teacher_logits[:, -supervised_count:, :]
    aligned_attention_mask = torch.ones(
        (1, supervised_count),
        device=student_logits.device,
        dtype=torch.float32,
    )
    return teacher_answer_logits, student_answer_logits, aligned_attention_mask, supervised_count


def _validate_rows(config) -> list[dict[str, Any]]:
    path = resolve_label_path(config.data)
    rows = read_jsonl(path, max_samples=config.data.max_samples)
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object.")
        for key in ("id", "image"):
            if row.get(key) in (None, ""):
                raise ValueError(f"{path}:{index} missing required field: {key}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not (row.get("query") or metadata.get("query")):
            raise ValueError(f"{path}:{index} missing query and metadata.query")
        if row.get("teacher_answer") in (None, ""):
            raise ValueError(f"{path}:{index} missing required teacher_answer")
        if row.get("teacher_tokens") is None:
            raise ValueError(f"{path}:{index} missing required teacher_tokens")
        validated.append(row)
    if not validated:
        raise ValueError(f"No training rows found in {path}.")
    return validated


def _maybe_enable_student_lora(config, model):
    if not config.student.use_lora:
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.student.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)
    target_modules = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=config.student.lora_rank,
        lora_alpha=config.student.lora_alpha,
        lora_dropout=config.student.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _apply_student_train_setup(config, model):
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model = _maybe_enable_student_lora(config, model)
    summary = freeze_student_vision_keep_merger_lm_trainable(model)
    model.train()
    return model, summary


def _build_optimizer(config, model):
    import torch

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable_parameters, lr=config.training.learning_rate)


def _dataloader(dataset, processor, batch_size: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=OnlineAlignCollator(processor),
    )


def _gpu_mem_stats() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def run_training(config, *, max_steps_override: int | None = None) -> Path:
    import torch

    if config.training.batch_size != 1:
        raise ValueError(
            "This online full-logits DBiLD script currently requires training.batch_size == 1 "
            "for safe supervised suffix alignment."
        )

    rows = _validate_rows(config)
    teacher_model, teacher_processor, teacher_model_path, _teacher_device_map, teacher_input_device = _load_teacher(config)
    student_model, student_processor, student_model_path, _student_device_map = _load_student(config)
    student_model, trainable_summary = _apply_student_train_setup(config, student_model)
    student_input_device = select_model_input_device(student_model, label="student")

    dataset = OnlineAlignDataset(rows, config, student_processor)
    dataloader = _dataloader(dataset, student_processor, config.training.batch_size)
    optimizer = _build_optimizer(config, student_model)

    max_steps = max_steps_override if max_steps_override is not None else config.training.max_steps
    total_target_steps = int(max_steps) if max_steps is not None else None
    grad_accum_steps = int(config.training.gradient_accumulation_steps)
    global_step = 0
    first_batch_debug_printed = False

    print("Online Align DBiLD training")
    print(f"teacher_model_path={teacher_model_path}")
    print(f"student_model_path={student_model_path}")
    print(f"teacher_quantization={config.teacher.quantization}")
    print(f"student_quantization={config.student.quantization}")
    print(f"mixed_precision={config.training.mixed_precision}")
    print("student vision frozen: true")
    print("VSD enabled: false")
    print(f"trainable_param_count={trainable_summary.count}")
    print(f"trainable_param_ratio={trainable_summary.ratio:.6f}")

    for epoch in range(int(config.training.epochs)):
        if total_target_steps is not None and global_step >= total_target_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(dataloader, start=1):
            if total_target_steps is not None and global_step >= total_target_steps:
                break

            teacher_inputs = _build_teacher_inputs(batch, teacher_processor, config)
            teacher_inputs = batch_to_device(teacher_inputs, teacher_input_device)

            student_batch = {
                key: value
                for key, value in batch.items()
                if key not in {"sample_id", "image_path", "teacher_prompt", "teacher_answer", "prompt_token_len"}
            }
            student_batch = batch_to_device(student_batch, student_input_device)
            labels = student_batch["labels"]

            with torch.no_grad():
                with _autocast_context(config.training.mixed_precision):
                    teacher_outputs = teacher_model(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits

            with _autocast_context(config.training.mixed_precision):
                student_outputs = student_model(**student_batch)
                student_logits = student_outputs.logits
                lm_loss = _causal_lm_loss(student_logits, labels)
                supervision_mask = build_supervision_mask(labels)
                aligned_teacher_logits, aligned_student_logits, aligned_attention_mask, supervised_count = (
                    align_logits_to_supervised_positions(teacher_logits, student_logits, labels)
                )
                align_loss = full_dynamic_bidirectional_logits_difference(
                    reference_logits=aligned_teacher_logits,
                    target_logits=aligned_student_logits,
                    attention_mask=aligned_attention_mask,
                    temperature=config.distillation.kd_temperature,
                    top_k=config.distillation.dbild_top_k,
                    top_k_mode=config.distillation.dbild_top_k_mode,
                    kneedle_candidate_k=config.distillation.dbild_kneedle_candidate_k,
                    min_top_k=config.distillation.dbild_min_top_k,
                    max_top_k=config.distillation.dbild_max_top_k,
                    kl_mode=config.distillation.dbild_kl_mode,
                )
                # VSD is intentionally disabled because teacher and student share the same vision backbone.
                # This script targets online full-logits DBiLD L_Align reproduction, not full Switch-KD with VSD.
                total_loss = lm_loss + align_loss

            if not first_batch_debug_printed:
                print(f"teacher_logits.shape={tuple(teacher_logits.shape)}")
                print(f"student_logits.shape={tuple(student_logits.shape)}")
                print(f"labels.shape={tuple(labels.shape)}")
                print(f"supervised_count={supervised_count}")
                print(f"supervision_mask.sum()={float(supervision_mask.sum().item())}")
                print(f"aligned_teacher_logits.shape={tuple(aligned_teacher_logits.shape)}")
                print(f"aligned_student_logits.shape={tuple(aligned_student_logits.shape)}")
                print(f"aligned_attention_mask.shape={tuple(aligned_attention_mask.shape)}")
                first_batch_debug_printed = True

            loss_for_backward = total_loss / grad_accum_steps
            loss_for_backward.backward()

            if micro_step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if config.training.log_every > 0 and global_step % int(config.training.log_every) == 0:
                    gpu_mem_allocated, gpu_mem_reserved = _gpu_mem_stats()
                    lr = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"step={global_step} "
                        f"lm_loss={float(lm_loss.detach().float().item()):.6f} "
                        f"align_loss={float(align_loss.detach().float().item()):.6f} "
                        f"total_loss={float(total_loss.detach().float().item()):.6f} "
                        f"lr={lr:.8g} "
                        f"gpu_mem_allocated={gpu_mem_allocated} "
                        f"gpu_mem_reserved={gpu_mem_reserved}"
                    )

                if total_target_steps is not None and global_step >= total_target_steps:
                    break

        if micro_step % grad_accum_steps != 0 and (total_target_steps is None or global_step < total_target_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    config.student.adapter_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(config.student.adapter_dir)
    student_processor.save_pretrained(config.student.adapter_dir)
    return config.student.adapter_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
