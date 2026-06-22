from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_schema import (
    PipelineConfig,
    format_prompt,
    resolve_label_path,
    resolve_switch_logits_path,
    resolve_teacher_logits_path,
)
from .data_manifest import read_jsonl
from .logits_cache_utils import (
    align_reference_logits,
    align_reference_logits_to_suffix,
    cached_vocab_size,
    materialize_cached_logits,
    vocab_sizes_compatible,
)
from .model_loading import apply_attn_implementation, resolve_model_path


class VlmTrainingDataset:
    """Tokenize multimodal samples lazily to avoid Arrow overflows on image tensors."""

    def __init__(self, rows: list[dict[str, Any]], config: PipelineConfig, processor):
        self.rows = rows
        self.config = config
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .vlm_batching import encode_vlm_training_sample, load_training_image

        example = self.rows[index]
        image = load_training_image(
            self.config.data.image_root,
            example["image"],
            resize_mode=self.config.training.image_resize,
        )
        metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=example.get("query") or metadata.get("query"),
            target_label=example.get("target_label") or metadata.get("target_label"),
            target_type=example.get("target_type") or metadata.get("target_type"),
            task=example.get("task", "vqa"),
        )
        target = example["teacher_answer"]
        encoded = encode_vlm_training_sample(
            self.processor,
            image=image,
            prompt=prompt,
            target=target,
            max_length=self.config.training.max_length,
            mask_prompt_labels=self.config.training.mask_prompt_labels,
        )
        item = dict(encoded.model_inputs)
        item["prompt_token_len"] = encoded.prompt_token_len

        teacher_field = self.config.distillation.teacher_logits_field
        switch_field = self.config.distillation.switch_logits_field
        if teacher_field in example:
            item[teacher_field] = example[teacher_field]
            item[f"{teacher_field}_prompt_len"] = example.get(f"{teacher_field}_prompt_len")
            item[f"{teacher_field}_vocab_size"] = example.get(f"{teacher_field}_vocab_size")
        if switch_field in example:
            item[switch_field] = example[switch_field]
            item[f"{switch_field}_prompt_len"] = example.get(f"{switch_field}_prompt_len")
            item[f"{switch_field}_vocab_size"] = example.get(f"{switch_field}_vocab_size")
        if "teacher_confidence" in example and example.get("teacher_confidence") is not None:
            item["teacher_confidence"] = float(example["teacher_confidence"])
        return item


@dataclass(frozen=True)
class VocabAlignment:
    shared_token_vocab_size: int


def train_student(config: PipelineConfig) -> Path:
    rows = _load_training_rows(config)
    config.student.output_dir.mkdir(parents=True, exist_ok=True)
    config.student.adapter_dir.mkdir(parents=True, exist_ok=True)

    if config.student.model_name.startswith("mock-"):
        return _train_mock_student(config, rows)

    return _train_hf_student(config, rows)


def _train_mock_student(config: PipelineConfig, rows: list[dict]) -> Path:
    artifact = {
        "model_name": config.student.model_name,
        "num_training_samples": len(rows),
        "target_field": "teacher_answer",
        "distillation_method": config.distillation.method,
        "note": "Mock student artifact. Use configs/hf_vlm.yaml for real training.",
    }
    output_path = config.student.adapter_dir / "mock_adapter.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Mock student artifact written: {output_path}")
    return output_path


def _train_hf_student(config: PipelineConfig, rows: list[dict]) -> Path:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install transformers, datasets and peft to run real training.") from exc

    from .vlm_batching import build_vlm_data_collator

    student_model_path = resolve_model_path(config.student.model_name)
    processor = AutoProcessor.from_pretrained(
        student_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = _load_student_model(config, student_model_path)

    if config.student.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)

    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if config.training.freeze_vision_tower:
        _freeze_vision_modules(model)

    if config.student.use_lora:
        target_modules = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=config.student.lora_rank,
            lora_alpha=config.student.lora_alpha,
            lora_dropout=config.student.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    train_dataset = VlmTrainingDataset(rows, config, processor)
    data_collator = build_vlm_data_collator(processor)
    args = TrainingArguments(
        output_dir=str(config.student.output_dir),
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        max_steps=config.training.max_steps or -1,
        logging_steps=config.training.log_every,
        save_steps=config.training.save_every,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.mixed_precision == "fp16",
        bf16=config.training.mixed_precision == "bf16",
        remove_unused_columns=False,
    )

    trainer_cls = Trainer
    trainer_kwargs: dict = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if config.distillation.method == "switch_kd":
        trainer_cls = _build_switch_kd_trainer()
        _warn_if_switch_logits_missing(config, rows)
        trainer_kwargs["switch_kd_config"] = config
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(config.student.adapter_dir)
    processor.save_pretrained(config.student.adapter_dir)
    return config.student.adapter_dir


def _load_student_model(config: PipelineConfig, model_path: str | None = None):
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover - fallback for older transformers
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    model_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    apply_attn_implementation(model_kwargs, config.student.attn_implementation)
    if config.student.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.student.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs["local_files_only"] = True
    model_name_or_path = model_path or resolve_model_path(config.student.model_name)
    return AutoModelForVLM.from_pretrained(model_name_or_path, **model_kwargs)


def _build_switch_kd_trainer():
    from transformers import Trainer

    from .loss_switch_kd import SwitchKDLoss
    from .vlm_batching import build_supervision_mask

    class SwitchKDTrainer(Trainer):
        _vocab_warning_emitted: set[str] = set()

        def __init__(self, *args, switch_kd_config: PipelineConfig, **kwargs):
            super().__init__(*args, **kwargs)
            self.switch_kd_config = switch_kd_config
            distill = switch_kd_config.distillation
            self.vocab_alignment = _build_vocab_alignment(switch_kd_config)
            self.switch_kd_loss = SwitchKDLoss(
                lm_weight=distill.lm_loss_weight,
                dbild_weight=distill.dbild_loss_weight,
                vsd_weight=distill.vsd_loss_weight,
                temperature=distill.kd_temperature,
                top_k=distill.dbild_top_k,
                min_prob=distill.dbild_min_prob,
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            distill = self.switch_kd_config.distillation
            teacher_field = distill.teacher_logits_field
            switch_field = distill.switch_logits_field

            teacher_cached = inputs.pop(teacher_field, None)
            switch_cached = inputs.pop(switch_field, None)
            teacher_prompt_len = _pop_metadata(inputs, f"{teacher_field}_prompt_len")
            switch_prompt_len = _pop_metadata(inputs, f"{switch_field}_prompt_len")
            teacher_vocab_size = _pop_metadata(inputs, f"{teacher_field}_vocab_size")
            switch_vocab_size = _pop_metadata(inputs, f"{switch_field}_vocab_size")
            student_prompt_len = _pop_metadata(inputs, "prompt_token_len")
            teacher_confidence = _pop_float_metadata(inputs, "teacher_confidence")

            labels = inputs.get("labels")
            outputs = model(**inputs)
            student_logits = outputs.logits
            target_shape = tuple(student_logits.shape)
            device = student_logits.device
            dtype = student_logits.dtype
            student_vocab_size = int(student_logits.shape[-1])

            teacher_logits = _prepare_reference_logits(
                cached=teacher_cached,
                label="teacher",
                distill=distill,
                student_vocab_size=student_vocab_size,
                reference_vocab_size_meta=teacher_vocab_size,
                target_shape=target_shape,
                device=device,
                dtype=dtype,
                student_prompt_len=student_prompt_len,
                reference_prompt_len=teacher_prompt_len,
                warning_bucket=self._vocab_warning_emitted,
                vocab_alignment=self.vocab_alignment,
            )
            teacher_token_weight = _prepare_reference_token_weight(
                cached=teacher_cached,
                label="teacher",
                target_shape=target_shape,
                student_prompt_len=student_prompt_len,
                reference_prompt_len=teacher_prompt_len,
                device=device,
                dtype=dtype,
            )
            switch_logits = _prepare_reference_logits(
                cached=switch_cached,
                label="switch",
                distill=distill,
                student_vocab_size=student_vocab_size,
                reference_vocab_size_meta=switch_vocab_size,
                target_shape=target_shape,
                device=device,
                dtype=dtype,
                student_prompt_len=student_prompt_len,
                reference_prompt_len=switch_prompt_len,
                warning_bucket=self._vocab_warning_emitted,
                vocab_alignment=self.vocab_alignment,
            )
            switch_token_weight = _prepare_reference_token_weight(
                cached=switch_cached,
                label="switch",
                target_shape=target_shape,
                student_prompt_len=student_prompt_len,
                reference_prompt_len=switch_prompt_len,
                device=device,
                dtype=dtype,
            )

            supervision_mask = build_supervision_mask(labels)
            loss_output = self.switch_kd_loss(
                student_logits=student_logits,
                labels=labels,
                teacher_logits=teacher_logits,
                switch_logits=switch_logits,
                attention_mask=supervision_mask,
                teacher_token_weight=teacher_token_weight,
                switch_token_weight=switch_token_weight,
                sample_weight=teacher_confidence if distill.confidence_weighting else None,
            )
            return (loss_output.loss, outputs) if return_outputs else loss_output.loss

    return SwitchKDTrainer


def _prepare_reference_logits(
    *,
    cached,
    label: str,
    distill,
    student_vocab_size: int,
    reference_vocab_size_meta: int | None,
    target_shape: tuple[int, ...],
    device,
    dtype,
    student_prompt_len: int | None,
    reference_prompt_len: int | None,
    warning_bucket: set[str],
    vocab_alignment: VocabAlignment | None,
):
    if cached is None:
        return None

    if reference_vocab_size_meta is not None:
        reference_vocab = int(reference_vocab_size_meta)
    else:
        reference_vocab = cached_vocab_size(cached)

    tensor = materialize_cached_logits(
        cached,
        device=device,
        dtype=dtype,
        vocab_size=reference_vocab,
    )
    effective_reference_prompt_len = _normalize_reference_prompt_len(reference_prompt_len, tensor.shape[1])
    if not vocab_sizes_compatible(reference_vocab, student_vocab_size):
        remapped = _remap_reference_logits_to_student_vocab(
            tensor,
            reference_vocab=reference_vocab,
            student_vocab_size=student_vocab_size,
            vocab_alignment=vocab_alignment,
        )
        if remapped is None:
            if distill.skip_kd_on_vocab_mismatch:
                key = f"{label}:{reference_vocab}->{student_vocab_size}"
                if key not in warning_bucket:
                    print(
                        f"Warning: Skipping {label} KD because cached vocab_size={reference_vocab} "
                        f"does not match student vocab_size={student_vocab_size}."
                    )
                    warning_bucket.add(key)
                return None
        else:
            key = f"{label}:{reference_vocab}->{student_vocab_size}:remapped"
            if key not in warning_bucket:
                print(
                    f"Info: Remapped {label} KD logits from vocab_size={reference_vocab} "
                    f"to student vocab_size={student_vocab_size} using shared_token_vocab_size="
                    f"{vocab_alignment.shared_token_vocab_size if vocab_alignment else 'unknown'}."
                )
                warning_bucket.add(key)
            tensor = remapped
    elif tensor.shape[-1] != student_vocab_size:
        tensor = align_reference_logits(tensor, target_shape=(*tensor.shape[:-1], student_vocab_size), dtype=dtype)

    if distill.align_kd_logits_to_answer:
        return align_reference_logits_to_suffix(
            tensor,
            target_shape=target_shape,
            reference_prompt_len=effective_reference_prompt_len,
            student_prompt_len=student_prompt_len,
            dtype=dtype,
        )
    return align_reference_logits(tensor, target_shape=target_shape, dtype=dtype)


def _pop_metadata(inputs: dict, key: str) -> int | None:
    value = inputs.pop(key, None)
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    return int(value)


def _pop_float_metadata(inputs: dict, key: str) -> float | None:
    value = inputs.pop(key, None)
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    return float(value)


def _prepare_reference_token_weight(
    *,
    cached,
    label: str,
    target_shape: tuple[int, ...],
    student_prompt_len: int | None,
    reference_prompt_len: int | None,
    device,
    dtype,
):
    from .logits_cache_utils import cached_token_weight, align_reference_token_weight_to_suffix

    if cached is None:
        return None

    token_weight = cached_token_weight(cached, device=device, dtype=dtype)
    if token_weight is None:
        return None

    return align_reference_token_weight_to_suffix(
        token_weight,
        target_shape=target_shape[:2],
        reference_prompt_len=_normalize_reference_prompt_len(reference_prompt_len, token_weight.shape[1]),
        student_prompt_len=student_prompt_len,
        dtype=dtype,
    )


def _normalize_reference_prompt_len(reference_prompt_len: int | None, cached_seq_len: int) -> int | None:
    if reference_prompt_len is None:
        return None
    if cached_seq_len <= 0:
        return reference_prompt_len
    if int(reference_prompt_len) >= int(cached_seq_len):
        return 0
    return int(reference_prompt_len)


def _freeze_vision_modules(model) -> None:
    vision_keywords = ("vision", "visual", "image_tower", "vision_tower")
    for name, parameter in model.named_parameters():
        if any(keyword in name.lower() for keyword in vision_keywords):
            parameter.requires_grad = False


def _build_vocab_alignment(config: PipelineConfig) -> VocabAlignment | None:
    try:
        from transformers import AutoProcessor
    except ImportError:
        return None

    try:
        student_processor = AutoProcessor.from_pretrained(
            resolve_model_path(config.student.model_name),
            trust_remote_code=True,
            local_files_only=True,
        )
        teacher_processor = AutoProcessor.from_pretrained(
            resolve_model_path(config.teacher.model_name),
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        return None

    student_tokenizer = getattr(student_processor, "tokenizer", student_processor)
    teacher_tokenizer = getattr(teacher_processor, "tokenizer", teacher_processor)

    if not hasattr(student_tokenizer, "get_vocab") or not hasattr(teacher_tokenizer, "get_vocab"):
        return None

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if set(student_vocab) != set(teacher_vocab):
        return None

    for token, student_id in student_vocab.items():
        if teacher_vocab.get(token) != student_id:
            return None

    if not student_vocab:
        return None

    shared_token_vocab_size = max(student_vocab.values()) + 1
    return VocabAlignment(shared_token_vocab_size=int(shared_token_vocab_size))


def _remap_reference_logits_to_student_vocab(
    reference,
    *,
    reference_vocab: int | None,
    student_vocab_size: int,
    vocab_alignment: VocabAlignment | None,
):
    import torch

    if reference_vocab is None or vocab_alignment is None:
        return None

    shared = int(vocab_alignment.shared_token_vocab_size)
    copy_len = min(shared, int(reference_vocab), int(student_vocab_size), int(reference.shape[-1]))
    if copy_len <= 0:
        return None

    fill_value = torch.finfo(reference.dtype).min
    remapped = torch.full(
        (*reference.shape[:-1], student_vocab_size),
        fill_value,
        device=reference.device,
        dtype=reference.dtype,
    )
    remapped[..., :copy_len] = reference[..., :copy_len]
    return remapped


def _warn_if_switch_logits_missing(config: PipelineConfig, rows: list[dict]) -> None:
    teacher_field = config.distillation.teacher_logits_field
    switch_field = config.distillation.switch_logits_field
    has_teacher = rows and teacher_field in rows[0]
    has_switch = rows and switch_field in rows[0]
    if not has_teacher:
        print(
            f"Warning: Switch-KD method selected but '{teacher_field}' is missing. "
            "DBiLD teacher supervision will be skipped."
        )
    if not has_switch:
        print(
            f"Warning: Switch-KD method selected but '{switch_field}' is missing. "
            "VSD supervision will be skipped. Precompute visual-switch logits or add an online VSD hook."
        )


def _load_training_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    paths = _training_data_paths(config)
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError(
            "No training data files were found. "
            f"Checked: {', '.join(str(path) for path in paths)}"
        )

    rows_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for path in existing_paths:
        for row in read_jsonl(path):
            row_id = str(row["id"])
            if row_id not in rows_by_id:
                rows_by_id[row_id] = dict(row)
                ordered_ids.append(row_id)
            else:
                rows_by_id[row_id].update(row)

    return [rows_by_id[row_id] for row_id in ordered_ids]


def _training_data_paths(config: PipelineConfig) -> list[Path]:
    candidates = [
        resolve_label_path(config.data),
        resolve_teacher_logits_path(config.data),
        resolve_switch_logits_path(config.data),
    ]
    ordered: list[Path] = []
    for path in candidates:
        if path not in ordered:
            ordered.append(path)
    return ordered
