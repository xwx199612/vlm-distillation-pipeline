from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_schema import (
    PipelineConfig,
    format_prompt,
    resolve_label_path,
    resolve_switch_logits_path,
)
from .data_manifest import read_jsonl
from .device_utils import (
    ensure_stage_uses_cuda,
    get_module_by_path,
    print_stage_model_debug,
    resolve_training_device_map,
    select_model_input_device,
)
from .logits_cache_utils import (
    align_compact_reference_to_suffix,
    align_reference_logits,
    align_reference_logits_to_suffix,
    cached_vocab_size,
    compact_logits_to_tensors,
    materialize_cached_logits,
    remap_compact_reference_to_student_vocab,
    vocab_sizes_compatible,
)
from .model_loading import apply_attn_implementation, resolve_model_path
from .token_alignment import build_token_mismatch_details, coerce_token_ids


class VlmTrainingDataset:
    """Tokenize multimodal samples lazily to avoid Arrow overflows on image tensors."""

    def __init__(self, rows: list[dict[str, Any]], config: PipelineConfig, processor):
        self.rows = rows
        self.config = config
        self.processor = processor
        self._token_identity_debug_printed = False

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
            canonical_answer_span=True,
        )
        item = dict(encoded.model_inputs)
        _validate_student_supervised_labels_against_teacher_tokens(
            example=example,
            labels=item["labels"],
            config=self.config,
        )
        if not self._token_identity_debug_printed:
            teacher_field = self.config.distillation.teacher_logits_field
            switch_field = self.config.distillation.switch_logits_field
            supervised_label_ids = [int(token_id) for token_id in item["labels"][item["labels"] != -100].tolist()]
            teacher_tokens = _extract_teacher_tokens(example)
            print("Switch-KD first sample label debug:")
            print(f"  prompt_token_len: {encoded.prompt_token_len}")
            print(f"  teacher_tokens_len: {len(_extract_teacher_tokens(example))}")
            print(f"  teacher_logits_answer_token_ids_len: {len(coerce_token_ids(example.get(f'{teacher_field}_answer_token_ids')))}")
            print(f"  switch_logits_answer_token_ids_len: {len(coerce_token_ids(example.get(f'{switch_field}_answer_token_ids')))}")
            print(f"  student_supervised_label_ids_len: {len(supervised_label_ids)}")
            print(f"  first_5_teacher_tokens: {teacher_tokens[:5]}")
            print(f"  first_5_student_labels: {supervised_label_ids[:5]}")
            print(f"  token_identity_validation_passed: {supervised_label_ids == teacher_tokens}")
            self._token_identity_debug_printed = True
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


def _model_name_family(model_name: str | None) -> str:
    return (model_name or "").lower().replace("_", "").replace("-", "").replace(".", "")


def _student_label_alignment_required(config: PipelineConfig) -> bool:
    teacher_family = _model_name_family(config.teacher.model_name)
    student_family = _model_name_family(config.student.model_name)
    return "qwen25vl" in teacher_family and "qwen25vl" in student_family


def _validate_token_identity_metadata(
    row: dict[str, Any],
    *,
    field_name: str,
    label: str,
    required: bool,
) -> None:
    payload = row.get(field_name)
    if payload is None:
        if required:
            raise RuntimeError(f"{label} missing for id={row.get('id')}.")
        return
    teacher_tokens = _extract_teacher_tokens(row)
    if row.get(f"{field_name}_token_identity_match") is not True:
        raise ValueError(f"{label} token identity validation missing or false for id={row.get('id')}.")
    answer_token_ids = row.get(f"{field_name}_answer_token_ids")
    if answer_token_ids is None:
        raise ValueError(f"{label}_answer_token_ids missing for id={row.get('id')}.")
    answer_token_ids = coerce_token_ids(answer_token_ids)
    if answer_token_ids != teacher_tokens:
        raise ValueError(
            f"{label} token identity mismatch for id={row.get('id')}: "
            f"{build_token_mismatch_details(expected=teacher_tokens, actual=answer_token_ids, actual_field_name='actual_answer_token_id')}"
        )


def _validate_student_supervised_labels_against_teacher_tokens(
    *,
    example: dict[str, Any],
    labels,
    config: PipelineConfig,
) -> None:
    if not _student_label_alignment_required(config):
        return
    teacher_tokens = _extract_teacher_tokens(example)
    if not teacher_tokens:
        return
    supervised_label_ids = [int(token_id) for token_id in labels[labels != -100].tolist()]
    if supervised_label_ids != teacher_tokens:
        raise ValueError(
            f"Student label token identity mismatch. id={example.get('id')}, "
            f"{build_token_mismatch_details(expected=teacher_tokens, actual=supervised_label_ids, actual_field_name='actual_student_label_id')}"
        )


def train_student(config: PipelineConfig) -> Path:
    rows = _load_training_rows(config)
    _print_training_row_summary(config, rows)
    if config.distillation.method == "switch_kd":
        if config.training.freeze_vision_tower:
            print(
                "WARNING: freeze_vision_tower=true limits VSD's ability to improve the student visual encoder. "
                "This is an offline/static VSD baseline."
            )
        _validate_switch_kd_training_rows(config, rows)
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
    model, resolved_device_map = _load_student_model(config, student_model_path)
    selected_input_device = select_model_input_device(
        model,
        preferred_modules=(
            get_module_by_path(model, "model.visual"),
            get_module_by_path(model, "visual"),
            get_module_by_path(model, "model.language_model.embed_tokens"),
            get_module_by_path(model, "model.language_model"),
        ),
        label="Train",
    )
    print_stage_model_debug(
        stage_label="Train",
        model_path=student_model_path,
        quantization_mode=config.student.quantization,
        requested_device_map=resolved_device_map,
        model=model,
        selected_input_device=selected_input_device,
    )
    ensure_stage_uses_cuda(
        stage_label="Train",
        requested_device_map=resolved_device_map,
        model=model,
        selected_input_device=selected_input_device,
        allow_distributed_none=True,
    )

    if config.student.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)

    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

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
    data_collator = build_vlm_data_collator(
        processor,
        logits_fields=(
            config.distillation.teacher_logits_field,
            config.distillation.switch_logits_field,
        ),
    )
    args = TrainingArguments(
        output_dir=str(config.student.output_dir),
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        ddp_find_unused_parameters=config.training.ddp_find_unused_parameters,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        max_steps=config.training.max_steps or -1,
        logging_steps=config.training.log_every,
        save_steps=config.training.save_every,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.mixed_precision == "fp16",
        bf16=config.training.mixed_precision == "bf16",
        remove_unused_columns=False,
        **_build_gradient_checkpointing_kwargs(config.training.gradient_checkpointing, TrainingArguments),
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
        trainer_kwargs["switch_kd_config"] = config
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(config.student.adapter_dir)
    processor.save_pretrained(config.student.adapter_dir)
    return config.student.adapter_dir


def _build_gradient_checkpointing_kwargs(
    enabled: bool,
    training_arguments_cls,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"gradient_checkpointing": enabled}
    if not enabled:
        return kwargs

    try:
        signature = inspect.signature(training_arguments_cls.__init__)
    except (TypeError, ValueError):
        return kwargs

    if "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return kwargs


def _load_student_model(config: PipelineConfig, model_path: str | None = None):
    import torch

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover - fallback for older transformers
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    resolved_device_map = resolve_training_device_map(
        config.student.device_map,
        quantization=config.student.quantization,
        role="student",
        allow_accelerate_ddp=True,
    )
    model_kwargs: dict = {
        "trust_remote_code": True,
    }
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    apply_attn_implementation(model_kwargs, config.student.attn_implementation)

    if config.student.quantization == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config.student.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.student.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs["local_files_only"] = True
    model_name_or_path = model_path or resolve_model_path(config.student.model_name)
    model = AutoModelForVLM.from_pretrained(model_name_or_path, **model_kwargs)
    return model, resolved_device_map


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
                top_k_mode=distill.dbild_top_k_mode,
                kneedle_candidate_k=distill.dbild_kneedle_candidate_k,
                min_top_k=distill.dbild_min_top_k,
                max_top_k=distill.dbild_max_top_k,
                kl_mode=distill.dbild_kl_mode,
                min_prob=distill.dbild_min_prob,
            )
            self._switch_kd_last_losses: dict[str, float] | None = None
            self._switch_kd_last_logged_step = -1

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
            self._switch_kd_last_losses = {
                "total_loss": float(loss_output.loss.detach().float().item()),
                "lm_loss": float(loss_output.lm_loss.detach().float().item()),
                "dbild_loss": float(loss_output.dbild_loss.detach().float().item()),
                "vsd_loss": float(loss_output.vsd_loss.detach().float().item()),
            }
            return (loss_output.loss, outputs) if return_outputs else loss_output.loss

        def log(self, logs, *args, **kwargs):
            super().log(logs, *args, **kwargs)
            self._maybe_log_switch_kd_progress()

        def _maybe_log_switch_kd_progress(self) -> None:
            import torch

            if self._switch_kd_last_losses is None:
                return
            global_step = int(self.state.global_step)
            if global_step <= 0 or global_step == self._switch_kd_last_logged_step:
                return
            if self.switch_kd_config.training.log_every <= 0:
                return
            if global_step % self.switch_kd_config.training.log_every != 0:
                return

            max_steps = int(self.state.max_steps or self.args.max_steps or 0)
            lr = 0.0
            if self.optimizer is not None and self.optimizer.param_groups:
                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            gpu_mem_allocated = 0
            gpu_mem_reserved = 0
            if torch.cuda.is_available():
                gpu_mem_allocated = int(torch.cuda.memory_allocated())
                gpu_mem_reserved = int(torch.cuda.memory_reserved())

            loss_values = self._switch_kd_last_losses
            print(
                f"[train] step={global_step}/{max_steps} "
                f"total_loss={loss_values['total_loss']:.6f} "
                f"lm_loss={loss_values['lm_loss']:.6f} "
                f"dbild_loss={loss_values['dbild_loss']:.6f} "
                f"vsd_loss={loss_values['vsd_loss']:.6f} "
                f"lr={lr:.8g} "
                f"gpu_mem_allocated={gpu_mem_allocated} "
                f"gpu_mem_reserved={gpu_mem_reserved}"
            )
            self._switch_kd_last_logged_step = global_step

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

    compact = compact_logits_to_tensors(cached, device=device, dtype=dtype)
    if compact is not None:
        effective_reference_prompt_len = _normalize_reference_prompt_len(reference_prompt_len, compact["indices"].shape[1])
        if not vocab_sizes_compatible(reference_vocab, student_vocab_size):
            remapped = remap_compact_reference_to_student_vocab(
                compact,
                reference_vocab=reference_vocab,
                student_vocab_size=student_vocab_size,
                shared_token_vocab_size=(
                    vocab_alignment.shared_token_vocab_size if vocab_alignment is not None else None
                ),
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
                raise ValueError(
                    f"Cannot use {label} KD because cached vocab_size={reference_vocab} "
                    f"does not match student vocab_size={student_vocab_size} and "
                    "skip_kd_on_vocab_mismatch=false."
                )
            else:
                key = f"{label}:{reference_vocab}->{student_vocab_size}:remapped"
                if key not in warning_bucket:
                    print(
                        f"Info: Remapped {label} KD logits from vocab_size={reference_vocab} "
                        f"to student vocab_size={student_vocab_size} using shared_token_vocab_size="
                        f"{vocab_alignment.shared_token_vocab_size if vocab_alignment else 'unknown'}."
                    )
                    warning_bucket.add(key)
                compact = remapped
        elif compact["vocab_size"] != student_vocab_size:
            compact["shape"] = (*compact["shape"][:-1], student_vocab_size)
            compact["vocab_size"] = int(student_vocab_size)

        token_weight = _extract_compact_reference_token_weight(cached, device=device, dtype=dtype)
        if distill.align_kd_logits_to_answer:
            aligned = align_compact_reference_to_suffix(
                compact,
                target_shape=target_shape,
                reference_prompt_len=effective_reference_prompt_len,
                student_prompt_len=student_prompt_len,
                dtype=dtype,
            )
        else:
            aligned = align_compact_reference_to_suffix(
                compact,
                target_shape=target_shape,
                reference_prompt_len=None,
                student_prompt_len=None,
                dtype=dtype,
            )

        return _finalize_compact_reference(
            aligned,
            token_weight=token_weight,
            student_prompt_len=student_prompt_len,
            reference_prompt_len=effective_reference_prompt_len if distill.align_kd_logits_to_answer else None,
        )

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
            raise ValueError(
                f"Cannot use {label} KD because cached vocab_size={reference_vocab} "
                f"does not match student vocab_size={student_vocab_size} and "
                "skip_kd_on_vocab_mismatch=false."
            )
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


def _extract_compact_reference_token_weight(cached, *, device, dtype):
    from .logits_cache_utils import cached_token_weight

    return cached_token_weight(cached, device=device, dtype=dtype)


def _finalize_compact_reference(
    reference: dict[str, Any],
    *,
    token_weight,
    student_prompt_len: int | None,
    reference_prompt_len: int | None,
):
    compact = dict(reference)
    compact["logits"] = compact["values"]
    if token_weight is not None:
        compact["token_weight"] = token_weight
        compact["entropy_weight"] = token_weight
    compact["student_prompt_len"] = student_prompt_len
    compact["reference_prompt_len"] = reference_prompt_len
    compact["is_compact"] = True
    return compact


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
    total_rows = len(rows)
    teacher_rows = sum(1 for row in rows if row.get(teacher_field) is not None)
    switch_rows = sum(1 for row in rows if row.get(switch_field) is not None)

    if total_rows == 0:
        raise RuntimeError("Switch-KD method selected but no training rows were loaded.")
    if teacher_rows == 0:
        raise RuntimeError(
            f"Switch-KD method selected but all rows are missing '{teacher_field}'. "
            "DBiLD teacher supervision would be fully disabled."
        )
    if switch_rows == 0:
        raise RuntimeError(
            f"Switch-KD method selected but all rows are missing '{switch_field}'. "
            "VSD supervision would be fully disabled. Precompute visual-switch logits or add an online VSD hook."
        )
    if teacher_rows < total_rows:
        print(
            f"Warning: Switch-KD rows missing '{teacher_field}': "
            f"{total_rows - teacher_rows}/{total_rows}. DBiLD teacher supervision will be skipped for those rows."
        )
    if switch_rows < total_rows:
        print(
            f"Warning: Switch-KD rows missing '{switch_field}': "
            f"{total_rows - switch_rows}/{total_rows}. VSD supervision will be skipped for those rows."
        )


def _validate_switch_kd_training_rows(config: PipelineConfig, rows: list[dict]) -> None:
    teacher_field = config.distillation.teacher_logits_field
    switch_field = config.distillation.switch_logits_field
    rows_with_teacher_answer = sum(1 for row in rows if row.get("teacher_answer") is not None)
    rows_with_teacher_tokens = sum(1 for row in rows if _extract_teacher_tokens(row))
    rows_with_teacher_logits = sum(1 for row in rows if row.get(teacher_field) is not None)
    rows_with_switch_logits = sum(1 for row in rows if row.get(switch_field) is not None)

    if rows_with_teacher_answer <= 0:
        raise RuntimeError("Switch-KD method selected but rows_with_teacher_answer=0.")
    if rows_with_teacher_tokens <= 0:
        raise RuntimeError("Switch-KD method selected but rows_with_teacher_tokens=0.")
    if config.distillation.teacher_logits and rows_with_teacher_logits <= 0:
        raise RuntimeError(f"Switch-KD method selected but rows_with_{teacher_field}=0.")
    if rows_with_switch_logits <= 0:
        raise RuntimeError(f"Switch-KD method selected but rows_with_{switch_field}=0.")

    _warn_if_switch_logits_missing(config, rows)

    for row_index, row in enumerate(rows):
        _validate_token_identity_metadata(
            row,
            field_name=teacher_field,
            label="teacher_logits",
            required=bool(config.distillation.teacher_logits),
        )
        _validate_token_identity_metadata(
            row,
            field_name=switch_field,
            label="switch_logits",
            required=True,
        )
        if row_index == 0:
            print("Switch-KD token identity debug:")
            print(f"  teacher_tokens_len: {len(_extract_teacher_tokens(row))}")
            print(f"  teacher_logits_answer_token_ids_len: {len(coerce_token_ids(row.get(f'{teacher_field}_answer_token_ids')))}")
            print(f"  switch_logits_answer_token_ids_len: {len(coerce_token_ids(row.get(f'{switch_field}_answer_token_ids')))}")
            print("  student_supervised_label_ids_len: pending")
            print("  token_identity_validation_passed: True")
        _validate_cached_logits_alignment(
            row,
            field_name=teacher_field,
            align_to_answer=config.distillation.align_kd_logits_to_answer,
            label="teacher_logits",
            required=bool(config.distillation.teacher_logits),
        )
        _validate_cached_logits_alignment(
            row,
            field_name=switch_field,
            align_to_answer=config.distillation.align_kd_logits_to_answer,
            label="switch_logits",
            required=True,
        )


def _validate_cached_logits_alignment(
    row: dict[str, Any],
    *,
    field_name: str,
    align_to_answer: bool,
    label: str,
    required: bool = False,
) -> None:
    payload = row.get(field_name)
    if payload is None:
        if required:
            raise RuntimeError(f"{label} missing for id={row.get('id')}.")
        return
    raw_seq_len = _cached_logits_seq_len(payload)
    if raw_seq_len is None:
        return
    prompt_len_value = row.get(f"{field_name}_prompt_len")
    prompt_len = int(prompt_len_value) if prompt_len_value is not None else 0
    aligned_to_answer = row.get(f"{field_name}_aligned_to_answer") is True
    effective_len = raw_seq_len if aligned_to_answer else raw_seq_len - (_normalize_reference_prompt_len(prompt_len, raw_seq_len) or 0)
    teacher_tokens_len = len(_extract_teacher_tokens(row))
    answer_label_token_count = teacher_tokens_len or None
    print(
        f"[train][{label}] id={row.get('id')} raw_logits_seq_len={raw_seq_len} "
        f"prompt_len={prompt_len} answer_only={aligned_to_answer} effective_logits_seq_len={effective_len} "
        f"teacher_tokens_len={teacher_tokens_len} answer_label_token_count={answer_label_token_count}"
    )
    if align_to_answer and teacher_tokens_len > 0 and not aligned_to_answer:
        raise ValueError(
            f"{label} is not marked as answer-only for id={row.get('id')}. "
            "Old full-sequence logits are not supported by this training path; regenerate precompute outputs."
        )
    if align_to_answer and teacher_tokens_len > 0 and effective_len != teacher_tokens_len:
        raise ValueError(
            f"{label} answer-only alignment is invalid for id={row.get('id')}: "
            f"raw_logits_seq_len={raw_seq_len}, prompt_len={prompt_len}, "
            f"effective_logits_seq_len={effective_len}, teacher_tokens_len={teacher_tokens_len}, "
            f"difference={effective_len - teacher_tokens_len}."
        )


def _cached_logits_seq_len(payload: Any) -> int | None:
    if isinstance(payload, dict):
        shape = payload.get("shape")
        if isinstance(shape, list | tuple) and len(shape) >= 2:
            return int(shape[1])
        indices = payload.get("indices")
        if isinstance(indices, list) and indices and isinstance(indices[0], list):
            return len(indices[0])
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return len(payload[0])
    return None


def _extract_teacher_tokens(row: dict[str, Any]) -> list[int]:
    tokens = row.get("teacher_tokens")
    if isinstance(tokens, list) and (not tokens or not isinstance(tokens[0], list)):
        return [int(value) for value in tokens]
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        return [int(value) for value in tokens[0]]
    generated = row.get("teacher_generated_ids")
    if isinstance(generated, list) and generated and isinstance(generated[0], list):
        return [int(value) for value in generated[0]]
    if isinstance(generated, list):
        return [int(value) for value in generated]
    return []


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


def _print_training_row_summary(config: PipelineConfig, rows: list[dict[str, Any]]) -> None:
    teacher_field = config.distillation.teacher_logits_field
    switch_field = config.distillation.switch_logits_field
    teacher_answer_rows = sum(1 for row in rows if row.get("teacher_answer") is not None)
    teacher_logits_rows = sum(1 for row in rows if row.get(teacher_field) is not None)
    switch_logits_rows = sum(1 for row in rows if row.get(switch_field) is not None)
    teacher_vocab_sizes = sorted(
        {
            int(value)
            for row in rows
            if (value := row.get(f"{teacher_field}_vocab_size")) is not None
        }
    )
    switch_vocab_sizes = sorted(
        {
            int(value)
            for row in rows
            if (value := row.get(f"{switch_field}_vocab_size")) is not None
        }
    )

    paths = _training_data_paths(config)
    print("Student training data summary:")
    for path in paths:
        exists = path.exists()
        path_rows = read_jsonl(path) if exists else []
        first_keys = sorted(path_rows[0].keys()) if path_rows else None
        label = "label_path"
        if path == resolve_switch_logits_path(config.data):
            label = "switch_logits_path"
        print(
            f"  {label}: {path} exists={exists} row_count={len(path_rows)} "
            f"first_row_keys={first_keys}"
        )
    print(f"  total rows: {len(rows)}")
    print(f"  rows with teacher_answer: {teacher_answer_rows}")
    print(f"  rows with {teacher_field}: {teacher_logits_rows}")
    print(f"  rows with {switch_field}: {switch_logits_rows}")
    print(
        "  teacher_logits vocab sizes:",
        teacher_vocab_sizes if teacher_vocab_sizes else "none",
    )
    print(
        "  switch_logits vocab sizes:",
        switch_vocab_sizes if switch_vocab_sizes else "none",
    )


def _training_data_paths(config: PipelineConfig) -> list[Path]:
    candidates = [
        resolve_label_path(config.data),
        resolve_switch_logits_path(config.data),
    ]
    ordered: list[Path] = []
    for path in candidates:
        if path not in ordered:
            ordered.append(path)
    return ordered
