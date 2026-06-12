from __future__ import annotations

import json
from pathlib import Path

from .config_schema import PipelineConfig
from .data_manifest import read_jsonl
from .logits_cache_utils import (
    align_reference_logits,
    align_reference_logits_to_suffix,
    cached_vocab_size,
    materialize_cached_logits,
    vocab_sizes_compatible,
)
from .vlm_batching import (
    build_supervision_mask,
    build_vlm_data_collator,
    encode_vlm_training_sample,
    load_training_image,
)


def train_student(config: PipelineConfig) -> Path:
    rows = read_jsonl(config.data.distill_path)
    config.student.output_dir.mkdir(parents=True, exist_ok=True)
    config.student.adapter_dir.mkdir(parents=True, exist_ok=True)

    if config.student.model_name.startswith("mock-"):
        return _train_mock_student(config, rows)

    return _train_hf_student(config, rows)


def _train_mock_student(config: PipelineConfig, rows: list[dict]) -> Path:
    artifact = {
        "model_name": config.student.model_name,
        "num_training_samples": len(rows),
        "target_field": config.distillation.target_field,
        "distillation_method": config.distillation.method,
        "note": "Mock student artifact. Use configs/hf_vlm.yaml for real training.",
    }
    output_path = config.student.adapter_dir / "mock_adapter.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Mock student artifact written: {output_path}")
    return output_path


def _train_hf_student(config: PipelineConfig, rows: list[dict]) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForVision2Seq,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install transformers, datasets and peft to run real training.") from exc

    processor = AutoProcessor.from_pretrained(config.student.model_name, trust_remote_code=True)
    model = _load_student_model(config)

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

    dataset = Dataset.from_list(rows)
    teacher_field = config.distillation.teacher_logits_field
    switch_field = config.distillation.switch_logits_field

    def tokenize(example: dict) -> dict:
        image = load_training_image(config.data.image_root, example["image"])
        query = example.get("query") or ""
        prompt = config.distillation.prompt_template.format(
            query=query,
            target_label=example.get("target_label", "target object"),
            task=example.get("task", "vqa"),
        )
        target = example[config.distillation.target_field]
        encoded = encode_vlm_training_sample(
            processor,
            image=image,
            prompt=prompt,
            target=target,
            max_length=config.training.max_length,
            mask_prompt_labels=config.training.mask_prompt_labels,
        )
        item = dict(encoded.model_inputs)
        item["prompt_token_len"] = encoded.prompt_token_len
        if teacher_field in example:
            item[teacher_field] = example[teacher_field]
            item[f"{teacher_field}_prompt_len"] = example.get(f"{teacher_field}_prompt_len")
            item[f"{teacher_field}_vocab_size"] = example.get(f"{teacher_field}_vocab_size")
        if switch_field in example:
            item[switch_field] = example[switch_field]
            item[f"{switch_field}_prompt_len"] = example.get(f"{switch_field}_prompt_len")
            item[f"{switch_field}_vocab_size"] = example.get(f"{switch_field}_vocab_size")
        return item

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
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
        "train_dataset": tokenized,
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


def _load_student_model(config: PipelineConfig):
    from transformers import AutoModelForVision2Seq

    model_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
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

    return AutoModelForVision2Seq.from_pretrained(config.student.model_name, **model_kwargs)


def _build_switch_kd_trainer():
    from transformers import Trainer

    from .loss_switch_kd import SwitchKDLoss

    class SwitchKDTrainer(Trainer):
        _vocab_warning_emitted: set[str] = set()

        def __init__(self, *args, switch_kd_config: PipelineConfig, **kwargs):
            super().__init__(*args, **kwargs)
            self.switch_kd_config = switch_kd_config
            distill = switch_kd_config.distillation
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
            )

            supervision_mask = build_supervision_mask(labels)
            loss_output = self.switch_kd_loss(
                student_logits=student_logits,
                labels=labels,
                teacher_logits=teacher_logits,
                switch_logits=switch_logits,
                attention_mask=supervision_mask,
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
):
    if cached is None:
        return None

    if reference_vocab_size_meta is not None:
        reference_vocab = int(reference_vocab_size_meta)
    else:
        reference_vocab = cached_vocab_size(cached)

    if distill.skip_kd_on_vocab_mismatch and not vocab_sizes_compatible(reference_vocab, student_vocab_size):
        key = f"{label}:{reference_vocab}->{student_vocab_size}"
        if key not in warning_bucket:
            print(
                f"Warning: Skipping {label} KD because cached vocab_size={reference_vocab} "
                f"does not match student vocab_size={student_vocab_size}."
            )
            warning_bucket.add(key)
        return None

    tensor = materialize_cached_logits(
        cached,
        device=device,
        dtype=dtype,
        vocab_size=student_vocab_size,
    )
    if distill.align_kd_logits_to_answer:
        return align_reference_logits_to_suffix(
            tensor,
            target_shape=target_shape,
            reference_prompt_len=reference_prompt_len,
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


def _freeze_vision_modules(model) -> None:
    vision_keywords = ("vision", "visual", "image_tower", "vision_tower")
    for name, parameter in model.named_parameters():
        if any(keyword in name.lower() for keyword in vision_keywords):
            parameter.requires_grad = False


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
