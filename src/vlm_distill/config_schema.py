from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest_path: Path
    distill_path: Path
    eval_path: Path | None = None
    image_root: Path = Path(".")
    max_samples: int | None = None


@dataclass
class TeacherConfig:
    model_name: str
    backend: str = "mock"
    device_map: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    request_timeout: int = 120
    torch_dtype: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 128


@dataclass
class StudentConfig:
    model_name: str
    output_dir: Path
    adapter_dir: Path
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    quantization: str = "none"


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_steps: int | None = None
    log_every: int = 10
    save_every: int = 500
    mixed_precision: str = "no"
    gradient_checkpointing: bool = True
    max_length: int = 512
    freeze_vision_tower: bool = True
    mask_prompt_labels: bool = True
    quantization: str = "none"

@dataclass
class DistillationConfig:
    target_field: str = "student_target"
    confidence_weighting: bool = True
    min_teacher_confidence: float = 0.0
    prompt_template: str = "Question: {question}\nAnswer:"
    method: str = "sft"
    lm_loss_weight: float = 1.0
    dbild_loss_weight: float = 0.5
    vsd_loss_weight: float = 0.5
    kd_temperature: float = 2.0
    dbild_top_k: int = 64
    dbild_min_prob: float = 0.0
    teacher_logits_field: str = "teacher_logits"
    switch_logits_field: str = "switch_logits"
    use_cached_logits: bool = True
    student_vision_path: str | None = None
    student_projector_path: str | None = None
    teacher_lm_path: str | None = None
    teacher_token_embedding_path: str | None = None
    teacher_lm_head_path: str | None = None
    visual_token_placeholder: str = "<image>"
    max_cached_logits_vocab: int | None = 4096
    align_kd_logits_to_answer: bool = True
    skip_kd_on_vocab_mismatch: bool = True


@dataclass
class EvaluationConfig:
    output_path: Path = Path("outputs/eval_report.json")
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "token_f1"])


@dataclass
class PipelineConfig:
    data: DataConfig
    teacher: TeacherConfig
    student: StudentConfig
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return PipelineConfig(
        seed=raw.get("seed", 42),
        data=_build_data_config(raw["data"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=_build_student_config(raw["student"]),
        training=TrainingConfig(**raw.get("training", {})),
        distillation=DistillationConfig(**raw.get("distillation", {})),
        evaluation=_build_evaluation_config(raw.get("evaluation", {})),
    )


def _build_data_config(raw: dict[str, Any]) -> DataConfig:
    values = dict(raw)
    for key in ("manifest_path", "distill_path", "eval_path", "image_root"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return DataConfig(**values)


def _build_student_config(raw: dict[str, Any]) -> StudentConfig:
    values = dict(raw)
    for key in ("output_dir", "adapter_dir"):
        values[key] = Path(values[key])
    return StudentConfig(**values)


def _build_evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    values = dict(raw)
    if values.get("output_path") is not None:
        values["output_path"] = Path(values["output_path"])
    return EvaluationConfig(**values)
