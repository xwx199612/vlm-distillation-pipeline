from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml

@dataclass
class DataConfig:
    manifest_path: Path
    distill_path: Path
    label_path: Path | None = None
    prediction_path: Path | None = None
    teacher_logits_path: Path | None = None
    switch_logits_path: Path | None = None
    eval_path: Path | None = None
    image_root: Path = Path(".")
    image_dir: Path | None = None
    output_dir: Path | None = None
    max_samples: int | None = None


@dataclass
class TeacherConfig:
    model_name: str
    backend: str = "mock"
    device_map: str | None = None
    attn_implementation: str = "sdpa"
    base_url: str | None = None
    api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    request_timeout: int = 120
    torch_dtype: str | None = None
    quantization: str = "none"
    temperature: float = 0.2
    max_new_tokens: int = 128
    image_resize: str = "original"


@dataclass
class StudentConfig:
    model_name: str
    output_dir: Path
    adapter_dir: Path
    inference_model_path: str | None = None
    attn_implementation: str = "sdpa"
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
    image_resize: str = "original"
    freeze_vision_tower: bool = True
    mask_prompt_labels: bool = True
    quantization: str = "none"

@dataclass
class DistillationConfig:
    confidence_weighting: bool = True
    min_teacher_confidence: float = 0.0
    prompt_template: str = "Query: {query}\nAnswer:"
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
    switch_cache_student_visual: bool = False
    student_visual_cache_dir: Path | None = None
    keep_student_visual_cache_on_cpu: bool = True
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


OUTPUT_ROOT_ENV_VARS = (
    "VLM_DISTILL_OUTPUT_ROOT",
    "CODEX_OUTPUT_ROOT",
)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    raw = _apply_config_options(raw)
    return PipelineConfig(
        seed=raw.get("seed", 42),
        data=_build_data_config(raw["data"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=_build_student_config(raw["student"]),
        training=TrainingConfig(**raw.get("training", {})),
        distillation=_build_distillation_config(raw.get("distillation", {})),
        evaluation=_build_evaluation_config(raw.get("evaluation", {})),
    )


def resolve_output_root() -> Path | None:
    for env_name in OUTPUT_ROOT_ENV_VARS:
        raw = os.environ.get(env_name)
        if not raw:
            raw = _read_windows_user_env(env_name)
        if raw:
            return Path(raw).expanduser()
    return None


def _read_windows_user_env(env_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, env_name)
    except OSError:
        return None

    return value if isinstance(value, str) and value else None


def remap_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    root = resolve_output_root()
    if root is None:
        return path

    parts = path.parts
    if not parts or parts[0] != "outputs":
        return path

    return root.joinpath(*parts[1:])


def remap_output_path_string(value: str | None) -> str | None:
    if not value:
        return value
    return str(remap_output_path(Path(value)))


def _build_data_config(raw: dict[str, Any]) -> DataConfig:
    values = dict(raw)
    for key in (
        "manifest_path",
        "distill_path",
        "label_path",
        "prediction_path",
        "teacher_logits_path",
        "switch_logits_path",
        "eval_path",
        "image_root",
        "image_dir",
        "output_dir",
    ):
        if values.get(key) is not None:
            values[key] = remap_output_path(Path(values[key]))
    return DataConfig(**values)


def _build_student_config(raw: dict[str, Any]) -> StudentConfig:
    values = dict(raw)
    for key in ("output_dir", "adapter_dir"):
        values[key] = remap_output_path(Path(values[key]))
    if values.get("inference_model_path") is not None:
        values["inference_model_path"] = remap_output_path_string(values["inference_model_path"])
    return StudentConfig(**values)


def _build_distillation_config(raw: dict[str, Any]) -> DistillationConfig:
    values = dict(raw)
    legacy_target_field = values.pop("target_field", None)
    if legacy_target_field not in (None, "student_target", "teacher_answer"):
        raise ValueError(
            "distillation.target_field is no longer configurable. "
            "Use teacher_answer as the single training target field."
        )
    for key in (
        "student_vision_path",
        "student_projector_path",
        "teacher_lm_path",
        "teacher_token_embedding_path",
        "teacher_lm_head_path",
    ):
        if values.get(key) is not None:
            values[key] = remap_output_path_string(values[key])
    if values.get("student_visual_cache_dir") is not None:
        values["student_visual_cache_dir"] = remap_output_path(Path(values["student_visual_cache_dir"]))
    return DistillationConfig(**values)


def _build_evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    values = dict(raw)
    if values.get("output_path") is not None:
        values["output_path"] = remap_output_path(Path(values["output_path"]))
    return EvaluationConfig(**values)


def _apply_config_options(raw: dict[str, Any]) -> dict[str, Any]:
    values = dict(raw)
    options_raw = values.pop("options", {})
    if not isinstance(options_raw, dict):
        raise ValueError("options must be a mapping when provided.")

    options: dict[str, str] = {
        key: str(value)
        for key, value in options_raw.items()
        if value is not None
    }
    quality = options.get("quality")
    teacher_quantization = (
        options.get("teacher_quantization")
        or options.get("teacher_label_quantization")
    )
    student_quantization = options.get("student_quantization")
    task_name = options.get("task_name", "parsing")

    if teacher_quantization:
        options.setdefault("teacher_quantization", teacher_quantization)
        options.setdefault("teacher_label_quantization", teacher_quantization)

    if quality and teacher_quantization:
        options.setdefault("label_profile", f"{quality}_{teacher_quantization}")
    if quality and teacher_quantization and student_quantization:
        options.setdefault(
            "response_profile",
            f"{quality}_{teacher_quantization}_student_{student_quantization}",
        )
    elif quality and teacher_quantization:
        options.setdefault("response_profile", f"{quality}_{teacher_quantization}")
    options.setdefault("task_name", task_name)

    return _interpolate_config_values(values, options)


def _interpolate_config_values(value: Any, options: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _interpolate_config_values(nested_value, options)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_interpolate_config_values(item, options) for item in value]
    if isinstance(value, str):
        return _replace_known_placeholders(value, options)
    return value


def _replace_known_placeholders(template: str, options: dict[str, str]) -> str:
    pattern = re.compile(r"\{([A-Za-z0-9_]+)\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return options.get(key, match.group(0))

    return pattern.sub(replace, template)


def build_prompt_context(
    *,
    query: str | None = None,
    target_label: str | None = None,
    target_type: str | None = None,
    task: str | None = None,
) -> dict[str, str]:
    query_text = query or ""
    return {
        "query": query_text,
        "question": query_text,
        "target_label": target_label or "",
        "target_type": target_type or "",
        "task": task or "",
    }


def format_prompt(
    template: str,
    *,
    query: str | None = None,
    target_label: str | None = None,
    target_type: str | None = None,
    task: str | None = None,
) -> str:
    return template.format(
        **build_prompt_context(
            query=query,
            target_label=target_label,
            target_type=target_type,
            task=task,
        )
    )


def resolve_label_path(data: DataConfig) -> Path:
    return data.label_path or data.distill_path


def resolve_prediction_path(data: DataConfig) -> Path:
    return data.prediction_path or data.distill_path


def resolve_teacher_logits_path(data: DataConfig) -> Path:
    return data.teacher_logits_path or data.distill_path


def resolve_switch_logits_path(data: DataConfig) -> Path:
    return data.switch_logits_path or data.distill_path
