from __future__ import annotations

from .stage_teacher_precompute import (
    HuggingFaceTeacher,
    MockTeacher,
    OllamaTeacher,
    OpenAICompatibleTeacher,
    TeacherBackend,
    _canonicalize_teacher_answer,
    _format_prompt,
    _label_sample,
    _load_teacher_image,
    _normalize_teacher_answer,
    _strip_special_tokens,
    build_teacher,
    create_distillation_dataset,
)

__all__ = [
    "HuggingFaceTeacher",
    "MockTeacher",
    "OllamaTeacher",
    "OpenAICompatibleTeacher",
    "TeacherBackend",
    "_canonicalize_teacher_answer",
    "_format_prompt",
    "_label_sample",
    "_load_teacher_image",
    "_normalize_teacher_answer",
    "_strip_special_tokens",
    "build_teacher",
    "create_distillation_dataset",
]
