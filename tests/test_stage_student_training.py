from __future__ import annotations

import types
from pathlib import Path

import pytest

from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.config_schema import DistillationConfig
from vlm_distill.stage_student_training import _load_student_model, _validate_switch_kd_training_rows


class _DummyModel:
    pass


def _make_config(tmp_path: Path, *, device_map="auto", quantization="none") -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(
            model_name="mock-student",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
            device_map=device_map,
            quantization=quantization,
        ),
    )


def test_load_student_model_omits_device_map_for_ddp(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(model_name_or_path, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return _DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = DummyAutoModel
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")

    model, resolved_device_map = _load_student_model(_make_config(tmp_path, device_map=None), "student-model")

    assert isinstance(model, _DummyModel)
    assert resolved_device_map is None
    assert captured["model_name_or_path"] == "student-model"
    assert "device_map" not in captured["kwargs"]
    assert captured["kwargs"]["trust_remote_code"] is True
    assert captured["kwargs"]["local_files_only"] is True


def test_load_student_model_passes_resolved_device_map_without_ddp(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(model_name_or_path, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return _DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = DummyAutoModel
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    model, resolved_device_map = _load_student_model(_make_config(tmp_path, device_map=None), "student-model")

    assert isinstance(model, _DummyModel)
    assert resolved_device_map == "auto"
    assert captured["kwargs"]["device_map"] == "auto"


def _switch_kd_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            switch_logits_path=tmp_path / "switch_logits.jsonl",
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd", align_kd_logits_to_answer=True),
    )


def _valid_logits(seq_len: int, vocab_size: int = 8) -> dict:
    return {
        "indices": [[[0] for _ in range(seq_len)]],
        "values": [[[1.0] for _ in range(seq_len)]],
        "shape": [1, seq_len, vocab_size],
        "vocab_size": vocab_size,
    }


def test_switch_kd_training_raises_if_teacher_logits_rows_zero(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "switch_logits": _valid_logits(4),
            "switch_logits_prompt_len": 0,
            "switch_logits_aligned_to_answer": True,
        }
    ]

    with pytest.raises(RuntimeError, match="rows_with_teacher_logits=0"):
        _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_switch_kd_training_raises_if_switch_logits_rows_zero(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "teacher_logits": _valid_logits(2),
            "teacher_logits_prompt_len": 0,
            "teacher_logits_aligned_to_answer": True,
        }
    ]

    with pytest.raises(RuntimeError, match="rows_with_switch_logits=0"):
        _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_switch_kd_training_raises_if_prompt_len_alignment_invalid(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "teacher_logits": _valid_logits(2),
            "teacher_logits_prompt_len": 0,
            "teacher_logits_aligned_to_answer": True,
            "switch_logits": _valid_logits(10),
            "switch_logits_prompt_len": 3,
        }
    ]

    with pytest.raises(ValueError, match="not marked as answer-only"):
        _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_switch_kd_training_valid_logits_pass_dataset_validation(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2, 3],
            "teacher_logits": _valid_logits(3),
            "teacher_logits_prompt_len": 0,
            "teacher_logits_aligned_to_answer": True,
            "switch_logits": _valid_logits(3),
            "switch_logits_prompt_len": 0,
            "switch_logits_aligned_to_answer": True,
        }
    ]

    _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)
