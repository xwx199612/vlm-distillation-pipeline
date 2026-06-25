from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.config_schema import DistillationConfig
from vlm_distill.stage_student_training import (
    VlmTrainingDataset,
    _load_student_model,
    _training_data_paths,
    _validate_switch_kd_training_rows,
)
from vlm_distill.vlm_batching import EncodedVlmSample


class _DummyModel:
    pass


class _CanonicalSpanProcessor:
    def __init__(self):
        self.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()
        self._token_map = {
            "<chat>prompt</chat>": [101, 102],
            "<chat>prompt answer</chat>": [101, 102, 5890, 7000],
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return f"<chat>{messages[0]['content'][1]['text']}</chat>"

    def __call__(self, images=None, text="", return_tensors="pt", truncation=True, max_length=128):
        del images, return_tensors, truncation, max_length
        if isinstance(text, list):
            text = text[0]
        token_ids = self._token_map[text]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 4, 4),
        }


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


def _valid_identity_metadata(prefix: str, tokens: list[int], vocab_size: int = 8) -> dict:
    return {
        f"{prefix}_prompt_len": 0,
        f"{prefix}_vocab_size": vocab_size,
        f"{prefix}_aligned_to_answer": True,
        f"{prefix}_token_identity_match": True,
        f"{prefix}_answer_token_ids": tokens,
    }


def test_switch_kd_training_raises_if_teacher_logits_rows_zero(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "switch_logits": _valid_logits(4),
            **_valid_identity_metadata("switch_logits", [1, 2]),
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
            **_valid_identity_metadata("teacher_logits", [1, 2]),
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
            **_valid_identity_metadata("teacher_logits", [1, 2]),
            "switch_logits": _valid_logits(10),
            "switch_logits_prompt_len": 3,
            "switch_logits_vocab_size": 8,
            "switch_logits_token_identity_match": True,
            "switch_logits_answer_token_ids": [1, 2],
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
            **_valid_identity_metadata("teacher_logits", [1, 2, 3]),
            "switch_logits": _valid_logits(3),
            **_valid_identity_metadata("switch_logits", [1, 2, 3]),
        }
    ]

    _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_switch_kd_training_fails_when_teacher_answer_token_ids_do_not_match(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "teacher_logits": _valid_logits(2),
            **_valid_identity_metadata("teacher_logits", [1, 9]),
            "switch_logits": _valid_logits(2),
            **_valid_identity_metadata("switch_logits", [1, 2]),
        }
    ]

    with pytest.raises(ValueError, match="teacher_logits token identity mismatch"):
        _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_switch_kd_training_fails_when_switch_answer_token_ids_do_not_match(tmp_path: Path):
    rows = [
        {
            "id": "sample-1",
            "teacher_answer": "answer",
            "teacher_tokens": [1, 2],
            "teacher_logits": _valid_logits(2),
            **_valid_identity_metadata("teacher_logits", [1, 2]),
            "switch_logits": _valid_logits(2),
            **_valid_identity_metadata("switch_logits", [1, 9]),
        }
    ]

    with pytest.raises(ValueError, match="switch_logits token identity mismatch"):
        _validate_switch_kd_training_rows(_switch_kd_config(tmp_path), rows)


def test_student_label_validation_fails_when_supervised_labels_differ(monkeypatch, tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="Qwen2.5-VL-7B-Instruct"),
        student=StudentConfig(
            model_name="Qwen2.5-VL-3B-Instruct",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
        ),
        distillation=DistillationConfig(method="switch_kd"),
    )
    rows = [
        {
            "id": "sample-1",
            "image": "screen.png",
            "task": "parsing",
            "query": "hello",
            "teacher_answer": "{}",
            "teacher_tokens": [10, 11],
            "teacher_logits": _valid_logits(2),
            **_valid_identity_metadata("teacher_logits", [10, 11]),
            "switch_logits": _valid_logits(2),
            **_valid_identity_metadata("switch_logits", [10, 11]),
        }
    ]

    monkeypatch.setattr("vlm_distill.vlm_batching.load_training_image", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "vlm_distill.vlm_batching.encode_vlm_training_sample",
        lambda *args, **kwargs: EncodedVlmSample(
            model_inputs={
                "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 10, 99], dtype=torch.long),
            },
            prompt_token_len=2,
        ),
    )

    dataset = VlmTrainingDataset(rows, config, processor=object())

    with pytest.raises(ValueError, match="Student label token identity mismatch"):
        _ = dataset[0]


def test_student_dataset_accepts_canonical_teacher_answer_span(monkeypatch, tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="Qwen2.5-VL-7B-Instruct"),
        student=StudentConfig(
            model_name="Qwen2.5-VL-3B-Instruct",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
        ),
        distillation=DistillationConfig(method="switch_kd"),
    )
    rows = [
        {
            "id": "sample-1",
            "image": "screen.png",
            "task": "parsing",
            "query": "hello",
            "teacher_answer": "answer",
            "teacher_tokens": [5890, 7000],
            "teacher_logits": _valid_logits(2),
            **_valid_identity_metadata("teacher_logits", [5890, 7000]),
            "switch_logits": _valid_logits(2),
            **_valid_identity_metadata("switch_logits", [5890, 7000]),
        }
    ]

    monkeypatch.setattr("vlm_distill.vlm_batching.load_training_image", lambda *args, **kwargs: object())
    monkeypatch.setattr("vlm_distill.stage_student_training.format_prompt", lambda *args, **kwargs: "prompt")

    dataset = VlmTrainingDataset(rows, config, processor=_CanonicalSpanProcessor())
    item = dataset[0]

    assert item["labels"].tolist() == [-100, -100, 5890, 7000]
    assert item["prompt_token_len"] == 2


def test_switch_kd_training_paths_ignore_legacy_teacher_logits_path(tmp_path: Path):
    config = _switch_kd_config(tmp_path)
    config.data.label_path = tmp_path / "labels.jsonl"
    config.data.teacher_logits_path = tmp_path / "legacy_teacher_logits.jsonl"

    assert _training_data_paths(config) == [
        tmp_path / "labels.jsonl",
        tmp_path / "switch_logits.jsonl",
    ]
