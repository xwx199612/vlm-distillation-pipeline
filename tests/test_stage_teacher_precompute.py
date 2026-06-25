from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.data_manifest import VlmSample
import vlm_distill.stage_teacher_precompute as stage_teacher_precompute
from vlm_distill.stage_teacher_precompute import TeacherLogitsGenerator, compute_teacher_forced_answer_logits


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="hf", device_map="cuda:0"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )


class _FakeProcessor:
    def __init__(self, token_map: dict[str, list[int]]):
        self._token_map = token_map

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, return_tensors="pt"):
        del images, return_tensors
        token_ids = self._token_map[text[0]]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}


class _FakeModel(torch.nn.Module):
    def __init__(self, seq_len: int, vocab_size: int = 8):
        super().__init__()
        self.device = torch.device("cpu")
        self._seq_len = seq_len
        self._vocab_size = vocab_size

    def forward(self, **kwargs):
        del kwargs
        return SimpleNamespace(logits=torch.zeros(1, self._seq_len, self._vocab_size))


def test_teacher_forcing_uses_full_input_answer_span_as_canonical_teacher_tokens(tmp_path: Path):
    config = _make_config(tmp_path)
    prompt = "prompt"
    teacher_answer = "answer"
    processor = _FakeProcessor(
        {
            prompt: [101, 102],
            f"{prompt} {teacher_answer}": [101, 102, 4913, 5890],
        }
    )
    payload = compute_teacher_forced_answer_logits(
        sample_id="parsing-000001",
        image=object(),
        prompt=prompt,
        teacher_answer=teacher_answer,
        teacher_tokens=[5890, 9999],
        model=_FakeModel(seq_len=4),
        processor=processor,
        config=config,
    )

    assert payload["teacher_tokens"] == [4913, 5890]
    assert payload["teacher_logits_answer_token_ids"] == [4913, 5890]
    assert payload["teacher_logits_token_identity_match"] is True
    assert payload["teacher_logits"]["shape"][1] == 2


def test_teacher_logits_generator_fallback_overwrites_teacher_tokens_with_canonical_answer_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    generator = TeacherLogitsGenerator(config)
    prompt = "prompt"
    teacher_answer = "answer"
    processor = _FakeProcessor(
        {
            prompt: [11, 12],
            f"{prompt} {teacher_answer}": [11, 12, 55, 66],
        }
    )

    def fake_load():
        generator._processor = processor
        generator._model = _FakeModel(seq_len=4)
        generator._input_device = torch.device("cpu")

    monkeypatch.setattr(generator, "load", fake_load)
    monkeypatch.setattr(stage_teacher_precompute, "_load_teacher_image", lambda *args, **kwargs: object())
    monkeypatch.setattr(stage_teacher_precompute, "_format_prompt", lambda _config, _sample: prompt)
    monkeypatch.setattr(
        stage_teacher_precompute,
        "_extract_generated_ids_and_scores",
        lambda generation, prompt_len, include_scores: (torch.tensor([[77, 88]], dtype=torch.long), [torch.zeros(8), torch.zeros(8)]),
    )
    monkeypatch.setattr(generator, "_generate", lambda inputs, include_scores: object())
    monkeypatch.setattr(generator, "_decode", lambda generated_ids: teacher_answer)
    monkeypatch.setattr(generator, "tokenize_teacher_answer", lambda answer: [999, 888])

    row = generator.generate_for_sample(
        VlmSample(id="sample-1", image="screen.png", task="vqa", query="hello"),
        mode="switch_kd",
    )

    assert row["teacher_logits_source"] == "teacher_forcing_forward"
    assert row["teacher_tokens"] == [55, 66]
    assert row["teacher_logits_answer_token_ids"] == [55, 66]
