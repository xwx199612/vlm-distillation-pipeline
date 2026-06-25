from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig, load_config
from vlm_distill.data_manifest import VlmSample
from vlm_distill.stage_visual_switch_logits import (
    VisualSwitchDistiller,
    _load_switch_base_rows,
    _validate_switch_logits_row,
)


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd", prompt_template="Query: {query}"),
    )


class _FakeProjector(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x=None, hidden_states=None, inputs_embeds=None):
        tensor = x if x is not None else hidden_states if hidden_states is not None else inputs_embeds
        if tensor is None:
            raise ValueError("expected tensor input")
        return self.linear(tensor)


def test_paper_mode_config_loads_correctly():
    config = load_config(Path("configs/parsing_switch_kd.yaml"))

    assert config.distillation.switch_kd.enabled is True
    assert config.distillation.switch_kd.visual_switch.mode == "paper"
    assert config.distillation.switch_kd.visual_switch.teacher_projector == "native"
    assert config.distillation.switch_kd.visual_switch.allow_fallback_adapter is False


def test_switch_logits_are_answer_only(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    row = distiller.generate_for_sample(
        VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world"),
        base_row={"teacher_tokens": [10, 11, 12], "visual_token_count": 7},
    )

    assert row["switch_logits_prompt_len"] == 0
    assert row["switch_logits_aligned_to_answer"] is True
    assert row["switch_logits"]["shape"][1] == len(row["teacher_tokens"])


def test_paper_mode_raises_on_shape_incompatibility(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=_FakeProjector(8, 12)),
    )

    with pytest.raises(ValueError, match="Switch-KD paper path incompatible"):
        distiller._paper_path_projection(torch.zeros(1, 2, 7))


def test_adapter_mode_requires_allow_fallback_adapter(tmp_path: Path):
    config_path = tmp_path / "bad_adapter.yaml"
    config_path.write_text(
        """
data:
  manifest_path: manifest.jsonl
  distill_path: distill.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: out
  adapter_dir: adapter
distillation:
  method: switch_kd
  switch_kd:
    enabled: true
    visual_switch:
      mode: adapter_to_teacher_projector
      teacher_projector: native
      allow_fallback_adapter: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_fallback_adapter"):
        load_config(config_path)


def test_adapter_mode_logs_project_specific_variant(capsys, tmp_path: Path):
    config = _make_config(tmp_path)
    config.distillation.switch_kd.visual_switch.mode = "adapter_to_teacher_projector"
    config.distillation.switch_kd.visual_switch.allow_fallback_adapter = True
    config.distillation.switch_kd.visual_switch.adapter_path = "connector"
    distiller = VisualSwitchDistiller(config)
    distiller._student_model = SimpleNamespace(connector=_FakeProjector(8, 12))
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=_FakeProjector(8, 12)),
    )

    component = distiller._visual_switch_projector_component()
    out = capsys.readouterr().out

    assert component is not None
    assert "This is a project-specific Switch-KD variant, not the original paper path." in out


def test_switch_logits_old_text_only_prompt_len_raises():
    row = {
        "id": "bad",
        "image": "screen.jpg",
        "teacher_tokens": list(range(477)),
        "switch_logits_prompt_len": 287,
        "switch_logits": {
            "indices": [[[0]]],
            "values": [[[1.0]]],
            "shape": [1, 2327, 152064],
            "vocab_size": 152064,
        },
    }

    with pytest.raises(ValueError, match="answer-only alignment"):
        _validate_switch_logits_row(
            row,
            field_name="switch_logits",
            visual_token_placeholder="<image>",
        )


def test_switch_logits_row_contains_valid_compact_payload(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    row = distiller.generate_for_sample(
        VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world"),
        base_row={"teacher_tokens": [1, 2], "visual_token_count": 3},
    )

    _validate_switch_logits_row(
        row,
        field_name="switch_logits",
        visual_token_placeholder="<image>",
    )
    assert {"indices", "values", "vocab_size"}.issubset(row["switch_logits"])
    assert row["switch_logits"]["shape"][1] == len(row["teacher_tokens"])


def test_switch_logits_reads_teacher_rows_from_label_path(tmp_path: Path):
    label_path = tmp_path / "labels.jsonl"
    switch_path = tmp_path / "switch_logits.jsonl"
    label_path.write_text('{"id":"label-row","teacher_answer":"{}","teacher_tokens":[1]}\n', encoding="utf-8")
    switch_path.write_text('{"id":"switch-row","teacher_answer":"{}","teacher_tokens":[2]}\n', encoding="utf-8")
    config = _make_config(tmp_path)
    config.data.label_path = label_path
    config.data.switch_logits_path = switch_path
    config.data.teacher_logits_path = tmp_path / "legacy_teacher_logits.jsonl"

    rows = _load_switch_base_rows(config)

    assert [row["id"] for row in rows] == ["label-row"]


def test_switch_logits_does_not_use_switch_logits_as_teacher_base_when_label_missing(tmp_path: Path):
    switch_path = tmp_path / "switch_logits.jsonl"
    switch_path.write_text('{"id":"switch-row","teacher_answer":"{}","teacher_tokens":[2]}\n', encoding="utf-8")
    config = _make_config(tmp_path)
    config.data.label_path = tmp_path / "missing_labels.jsonl"
    config.data.switch_logits_path = switch_path

    assert _load_switch_base_rows(config) == []
