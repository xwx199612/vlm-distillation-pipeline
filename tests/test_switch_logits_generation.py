from __future__ import annotations

from pathlib import Path

import pytest

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.data_manifest import VlmSample
from vlm_distill.stage_visual_switch_logits import VisualSwitchDistiller, _validate_switch_logits_row


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


def test_switch_logits_prompt_len_includes_visual_tokens(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    row = distiller.generate_for_sample(
        VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world"),
        base_row={"teacher_tokens": [10, 11, 12], "visual_token_count": 7},
    )

    assert row["switch_logits_prompt_len"] == 9
    assert row["switch_logits"]["shape"][1] - row["switch_logits_prompt_len"] == len(row["teacher_tokens"])


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

    with pytest.raises(ValueError, match="prompt_len alignment"):
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
