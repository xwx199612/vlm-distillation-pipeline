from __future__ import annotations

from pathlib import Path


SCRIPT = Path("scripts/run_parallel_switch_kd_precompute_4gpu.sh")


def test_parallel_precompute_script_preserves_switch_kd_method_in_generated_configs():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'base_config_path = Path("configs/parsing_switch_kd.yaml")' in text
    assert "distillation.method={config_data.get('distillation', {}).get('method')}" in text
    assert "teacher_logits_field={config_data.get('distillation', {}).get('teacher_logits_field')}" in text


def test_parallel_precompute_script_validates_teacher_logits_before_merge():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "teacher-logits shard has zero valid teacher_logits rows" in text
    assert "merged teacher logits rows missing valid teacher_logits" in text
    assert "first merged teacher logits row does not contain teacher_logits dict" in text
