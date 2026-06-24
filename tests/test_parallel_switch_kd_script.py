from __future__ import annotations

from pathlib import Path
import sys

import yaml


SCRIPT = Path("scripts/run_parallel_switch_kd_precompute_4gpu.sh")
BASE_CONFIG = Path("configs/parsing_switch_kd.yaml")
DDP_CONFIG = Path("configs/parsing_switch_kd_ddp.yaml")
GENERATED_CONFIGS = [
    Path(f"configs/generated/parsing_switch_kd_label_gpu{gpu}.yaml")
    for gpu in range(4)
] + [
    Path(f"configs/generated/parsing_switch_kd_switch_logits_gpu{gpu}.yaml")
    for gpu in range(4)
]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_base_switch_kd_configs_are_unified_teacher_precompute_source_of_truth():
    for path in (BASE_CONFIG, DDP_CONFIG):
        if not path.exists():
            continue
        config = _read_yaml(path)

        assert config["distillation"]["method"] == "switch_kd"
        assert config["distillation"]["teacher_logits"] is True
        assert config["data"]["label_path"] == (
            "outputs/switch-kd/{task_name}_teacher_labels_{label_profile}.jsonl"
        )
        assert "teacher_logits_path" not in config["data"]


def test_parallel_precompute_script_preserves_switch_kd_method_in_generated_configs():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'base_config_path = Path("configs/parsing_switch_kd.yaml")' in text
    assert "distillation.method={config_data.get('distillation', {}).get('method')}" in text
    assert "distillation.teacher_logits={config_data.get('distillation', {}).get('teacher_logits')}" in text
    assert "teacher_logits_field={config_data.get('distillation', {}).get('teacher_logits_field')}" in text
    assert "canonical_teacher_output_path=label_path" in text
    assert "teacher-logits stage is removed/deprecated" in text


def test_generated_shard_configs_preserve_unified_teacher_precompute_semantics():
    for path in GENERATED_CONFIGS:
        config = _read_yaml(path)

        assert config["distillation"]["method"] == "switch_kd"
        assert config["distillation"]["teacher_logits"] is True
        assert config["data"]["label_path"].startswith(
            "outputs/switch-kd/shards/parsing_teacher_labels_shard"
        )
        assert "teacher_logits_path" not in config["data"]


def test_teacher_logits_resolver_falls_back_to_label_path_when_deprecated_path_absent():
    from vlm_distill.config_schema import DataConfig, resolve_teacher_logits_path

    data = DataConfig(
        manifest_path=Path("manifest.jsonl"),
        distill_path=Path("distill.jsonl"),
        label_path=Path("labels.jsonl"),
    )

    assert resolve_teacher_logits_path(data) == Path("labels.jsonl")


def test_parallel_precompute_script_validates_teacher_logits_before_merge():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "unified teacher precompute shard has zero valid teacher_logits rows" in text
    assert "merged teacher logits rows missing valid teacher_logits" in text
    assert "first merged teacher logits row does not contain teacher_logits dict" in text


def test_parallel_precompute_script_removes_teacher_logits_stage_from_main_workflow():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'run_stage "label"' in text
    assert 'run_stage "switch-logits"' in text
    assert 'run_stage "teacher-logits"' not in text
    assert 'label|switch-logits|all)' in text
    assert "label|teacher-logits|switch-logits|all" not in text


def test_cli_label_uses_unified_teacher_precompute(monkeypatch, tmp_path):
    import vlm_distill.cli as cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  manifest_path: manifest.jsonl
  distill_path: distill.jsonl
  label_path: labels.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: out
  adapter_dir: adapter
distillation:
  method: switch_kd
  teacher_logits: true
""".strip(),
        encoding="utf-8",
    )
    calls: list[str] = []

    monkeypatch.setattr(sys, "argv", ["vlm-distill", "label", "--config", str(config_path)])
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: ["sample"])
    monkeypatch.setattr(
        cli,
        "create_teacher_precompute_dataset",
        lambda config, samples: calls.append("teacher_precompute") or tmp_path / "labels.jsonl",
    )

    cli.main()

    assert calls == ["teacher_precompute"]


def test_cli_no_longer_imports_old_label_generation_flow():
    text = Path("src/vlm_distill/cli.py").read_text(encoding="utf-8")

    assert "stage_teacher_precompute import create_teacher_precompute_dataset" in text
    assert "stage_answer_labeling import create_distillation_dataset" not in text
    assert '"teacher-logits",' not in text


def test_old_teacher_modules_are_wrappers_only():
    for path in (
        Path("src/vlm_distill/stage_answer_labeling.py"),
        Path("src/vlm_distill/stage_teacher_logits.py"),
    ):
        text = path.read_text(encoding="utf-8")

        assert "from .stage_teacher_precompute import" in text
        assert "class HuggingFaceTeacher" not in text
        assert "class TeacherLogitsGenerator" not in text
        assert "def create_teacher_precompute_dataset" not in text
        assert "def create_distillation_dataset" not in text
