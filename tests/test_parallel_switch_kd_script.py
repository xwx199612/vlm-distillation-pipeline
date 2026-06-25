from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys

import yaml


SCRIPT = Path("scripts/run_parallel_switch_kd_precompute_4gpu.sh")
BASE_CONFIG = Path("configs/parsing_switch_kd.yaml")
DDP_CONFIG = Path("configs/parsing_switch_kd_ddp.yaml")
GENERATED_CONFIGS = [
    Path(f"configs/generated/parsing_switch_kd_teacher_precompute_gpu{gpu}.yaml")
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
        assert config["distillation"]["switch_kd"]["enabled"] is True
        assert config["distillation"]["switch_kd"]["visual_switch"]["mode"] == "paper"
        assert config["distillation"]["switch_kd"]["visual_switch"]["teacher_projector"] == "native"
        assert config["distillation"]["switch_kd"]["visual_switch"]["allow_fallback_adapter"] is False
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
    assert "switch_kd.visual_switch.mode=" in text
    assert "Switch-KD visual-switch mode:" in text
    assert "T-Projector definition: teacher native projector / merger" in text
    assert "Visual-switch path: student visual encoder output -> teacher projector/merger -> teacher LLM" in text
    assert "Fallback adapter:" in text
    assert "teacher-logits" not in text


def test_generated_shard_configs_preserve_unified_teacher_precompute_semantics():
    for path in GENERATED_CONFIGS:
        config = _read_yaml(path)

        assert config["distillation"]["method"] == "switch_kd"
        assert config["distillation"]["teacher_logits"] is True
        assert config["distillation"]["switch_kd"]["visual_switch"]["mode"] == "paper"
        assert config["distillation"]["switch_kd"]["visual_switch"]["allow_fallback_adapter"] is False
        assert config["data"]["label_path"].startswith(
            "outputs/switch-kd/shards/parsing_teacher_labels_shard"
        )
        assert "teacher_logits_path" not in config["data"]


def test_teacher_logits_resolver_falls_back_to_label_path_when_deprecated_path_absent():
    from vlm_distill.config_schema import DataConfig, DistillationConfig, resolve_teacher_logits_path

    data = DataConfig(
        manifest_path=Path("manifest.jsonl"),
        distill_path=Path("distill.jsonl"),
        label_path=Path("labels.jsonl"),
    )

    assert resolve_teacher_logits_path(data) == Path("labels.jsonl")
    assert DistillationConfig().teacher_logits is True


def test_parallel_precompute_script_validates_teacher_logits_before_merge():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "unified teacher precompute shard has zero valid teacher_logits rows" in text
    assert "merged teacher logits rows missing valid teacher_logits" in text
    assert "first merged teacher logits row does not contain teacher_logits dict" in text


def test_parallel_precompute_script_removes_teacher_logits_stage_from_main_workflow():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'run_stage "teacher-precompute"' in text
    assert 'run_stage "switch-logits"' in text
    assert 'run_stage "teacher-logits"' not in text
    assert 'teacher-precompute|switch-logits|all)' in text
    assert "label|teacher-precompute|switch-logits|all" not in text
    assert "run_parallel_switch_kd_precompute_4gpu.sh label" not in text


def test_parallel_precompute_script_rejects_label_stage():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "label" not in text.split("case \"${REQUESTED_STAGE}\" in", 1)[1].split(";;", 1)[0]


def test_parallel_precompute_script_fails_on_label_stage():
    result = subprocess.run(
        ["bash", str(SCRIPT), "label"],
        cwd=Path("."),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ERROR: unsupported stage: label" in (result.stdout + result.stderr)


def test_parallel_precompute_script_dry_run_all_runs_teacher_precompute_then_switch_logits(tmp_path):
    repo_root = Path(".").resolve()
    home = tmp_path / "home"
    conda_dir = home / "miniforge3" / "etc" / "profile.d"
    conda_dir.mkdir(parents=True)
    (conda_dir / "conda.sh").write_text(
        "conda() { :; }\n",
        encoding="utf-8",
    )

    project_dir = home / "vlm_distill"
    project_dir.mkdir(parents=True)
    (project_dir / "Switch-KD").symlink_to(repo_root, target_is_directory=True)

    manifest_path = repo_root / "outputs" / "switch-kd" / "parsing_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                '{"id":"0","query":"sample 0"}',
                '{"id":"1","query":"sample 1"}',
                '{"id":"2","query":"sample 2"}',
                '{"id":"3","query":"sample 3"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HOME"] = str(home)
    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_parallel_switch_kd_precompute_4gpu.sh"), "--dry-run", "all"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    try:
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        first = output.index("python -m vlm_distill.cli teacher-precompute")
        second = output.index("python -m vlm_distill.cli switch-logits")
        assert first < second
        assert "Switch-KD visual-switch mode: paper" in output
        assert "T-Projector definition: teacher native projector / merger" in output
        assert "Visual-switch path: student visual encoder output -> teacher projector/merger -> teacher LLM" in output
        assert "Fallback adapter: disabled" in output
    finally:
        manifest_path.unlink(missing_ok=True)
        for gpu in range(4):
            shard_path = repo_root / "outputs" / "switch-kd" / "shards" / f"parsing_manifest_shard{gpu}.jsonl"
            shard_path.unlink(missing_ok=True)


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


def test_cli_teacher_precompute_uses_unified_teacher_precompute(monkeypatch, tmp_path):
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

    monkeypatch.setattr(sys, "argv", ["vlm-distill", "teacher-precompute", "--config", str(config_path)])
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: ["sample"])
    monkeypatch.setattr(
        cli,
        "create_teacher_precompute_dataset",
        lambda config, samples: calls.append("teacher_precompute") or tmp_path / "labels.jsonl",
    )

    cli.main()

    assert calls == ["teacher_precompute"]


def test_cli_teacher_logits_is_not_a_compute_command(monkeypatch, tmp_path):
    import vlm_distill.cli as cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(sys, "argv", ["vlm-distill", "teacher-logits", "--config", str(config_path)])
    monkeypatch.setattr(
        cli,
        "create_teacher_precompute_dataset",
        lambda *args, **kwargs: calls.append("teacher_precompute"),
    )

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover - argparse must reject the removed command
        raise AssertionError("teacher-logits command unexpectedly succeeded")

    assert calls == []


def test_cli_no_longer_imports_old_label_generation_flow():
    text = Path("src/vlm_distill/cli.py").read_text(encoding="utf-8")

    assert Path("src/vlm_distill/stage_teacher_precompute.py").exists()
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
        assert "create_teacher_logits_dataset" not in text


def test_no_independent_teacher_logits_dataset_path_remains():
    for path in Path("src/vlm_distill").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def create_teacher_logits_dataset" not in text
        assert "create_teacher_logits_compat_dataset" not in text
