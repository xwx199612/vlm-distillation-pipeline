from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import vlm_distill.stage_teacher_logits as stage_teacher_logits

from vlm_distill.config_schema import (
    DataConfig,
    DistillationConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
    load_config,
    resolve_label_path,
    resolve_prediction_path,
    resolve_switch_logits_path,
    resolve_teacher_logits_path,
)
from vlm_distill.data_manifest import VlmSample, summarize_label_rows, validate_manifest
from vlm_distill.manifest_builder import infer_manifest_task_from_config_path
from vlm_distill.stage_answer_labeling import _format_prompt, _load_teacher_image
from vlm_distill.stage_teacher_logits import (
    TeacherLogitsGenerator,
    _is_valid_logits_row,
    _load_completed_ids,
    _resolve_teacher_logits_mode,
    create_teacher_logits_dataset,
)
from vlm_distill.vlm_batching import load_training_image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_parsing_manifest_validates_without_question(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "screen.jpg")
    manifest = tmp_path / "screen.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "task": "parsing",
                "query": "List all visible UI elements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = validate_manifest(manifest, image_root=image_root)
    assert len(samples) == 1
    assert samples[0].query == "List all visible UI elements."
    assert samples[0].target_label is None


def test_grounding_manifest_requires_target_label(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "ground.jpg")
    manifest = tmp_path / "ground.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "ground-1",
                "image": "ground.jpg",
                "task": "grounding",
                "target_label": "YouTube",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = validate_manifest(manifest, image_root=image_root)
    assert len(samples) == 1
    assert samples[0].target_label == "YouTube"


def test_grounding_prompt_formats_target_label(tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(manifest_path=tmp_path / "manifest.jsonl", distill_path=tmp_path / "distill.jsonl"),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(prompt_template="Target label: {target_label}\nTask: {task}"),
    )
    prompt = _format_prompt(
        config,
        VlmSample(
            id="ground-1",
            image="ground.jpg",
            task="grounding",
            target_label="YouTube",
            target_type="app_icon",
        ),
    )

    assert prompt == "Target label: YouTube\nTask: grounding"


def test_load_config_accepts_legacy_target_field(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
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
  target_field: student_target
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.distillation.prompt_template == "Query: {query}\nAnswer:"


def test_load_config_interpolates_response_options(tmp_path: Path):
    config_path = tmp_path / "response.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
  student_quantization: 4bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/{task_name}_response_{response_profile}
  adapter_dir: outputs/{task_name}_response_{response_profile}/adapter
  quantization: "{student_quantization}"
distillation:
  method: response
  prompt_template: "query: {query}\\nAnswer:"
evaluation:
  output_path: outputs/{task_name}_response_{response_profile}/eval_report.json
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert str(config.data.distill_path) == "D:\\TV_data\\teacher_parsing\\parsing_teacher_labels_480p_8bit.jsonl"
    assert str(config.student.output_dir) == "outputs\\parsing_response_480p_8bit_student_4bit"
    assert config.student.quantization == "4bit"
    assert config.distillation.prompt_template == "query: {query}\nAnswer:"


def test_load_config_interpolates_split_distillation_paths(tmp_path: Path):
    config_path = tmp_path / "switch.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
  student_quantization: 4bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: outputs/{task_name}_switch_kd_{response_profile}.jsonl
  label_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
  teacher_logits_path: outputs/{task_name}_teacher_logits_{label_profile}.jsonl
  switch_logits_path: outputs/{task_name}_switch_logits_{response_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/out
  adapter_dir: outputs/adapter
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert str(resolve_label_path(config.data)) == "D:\\TV_data\\teacher_parsing\\parsing_teacher_labels_480p_8bit.jsonl"
    assert str(resolve_teacher_logits_path(config.data)) == "outputs\\parsing_teacher_logits_480p_8bit.jsonl"
    assert str(resolve_switch_logits_path(config.data)) == "outputs\\parsing_switch_logits_480p_8bit_student_4bit.jsonl"


def test_load_config_interpolates_prediction_path(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
  prediction_path: outputs/{task_name}_merged_predictions_{label_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/out
  adapter_dir: outputs/adapter
  inference_model_path: outputs/student/merged
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert str(resolve_prediction_path(config.data)) == "outputs\\parsing_merged_predictions_480p_8bit.jsonl"
    assert config.student.inference_model_path == "outputs/student/merged"


def test_teacher_logits_command_uses_adaptive_topk_for_switch_kd_config(tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(
            model_name="mock-student",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
        ),
        distillation=DistillationConfig(method="switch_kd"),
    )

    assert _resolve_teacher_logits_mode(config) == "adaptive_topk"


def test_teacher_logits_switch_kd_ignores_label_only_completed_rows(tmp_path: Path):
    output_path = tmp_path / "teacher_logits.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "teacher_answer": "answer",
                "teacher_generated_ids": [[1, 2]],
                "teacher_rationale": "label only",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _load_completed_ids(output_path, field_name="teacher_logits", require_logits=True)

    assert completed.ids == set()
    assert completed.valid_count == 0
    assert completed.invalid_count == 1
    assert completed.first_invalid_keys == [
        "id",
        "teacher_answer",
        "teacher_generated_ids",
        "teacher_rationale",
    ]


def test_teacher_logits_row_validation_rejects_label_only_row():
    assert not _is_valid_logits_row({"id": "x", "teacher_answer": "answer"}, "teacher_logits")


def test_teacher_logits_switch_kd_mock_writes_logits_dict(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "screen.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample-1", "image": "screen.jpg", "task": "parsing", "query": "q"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "teacher_logits.jsonl"
    config = PipelineConfig(
        data=DataConfig(
            manifest_path=manifest,
            distill_path=tmp_path / "distill.jsonl",
            teacher_logits_path=output_path,
            image_root=image_root,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )

    create_teacher_logits_dataset(config)
    row = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert row["teacher_logits_format"] == "adaptive_topk"
    assert row["teacher_logits_prompt_len"] == 0
    assert row["teacher_logits_vocab_size"] == row["teacher_logits"]["vocab_size"]
    assert _is_valid_logits_row(row, "teacher_logits")


def test_load_config_accepts_legacy_teacher_label_quantization_option(tmp_path: Path):
    config_path = tmp_path / "response_legacy.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_label_quantization: 8bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
teacher:
  model_name: mock-teacher
  quantization: "{teacher_quantization}"
student:
  model_name: mock-student
  output_dir: outputs/out
  adapter_dir: outputs/adapter
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert str(config.data.distill_path) == "D:\\TV_data\\teacher_parsing\\parsing_teacher_labels_480p_8bit.jsonl"
    assert config.teacher.quantization == "8bit"


def test_load_training_image_resizes_to_target_height(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / "tall.png"
    Image.new("RGB", (900, 1800), color=(255, 255, 255)).save(image_path)

    resized = load_training_image(tmp_path, "images/tall.png", resize_mode="480p")
    original = load_training_image(tmp_path, "images/tall.png", resize_mode="original")

    assert resized.size == (240, 480)
    assert original.size == (900, 1800)


def test_load_teacher_image_resizes_to_1080p(tmp_path: Path):
    image_path = tmp_path / "teacher_tall.png"
    Image.new("RGB", (900, 1800), color=(255, 255, 255)).save(image_path)

    resized = _load_teacher_image(image_path, "1080p")

    assert resized.size == (540, 1080)


def test_teacher_logits_uses_teacher_resize_setting(tmp_path: Path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / "teacher_tall.png"
    Image.new("RGB", (900, 1800), color=(255, 255, 255)).save(image_path)

    config = PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
            image_root=image_root,
        ),
        teacher=TeacherConfig(
            model_name="mock-teacher",
            backend="hf",
            image_resize="480p",
        ),
        student=StudentConfig(
            model_name="mock-student",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
        ),
        distillation=DistillationConfig(),
    )

    generator = TeacherLogitsGenerator(config)
    captured: dict[str, str] = {}

    class DummyModel:
        device = "cpu"

    class DummyInputIds:
        shape = (1, 3)

    class DummyGeneratedIds:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[4]]

    def fake_load_teacher_image(path: Path, resize_mode: str):
        captured["path"] = str(path)
        captured["resize_mode"] = resize_mode
        return _load_teacher_image(path, resize_mode)

    def fake_load():
        generator._model = DummyModel()
        generator._processor = object()

    monkeypatch.setattr(stage_teacher_logits, "_load_teacher_image", fake_load_teacher_image)
    monkeypatch.setattr(generator, "load", fake_load)
    monkeypatch.setattr(generator, "_build_multimodal_inputs", lambda image, prompt: {"input_ids": DummyInputIds()})
    monkeypatch.setattr(stage_teacher_logits, "_move_batch_to_device", lambda batch, device: batch)
    monkeypatch.setattr(generator, "_generate", lambda inputs, include_scores: object())
    monkeypatch.setattr(
        stage_teacher_logits,
        "_extract_generated_ids_and_scores",
        lambda generation, prompt_len, include_scores: (DummyGeneratedIds(), []),
    )
    monkeypatch.setattr(generator, "_decode", lambda generated_ids: '{"elements":["Home"]}')

    result = generator.generate_for_sample(
        VlmSample(
            id="sample-1",
            image="teacher_tall.png",
            task="parsing",
            query="List visible UI elements.",
        ),
        mode="response",
    )

    assert result["teacher_answer"]
    assert captured == {
        "path": str(image_path),
        "resize_mode": "480p",
    }


def test_infer_manifest_task_from_config_path():
    assert infer_manifest_task_from_config_path(Path("configs/parsing_test.yaml")) == "parsing"
    assert infer_manifest_task_from_config_path(Path("configs/grounding_test.yaml")) == "grounding"


def test_summarize_label_rows_counts_teacher_answer_rows(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "teacher_answer": "hello"}),
                json.dumps({"id": "b"}),
                json.dumps({"id": "c", "teacher_answer": "   "}),
                json.dumps({"id": "d", "teacher_answer": {"elements": ["Home"]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_label_rows(labels)

    assert summary == {
        "total_rows": 4,
        "teacher_answer_rows": 3,
        "non_empty_teacher_answer_rows": 2,
    }
