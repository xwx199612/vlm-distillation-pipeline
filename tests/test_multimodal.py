import torch
from PIL import Image

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.logits_cache_utils import align_reference_logits_to_suffix, compact_logits
from vlm_distill.stage_student_training import VocabAlignment, _load_training_rows, _remap_reference_logits_to_student_vocab
from vlm_distill.vlm_batching import VlmDataCollator, encode_vlm_training_sample


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()

    def __call__(self, images=None, text="", return_tensors="pt", truncation=True, max_length=128):
        del images, truncation, max_length
        if isinstance(text, list):
            text = text[0]
        token_count = max(1, len(text.split()))
        input_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 4, 4),
        }


def test_encode_vlm_training_sample_masks_prompt_and_image_prefix():
    image = Image.new("RGB", (8, 8))
    encoded = encode_vlm_training_sample(
        _FakeProcessor(),
        image=image,
        prompt="Question: cup?",
        target="a cup",
        max_length=64,
    )
    assert encoded.prompt_token_len == 2
    assert encoded.model_inputs["labels"][0].item() == -100
    assert encoded.model_inputs["labels"][-1].item() != -100
    assert "pixel_values" in encoded.model_inputs


def test_collator_keeps_logits_metadata():
    collator = VlmDataCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, -100, 3]),
                "pixel_values": torch.zeros(3, 4, 4),
                "prompt_token_len": 2,
                "teacher_logits": {"indices": [], "values": [], "shape": [1, 2, 4], "vocab_size": 4},
                "teacher_logits_prompt_len": 2,
                "teacher_logits_vocab_size": 4,
            }
        ]
    )
    assert batch["teacher_logits"]["vocab_size"] == 4
    assert batch["teacher_logits_prompt_len"] == 2


def test_align_reference_logits_to_suffix_places_answer_region():
    reference = torch.randn(1, 5, 4)
    aligned = align_reference_logits_to_suffix(
        reference,
        target_shape=(1, 6, 4),
        reference_prompt_len=2,
        student_prompt_len=3,
        dtype=torch.float32,
    )
    assert aligned.shape == (1, 6, 4)
    assert torch.isfinite(aligned[:, 3:, :]).all()


def test_load_training_rows_merges_label_and_logits_files(tmp_path):
    label_path = tmp_path / "labels.jsonl"
    switch_logits_path = tmp_path / "switch_logits.jsonl"

    label_path.write_text(
        '{"id":"sample-1","image":"a.png","task":"parsing","query":"q","teacher_answer":"{\\"elements\\":[\\"Home\\"]}","teacher_logits":{"vocab_size":4},"teacher_logits_prompt_len":2,"teacher_logits_vocab_size":4}\n',
        encoding="utf-8",
    )
    switch_logits_path.write_text(
        '{"id":"sample-1","switch_logits":{"vocab_size":4},"switch_logits_prompt_len":3,"switch_logits_vocab_size":4}\n',
        encoding="utf-8",
    )

    config = PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "legacy.jsonl",
            label_path=label_path,
            switch_logits_path=switch_logits_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )

    rows = _load_training_rows(config)

    assert len(rows) == 1
    assert rows[0]["teacher_answer"] == '{"elements":["Home"]}'
    assert rows[0]["teacher_logits"]["vocab_size"] == 4
    assert rows[0]["switch_logits"]["vocab_size"] == 4


def test_remap_reference_logits_to_student_vocab_keeps_shared_prefix():
    reference = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32)
    remapped = _remap_reference_logits_to_student_vocab(
        reference,
        reference_vocab=5,
        student_vocab_size=7,
        vocab_alignment=VocabAlignment(shared_token_vocab_size=3),
    )

    assert remapped.shape == (1, 1, 7)
    assert torch.equal(remapped[..., :3], reference[..., :3])
    assert (remapped[..., 3:] == torch.finfo(reference.dtype).min).all()
