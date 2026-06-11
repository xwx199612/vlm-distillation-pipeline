from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_schema import PipelineConfig
from .data_manifest import VlmSample, read_jsonl, validate_manifest, write_jsonl
from .logits_cache_utils import compact_logits


class TeacherLogitsGenerator:
    """Offline teacher forward pass that caches token logits for DBiLD."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.config.teacher.model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.teacher.model_name,
            device_map=self.config.teacher.device_map or "auto",
            trust_remote_code=True,
        ).eval()

    def generate_for_sample(self, sample: VlmSample, target_text: str) -> dict[str, Any]:
        if self._model is None:
            self.load()

        import torch
        from PIL import Image

        image_path = self.config.data.image_root / sample.image
        image = Image.open(image_path).convert("RGB")
        prompt = self.config.distillation.prompt_template.format(question=sample.question)
        text = f"{prompt} {target_text}".strip()

        with torch.no_grad():
            prompt_inputs = self._processor(images=image, text=prompt, return_tensors="pt")
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            outputs = self._model(**inputs)
            if not hasattr(outputs, "logits"):
                raise ValueError("Teacher model forward did not return logits.")
            cached = compact_logits(
                outputs.logits,
                self.config.distillation.max_cached_logits_vocab,
            )

        field = self.config.distillation.teacher_logits_field
        return {
            field: cached,
            f"{field}_format": "topk" if self.config.distillation.max_cached_logits_vocab else "dense",
            f"{field}_prompt_len": int(prompt_inputs["input_ids"].shape[1]),
            f"{field}_vocab_size": int(outputs.logits.shape[-1]),
        }


def create_teacher_logits_dataset(config: PipelineConfig) -> Path:
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    base_rows = read_jsonl(config.data.distill_path) if config.data.distill_path.exists() else []
    rows_by_id = {str(row["id"]): row for row in base_rows}
    generator = TeacherLogitsGenerator(config)
    target_field = config.distillation.target_field
    output_rows: list[dict[str, Any]] = []

    for sample in samples:
        row = rows_by_id.get(sample.id, asdict(sample))
        target_text = str(row.get(target_field) or sample.answer or "")
        if not target_text:
            raise ValueError(
                f"Sample {sample.id} is missing '{target_field}'. Run `label` before `teacher-logits`."
            )
        row.update(generator.generate_for_sample(sample, target_text=target_text))
        output_rows.append(row)

    write_jsonl(config.data.distill_path, output_rows)
    return config.data.distill_path
