from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .config_schema import PipelineConfig, format_prompt, resolve_prediction_path
from .data_manifest import VlmSample, read_jsonl
from .model_loading import apply_attn_implementation, resolve_model_path
from .stage_teacher_precompute import _load_teacher_image, _normalize_teacher_answer


class StudentBackend(Protocol):
    def answer(self, sample: VlmSample) -> dict:
        ...


class MockStudent:
    def answer(self, sample: VlmSample) -> dict:
        seed = f"{sample.id}:{sample.query}:{sample.answer or ''}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        confidence = 0.55 + (int(digest[:4], 16) / 0xFFFF) * 0.4

        if sample.answer:
            student_answer = sample.answer
        elif sample.task == "parsing":
            elements = sample.metadata.get("elements") if isinstance(sample.metadata, dict) else None
            student_answer = json.dumps(
                {
                    "elements": elements if isinstance(elements, list) else ["mock icon", "mock settings"],
                },
                ensure_ascii=False,
            )
        elif sample.task == "grounding":
            bbox = sample.metadata.get("bbox") if isinstance(sample.metadata, dict) else None
            student_answer = json.dumps(
                {
                    "label": sample.target_label or "target",
                    "bbox": bbox or [0, 0, 100, 100],
                },
                ensure_ascii=False,
            )
        else:
            student_answer = f"mock answer for {sample.task}"

        return {
            "student_answer": student_answer,
            "student_confidence": round(confidence, 4),
            "student_rationale": "Mock student backend used for pipeline validation.",
        }


class HuggingFaceStudent:
    def __init__(self, config: PipelineConfig):
        self.config = config
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVLM
            except ImportError:  # pragma: no cover - fallback for older transformers
                from transformers import AutoModelForVision2Seq as AutoModelForVLM
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers and bitsandbytes to use the Hugging Face student backend."
            ) from exc

        model_path = resolve_model_path(
            config.student.inference_model_path or config.student.model_name
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )

        model_kwargs: dict = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, config.student.attn_implementation)
        if config.student.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.student.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs["local_files_only"] = True
        self.model = AutoModelForVLM.from_pretrained(model_path, **model_kwargs)
        if _should_load_prediction_adapter(config):
            adapter_path = _resolve_prediction_adapter_path(config)
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                local_files_only=True,
            )
            if config.student.merge_adapter:
                self.model = self.model.merge_and_unload()
        self.model.eval()

    def answer(self, sample: VlmSample) -> dict:
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=sample.query,
            target_label=sample.target_label,
            target_type=sample.target_type,
            task=sample.task,
        )
        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(image_path, self.config.training.image_resize)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.config.teacher.max_new_tokens,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "student_answer": answer.strip(),
            "student_confidence": 1.0,
            "student_rationale": "Generated by Hugging Face student backend.",
        }


def build_student_backend(config: PipelineConfig) -> StudentBackend:
    if config.student.model_name.startswith("mock-"):
        return MockStudent()
    return HuggingFaceStudent(config)


def _should_load_prediction_adapter(config: PipelineConfig) -> bool:
    return config.student.load_adapter or config.student.merge_adapter


def _resolve_prediction_adapter_path(config: PipelineConfig) -> Path:
    return config.student.inference_adapter_path or config.student.adapter_dir


def create_student_predictions(config: PipelineConfig, samples: list[VlmSample]) -> Path:
    output_path = resolve_prediction_path(config.data)
    completed_ids = _load_completed_ids(output_path)
    total = len(samples)
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Predicting samples: total={total}, completed={len(completed_ids)}, "
        f"pending={len(pending_samples)}, output={output_path}"
    )

    if not pending_samples:
        print("No pending samples. Existing prediction output is already complete for this manifest.")
        return output_path

    student = build_student_backend(config)
    completed_now = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(pending_samples, start=1):
            started = time.perf_counter()
            row = _predict_sample(student, sample)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed_now += 1
            elapsed = time.perf_counter() - started
            total_done = len(completed_ids) + completed_now
            print(
                f"[predict] wrote {total_done}/{total} "
                f"id={sample.id} elapsed={elapsed:.2f}s"
            )

    return output_path


def _predict_sample(student: StudentBackend, sample: VlmSample) -> dict:
    prediction = student.answer(sample)
    prediction["student_answer"] = _normalize_teacher_answer(sample, prediction["student_answer"])
    return {
        **asdict(sample),
        **prediction,
    }


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            completed_ids.add(str(sample_id))
    return completed_ids
