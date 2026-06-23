from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from .config_schema import PipelineConfig, resolve_teacher_logits_path
from .data_manifest import VlmSample, validate_manifest, write_jsonl
from .device_utils import (
    batch_to_device,
    ensure_stage_uses_cuda,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from .model_loading import apply_attn_implementation, resolve_model_path
from .stage_answer_labeling import _load_teacher_image


DistillationMode = Literal["response", "adaptive_topk", "switch_kd"]

INACTIVE_LOGIT = -1.0e4


class TeacherLogitsGenerator:
    """
    Generate teacher distillation data in one pass.

    Supported modes:

    1. response
       - teacher.generate()
       - saves teacher_answer only
       - for normal response distillation / SFT

    2. adaptive_topk
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k generation logits
       - for adaptive top-k logits distillation

    3. switch_kd
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k logits + entropy + entropy weights
       - for Switch-KD style distillation data
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._input_device = None

    def load(self) -> None:
        if self.config.teacher.backend == "mock":
            return

        if self.config.teacher.backend != "hf":
            raise ValueError(
                "teacher-logits currently supports backend='hf' or backend='mock'. "
                f"Got backend={self.config.teacher.backend!r}."
            )

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        model_path = resolve_model_path(self.config.teacher.model_name)
        requested_device_map = resolve_requested_device_map(
            self.config.teacher.device_map,
            quantization=self.config.teacher.quantization,
            role="teacher",
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": requested_device_map,
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, self.config.teacher.attn_implementation)

        if self.config.teacher.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.teacher.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if self.config.teacher.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.config.teacher.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.config.teacher.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForVLM.from_pretrained(
            model_path,
            **model_kwargs,
            local_files_only=True,
        ).eval()
        self._input_device = select_model_input_device(
            self._model,
            preferred_modules=(getattr(self._model, "visual", None),),
            label="Teacher",
        )
        print_stage_model_debug(
            stage_label="Teacher logits",
            model_path=model_path,
            quantization_mode=self.config.teacher.quantization,
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Teacher logits",
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )

    def generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        if self.config.teacher.backend == "mock":
            return self._mock_generate_for_sample(sample, mode=mode)

        if self._model is None or self._processor is None:
            self.load()

        import torch

        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(
            image_path,
            self.config.teacher.image_resize,
        )
        prompt = _format_prompt(self.config, sample)

        with torch.no_grad():
            inputs = self._build_multimodal_inputs(image, prompt)
            prompt_len = int(inputs["input_ids"].shape[1])
            inputs = batch_to_device(inputs, self._input_device)

            include_scores = mode in {"adaptive_topk", "switch_kd"}
            generation = self._generate(inputs, include_scores=include_scores)

            generated_ids, scores = _extract_generated_ids_and_scores(
                generation,
                prompt_len=prompt_len,
                include_scores=include_scores,
            )

            answer = self._decode(generated_ids).strip()

            result: dict[str, Any] = {
                "teacher_answer": answer,
                "teacher_confidence": 1.0,
                "teacher_rationale": f"Generated by Hugging Face teacher in {mode} mode.",
                "distillation_mode": mode,
                "teacher_generated_ids": generated_ids.detach().cpu().tolist(),
            }

            if mode == "response":
                return result

            if not scores:
                raise ValueError(
                    "Teacher generation did not return scores. "
                    "Cannot build logits distillation dataset."
                )

            logits_payload = self._build_generation_logits_payload(
                scores=scores,
                mode=mode,
                prompt_len=prompt_len,
            )
            result.update(logits_payload)
            return result

    def _generate(self, inputs: dict[str, Any], *, include_scores: bool):
        temperature = float(self.config.teacher.temperature)
        do_sample = temperature > 0

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.teacher.max_new_tokens),
            "do_sample": do_sample,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature

        if include_scores:
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True

        return self._model.generate(
            **inputs,
            **generate_kwargs,
        )

    def _decode(self, generated_ids):
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _build_multimodal_inputs(self, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

    def _build_generation_logits_payload(
        self,
        *,
        scores: list[Any],
        mode: DistillationMode,
        prompt_len: int,
    ) -> dict[str, Any]:
        field = (
            self.config.distillation.switch_logits_field
            if mode == "switch_kd"
            else self.config.distillation.teacher_logits_field
        )

        compact = _compact_adaptive_generation_scores(
            scores=scores,
            mode=mode,
            base_k=int(self.config.distillation.dbild_top_k),
            max_cached_logits_vocab=self.config.distillation.max_cached_logits_vocab,
            temperature=float(self.config.distillation.kd_temperature),
        )

        return {
            field: compact,
            f"{field}_format": mode,
            f"{field}_prompt_len": prompt_len,
            f"{field}_vocab_size": compact["vocab_size"],
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
        }

    def _mock_generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        answer = _mock_answer(sample)

        result: dict[str, Any] = {
            **asdict(sample),
            "teacher_answer": answer,
            "teacher_confidence": 1.0,
            "teacher_rationale": f"Mock teacher used in {mode} mode.",
            "distillation_mode": mode,
            "teacher_generated_ids": [[1, 2, 3]],
        }

        if mode == "response":
            return result

        field = (
            self.config.distillation.switch_logits_field
            if mode == "switch_kd"
            else self.config.distillation.teacher_logits_field
        )

        steps = max(1, min(len(answer.split()), 8))
        vocab_size = 16
        base_k = int(self.config.distillation.dbild_top_k)
        max_k = min(vocab_size, max(2, min(base_k, 8)))

        indices = []
        values = []
        entropy = []
        token_k = []
        entropy_weight = []

        for _ in range(steps):
            step_indices = list(range(max_k))
            step_values = [5.0 - rank for rank in range(max_k)]
            step_entropy = 1.0
            step_k = min(max_k, max(2, base_k))

            indices.append(step_indices)
            values.append(step_values)
            entropy.append(step_entropy)
            token_k.append(step_k)
            entropy_weight.append(_entropy_to_weight(step_entropy))

        compact: dict[str, Any] = {
            "indices": [indices],
            "values": [values],
            "shape": [1, steps, vocab_size],
            "vocab_size": vocab_size,
            "token_k": [token_k],
            "entropy": [entropy],
            "adaptive": True,
        }

        if mode == "switch_kd":
            compact["entropy_weight"] = [entropy_weight]
            compact["switch_kd"] = True

        result.update(
            {
                field: compact,
                f"{field}_format": mode,
                f"{field}_prompt_len": 0,
                f"{field}_vocab_size": vocab_size,
                f"{field}_temperature": float(self.config.distillation.kd_temperature),
            }
        )
        return result


def create_teacher_logits_dataset(config: PipelineConfig) -> Path:
    """
    Create distillation dataset directly from manifest.

    This function intentionally does NOT read old teacher_answer from distill_path.
    It generates fresh teacher outputs according to distillation.method.

    distillation.method:
      - response
      - adaptive_topk
      - switch_kd

    Compatible aliases:
      - sft -> response
      - response_distillation -> response
      - topk / topk_logits / dbild -> adaptive_topk
      - switch / switch-kd -> switch_kd
    """

    mode = _resolve_teacher_logits_mode(config)
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )

    generator = TeacherLogitsGenerator(config)
    rows: list[dict[str, Any]] = []

    for sample in samples:
        row = {
            **asdict(sample),
            **generator.generate_for_sample(sample, mode=mode),
        }
        rows.append(row)

    output_path = resolve_teacher_logits_path(config.data)
    write_jsonl(output_path, rows)
    return output_path


def _resolve_teacher_logits_mode(config: PipelineConfig) -> DistillationMode:
    mode = _resolve_distillation_mode(config)
    if mode == "switch_kd":
        return "adaptive_topk"
    return mode


def _resolve_distillation_mode(config: PipelineConfig) -> DistillationMode:
    raw_mode = str(
        getattr(config.distillation, "mode", None)
        or getattr(config.distillation, "method", "response")
        or "response"
    ).strip().lower()

    aliases = {
        "sft": "response",
        "response": "response",
        "response_distillation": "response",
        "response-distillation": "response",
        "topk": "adaptive_topk",
        "topk_logits": "adaptive_topk",
        "top-k": "adaptive_topk",
        "top-k-logits": "adaptive_topk",
        "adaptive_topk": "adaptive_topk",
        "adaptive-topk": "adaptive_topk",
        "adaptive_topk_logits": "adaptive_topk",
        "adaptive-topk-logits": "adaptive_topk",
        "dbild": "adaptive_topk",
        "switch": "switch_kd",
        "switch_kd": "switch_kd",
        "switch-kd": "switch_kd",
    }

    if raw_mode not in aliases:
        raise ValueError(
            f"Unsupported distillation method: {raw_mode!r}. "
            "Expected one of: response, adaptive_topk, switch_kd."
        )

    return aliases[raw_mode]  # type: ignore[return-value]


def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    template = config.distillation.prompt_template

    try:
        return template.format(
            query=sample.query or "",
            question=sample.query or "",
            target_label=sample.target_label or "",
            target_type=sample.target_type or "",
            task=sample.task,
        )
    except KeyError as exc:
        raise KeyError(
            f"Prompt template references unsupported placeholder: {exc}. "
            "Supported placeholders are: query, question, target_label, target_type, task."
        ) from exc


def _compact_adaptive_generation_scores(
    *,
    scores: list[Any],
    mode: DistillationMode,
    base_k: int,
    max_cached_logits_vocab: int | None,
    temperature: float,
) -> dict[str, Any]:
    """
    Convert generation scores into a compact adaptive top-k cache.

    Input:
      scores: list of tensors, each shaped [batch, vocab]

    Output:
      {
        "indices": [batch, generated_steps, max_k],
        "values": [batch, generated_steps, max_k],
        "shape": [batch, generated_steps, vocab],
        "vocab_size": vocab,
        "token_k": [batch, generated_steps],
        "entropy": [batch, generated_steps],
        "entropy_weight": [batch, generated_steps],  # switch_kd only
      }

    Notes:
      - We keep a rectangular [B, T, max_k] structure so existing cache
        materialization code can still scatter indices/values.
      - For each token, only the first token_k entries are active.
      - Inactive entries are filled with INACTIVE_LOGIT.
    """

    import torch

    if not scores:
        raise ValueError("Cannot compact empty generation scores.")

    first = scores[0].detach().float().cpu()
    if first.ndim != 2:
        raise ValueError(
            f"Expected each generation score to have shape [batch, vocab], got {tuple(first.shape)}"
        )

    batch_size, vocab_size = first.shape

    base_k = max(1, int(base_k))
    low_k = max(1, base_k // 4)
    mid_k = base_k
    high_k = base_k * 2

    if max_cached_logits_vocab is not None:
        high_k = min(high_k, int(max_cached_logits_vocab))

    max_k = min(vocab_size, max(low_k, mid_k, high_k))

    low_entropy_threshold = 1.0
    high_entropy_threshold = 2.5

    step_indices: list[Any] = []
    step_values: list[Any] = []
    step_entropy: list[Any] = []
    step_token_k: list[Any] = []
    step_entropy_weight: list[Any] = []

    safe_temperature = max(float(temperature), 1e-6)

    for score in scores:
        logits = score.detach().float().cpu()
        if logits.ndim != 2:
            raise ValueError(
                f"Expected each generation score to have shape [batch, vocab], got {tuple(logits.shape)}"
            )

        if logits.shape[0] != batch_size or logits.shape[1] != vocab_size:
            raise ValueError(
                "All generation scores must share the same [batch, vocab] shape. "
                f"Expected {(batch_size, vocab_size)}, got {tuple(logits.shape)}"
            )

        scaled_logits = logits / safe_temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        top_values, top_indices = torch.topk(
            logits,
            k=max_k,
            dim=-1,
        )

        token_k = torch.empty((batch_size,), dtype=torch.long)
        entropy_weight = torch.empty((batch_size,), dtype=torch.float32)

        for batch_index in range(batch_size):
            entropy_value = float(entropy[batch_index].item())
            active_k = _adaptive_k(
                entropy_value,
                low_entropy_threshold=low_entropy_threshold,
                high_entropy_threshold=high_entropy_threshold,
                low_k=low_k,
                mid_k=mid_k,
                high_k=high_k,
                max_k=max_k,
            )

            token_k[batch_index] = active_k
            entropy_weight[batch_index] = _entropy_to_weight(entropy_value)

            if active_k < max_k:
                top_values[batch_index, active_k:] = INACTIVE_LOGIT

        step_indices.append(top_indices)
        step_values.append(top_values)
        step_entropy.append(entropy)
        step_token_k.append(token_k)
        step_entropy_weight.append(entropy_weight)

    indices_tensor = torch.stack(step_indices, dim=1)
    values_tensor = torch.stack(step_values, dim=1)
    entropy_tensor = torch.stack(step_entropy, dim=1)
    token_k_tensor = torch.stack(step_token_k, dim=1)

    compact: dict[str, Any] = {
        "indices": indices_tensor.tolist(),
        "values": values_tensor.tolist(),
        "shape": [batch_size, len(scores), vocab_size],
        "vocab_size": int(vocab_size),
        "adaptive": True,
        "token_k": token_k_tensor.tolist(),
        "entropy": entropy_tensor.tolist(),
        "k_policy": {
            "low_entropy_threshold": low_entropy_threshold,
            "high_entropy_threshold": high_entropy_threshold,
            "low_k": int(low_k),
            "mid_k": int(mid_k),
            "high_k": int(high_k),
            "max_k": int(max_k),
        },
    }

    if mode == "switch_kd":
        entropy_weight_tensor = torch.stack(step_entropy_weight, dim=1)
        compact["entropy_weight"] = entropy_weight_tensor.tolist()
        compact["switch_kd"] = True

    return compact


def _adaptive_k(
    entropy: float,
    *,
    low_entropy_threshold: float,
    high_entropy_threshold: float,
    low_k: int,
    mid_k: int,
    high_k: int,
    max_k: int,
) -> int:
    if entropy < low_entropy_threshold:
        return min(max_k, low_k)

    if entropy < high_entropy_threshold:
        return min(max_k, mid_k)

    return min(max_k, high_k)


def _entropy_to_weight(entropy: float) -> float:
    return 1.0 / (1.0 + max(float(entropy), 0.0))


def _extract_generated_ids_and_scores(
    generation: Any,
    *,
    prompt_len: int,
    include_scores: bool,
):
    if include_scores:
        sequences = generation.sequences
        scores = list(generation.scores or [])
        generated_steps = len(scores)

        if generated_steps > 0:
            if sequences.shape[1] >= prompt_len + generated_steps:
                generated_ids = sequences[:, prompt_len : prompt_len + generated_steps]
            else:
                generated_ids = sequences[:, -generated_steps:]
        else:
            generated_ids = sequences[:, prompt_len:] if sequences.shape[1] > prompt_len else sequences

        return generated_ids, scores

    sequences = generation
    if sequences.shape[1] > prompt_len:
        generated_ids = sequences[:, prompt_len:]
    else:
        generated_ids = sequences

    return generated_ids, []


def _mock_answer(sample: VlmSample) -> str:
    if sample.answer:
        return sample.answer

    if sample.task == "parsing":
        return json.dumps(
            {
                "focused_element": "mock settings",
                "elements": [
                    {
                        "label": "mock icon",
                        "type": "app_icon",
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "label": "mock settings",
                        "type": "button",
                        "bbox": [120, 0, 220, 100],
                    },
                ],
            },
            ensure_ascii=False,
        )

    if sample.task == "grounding":
        return json.dumps(
            {
                "label": sample.target_label or "target",
                "type": sample.target_type or "object",
                "bbox": [0, 0, 100, 100],
            },
            ensure_ascii=False,
        )

    return f"mock answer for {sample.task}"
