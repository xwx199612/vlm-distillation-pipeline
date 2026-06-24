from __future__ import annotations

import gc
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config_schema import (
    PipelineConfig,
    format_prompt,
    resolve_label_path,
    resolve_switch_logits_path,
)
from .data_manifest import VlmSample, read_jsonl, validate_manifest
from .device_utils import (
    batch_to_device,
    ensure_stage_uses_cuda,
    get_module_by_path,
    module_device,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from .model_loading import resolve_attn_implementation, resolve_model_path


INACTIVE_LOGIT = -1.0e4


@dataclass
class VSDComponents:
    student_vision: object
    student_projector: object
    teacher_lm: object
    teacher_token_embedding: object
    teacher_lm_head: object | None = None


class VisualSwitchDistiller:
    """Generate VSD switch logits.

    Flow:
      student vision encoder -> student projector -> optional dimension aligner
      -> teacher LLM input embedding stream -> teacher LLM -> switch logits

    Component paths are configurable because VLM repositories expose different
    attribute names for their vision tower, multimodal projector, and language model.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._torch = None
        self._student_model = None
        self._teacher_model = None
        self._student_processor = None
        self._teacher_processor = None
        self._aligner = None
        self._student_input_device = None
        self._teacher_text_device = None

    def load(self) -> None:
        self.load_student_only()
        self.load_teacher_only()

    def load_student_only(self) -> None:
        if self._is_mock_mode():
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        student_model_path = resolve_model_path(self.config.student.model_name)
        self._torch = torch
        if self._student_processor is None:
            self._student_processor = AutoProcessor.from_pretrained(
                student_model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        if self._student_model is not None:
            return
        self._student_model = AutoModelForVLM.from_pretrained(
            student_model_path,
            **_build_vlm_load_kwargs(
                requested_device_map="auto",
                quantization=getattr(self.config.student, "quantization", "none"),
                torch_dtype=None,
                attn_implementation=self.config.student.attn_implementation,
                BitsAndBytesConfig=BitsAndBytesConfig,
            ),
        ).eval()
        self._student_input_device = select_model_input_device(
            self._student_model,
            preferred_modules=(
                get_module_by_path(self._student_model, "model.visual"),
                get_module_by_path(self._student_model, "visual"),
                get_module_by_path(self._student_model, "vision_tower"),
                get_module_by_path(self._student_model, "model.vision_tower"),
                get_module_by_path(self._student_model, "model.language_model.embed_tokens"),
            ),
            label="Switch logits student",
        )
        print_stage_model_debug(
            stage_label="Switch logits student",
            model_path=student_model_path,
            quantization_mode=getattr(self.config.student, "quantization", "none"),
            requested_device_map="auto",
            model=self._student_model,
            selected_input_device=self._student_input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Switch logits student",
            requested_device_map="auto",
            model=self._student_model,
            selected_input_device=self._student_input_device,
        )

    def unload_student(self) -> None:
        self._student_model = None
        self._student_input_device = None
        self._aligner = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def load_teacher_only(self) -> None:
        if self._is_mock_mode():
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        teacher_model_path = resolve_model_path(self.config.teacher.model_name)
        teacher_requested_device_map = resolve_requested_device_map(
            self.config.teacher.device_map,
            quantization=getattr(self.config.teacher, "quantization", "none"),
            role="teacher",
        )
        self._torch = torch
        if self._teacher_processor is None:
            self._teacher_processor = AutoProcessor.from_pretrained(
                teacher_model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        if self._teacher_model is not None:
            return
        self._teacher_model = AutoModelForVLM.from_pretrained(
            teacher_model_path,
            **_build_vlm_load_kwargs(
                requested_device_map=teacher_requested_device_map,
                quantization=getattr(self.config.teacher, "quantization", "none"),
                torch_dtype=getattr(self.config.teacher, "torch_dtype", None),
                attn_implementation=self.config.teacher.attn_implementation,
                BitsAndBytesConfig=BitsAndBytesConfig,
            ),
        ).eval()
        self._teacher_text_device = select_model_input_device(
            self._teacher_model,
            preferred_modules=(
                get_module_by_path(self._teacher_model, "model.language_model.embed_tokens"),
                get_module_by_path(self._teacher_model, "model.language_model"),
                get_module_by_path(self._teacher_model, "language_model"),
            ),
            label="Switch logits teacher",
        )
        print_stage_model_debug(
            stage_label="Switch logits teacher",
            model_path=teacher_model_path,
            quantization_mode=getattr(self.config.teacher, "quantization", "none"),
            requested_device_map=teacher_requested_device_map,
            model=self._teacher_model,
            selected_input_device=self._teacher_text_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Switch logits teacher",
            requested_device_map=teacher_requested_device_map,
            model=self._teacher_model,
            selected_input_device=self._teacher_text_device,
        )

    def generate_for_sample(self, sample: VlmSample, *, base_row: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._is_mock_mode():
            return self._mock_generate_for_sample(sample, base_row=base_row)
        if self._student_model is None or self._teacher_model is None:
            self.load()

        import torch
        from .vlm_batching import load_training_image

        image = load_training_image(
            self.config.data.image_root,
            sample.image,
            resize_mode=self.config.training.image_resize,
        )
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=sample.query,
            target_label=sample.target_label,
            target_type=sample.target_type,
            task=sample.task,
        )
        with torch.no_grad():
            student_inputs = self._student_image_inputs(image)
            student_visual = self._student_visual_outputs(student_inputs)
            projected_visual = self._student_projector_outputs(student_visual, student_inputs)
            return self._generate_for_sample_from_projected_visual(
                sample=sample,
                prompt=prompt,
                projected_visual=projected_visual,
                base_row=base_row,
            )

    def generate_student_visual_cache_for_sample(self, sample: VlmSample, cache_path: Path) -> None:
        if self._is_mock_mode():
            raise RuntimeError("Student visual cache generation is not supported in mock mode.")
        if self._student_model is None:
            self.load_student_only()

        from .vlm_batching import load_training_image

        image = load_training_image(
            self.config.data.image_root,
            sample.image,
            resize_mode=self.config.training.image_resize,
        )
        with self._torch.no_grad():
            student_inputs = self._student_image_inputs(image)
            student_visual = self._student_visual_outputs(student_inputs)
            projected_visual = self._student_projector_outputs(student_visual, student_inputs)
            projected_visual = projected_visual.detach()
            if self.config.distillation.keep_student_visual_cache_on_cpu:
                projected_visual = projected_visual.cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "id": str(sample.id),
                "projected_visual": projected_visual,
                "shape": list(projected_visual.shape),
                "dtype": str(projected_visual.dtype),
            },
            cache_path,
        )

    def generate_for_sample_from_visual_cache(
        self,
        sample: VlmSample,
        cache_path: Path,
        *,
        base_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._is_mock_mode():
            return self._mock_generate_for_sample(sample, base_row=base_row)
        if self._teacher_model is None:
            self.load_teacher_only()

        cached = self._torch.load(cache_path, map_location="cpu")
        projected_visual = cached["projected_visual"]
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=sample.query,
            target_label=sample.target_label,
            target_type=sample.target_type,
            task=sample.task,
        )
        with self._torch.no_grad():
            return self._generate_for_sample_from_projected_visual(
                sample=sample,
                prompt=prompt,
                projected_visual=projected_visual,
                base_row=base_row,
            )

    def _is_mock_mode(self) -> bool:
        teacher_backend = (self.config.teacher.backend or "").lower()
        student_name = (self.config.student.model_name or "").lower()
        return teacher_backend == "mock" or student_name.startswith("mock-")

    def _mock_generate_for_sample(self, sample: VlmSample, *, base_row: dict[str, Any] | None = None) -> dict[str, Any]:
        field = self.config.distillation.switch_logits_field
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=sample.query,
            target_label=sample.target_label,
            target_type=sample.target_type,
            task=sample.task,
        )
        text_prompt_len = max(1, len(prompt.split()))
        visual_token_count = int((base_row or {}).get("visual_token_count") or 0)
        teacher_tokens = _extract_teacher_tokens(base_row or {})
        answer_len = len(teacher_tokens) if teacher_tokens else max(2, min(6, text_prompt_len // 2))
        vocab_size = 32
        base_k = int(self.config.distillation.dbild_top_k)
        max_k = min(vocab_size, max(2, min(base_k, 8)))

        indices = []
        values = []
        token_k = []
        entropy = []
        entropy_weight = []
        for step_index in range(answer_len):
            peak_index = (step_index * 3) % vocab_size
            step_indices = [(peak_index + offset) % vocab_size for offset in range(max_k)]
            step_values = [5.0 - rank for rank in range(max_k)]
            step_entropy = 1.0
            indices.append(step_indices)
            values.append(step_values)
            token_k.append(min(max_k, max(2, base_k)))
            entropy.append(step_entropy)
            entropy_weight.append(_entropy_to_weight(step_entropy))

        cached_logits = {
            "indices": [indices],
            "values": [values],
            "shape": [1, answer_len, vocab_size],
            "vocab_size": vocab_size,
            "adaptive": True,
            "token_k": [token_k],
            "entropy": [entropy],
            "entropy_weight": [entropy_weight],
            "switch_kd": True,
        }
        return {
            field: cached_logits,
            f"{field}_format": "switch_kd",
            f"{field}_prompt_len": 0,
            f"{field}_vocab_size": vocab_size,
            f"{field}_aligned_to_answer": True,
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
            "teacher_tokens": teacher_tokens,
        }

    def _components(self) -> VSDComponents:
        distill = self.config.distillation
        student_vision = self._student_component(
            distill.student_vision_path,
            _STUDENT_VISION_CANDIDATES,
            "student vision encoder",
        )
        student_projector = self._student_component(
            distill.student_projector_path,
            _STUDENT_PROJECTOR_CANDIDATES,
            "student projector",
        )
        teacher_lm = self._teacher_component(
            distill.teacher_lm_path,
            _TEACHER_LM_CANDIDATES,
            "teacher LLM",
        )
        teacher_token_embedding = self._teacher_component(
            distill.teacher_token_embedding_path,
            _TEACHER_EMBEDDING_CANDIDATES,
            "teacher token embedding",
        )
        teacher_lm_head = self._teacher_optional_component(
            distill.teacher_lm_head_path,
            _TEACHER_LM_HEAD_CANDIDATES,
        )
        return VSDComponents(
            student_vision=student_vision,
            student_projector=student_projector,
            teacher_lm=teacher_lm,
            teacher_token_embedding=teacher_token_embedding,
            teacher_lm_head=teacher_lm_head,
        )

    def _student_component(self, configured_path: str | None, candidates: tuple[str, ...], label: str):
        if self._student_model is None:
            raise RuntimeError(f"Student model is not loaded while resolving {label}.")
        return _resolve_component(self._student_model, configured_path, candidates, label)

    def _teacher_component(self, configured_path: str | None, candidates: tuple[str, ...], label: str):
        if self._teacher_model is None:
            raise RuntimeError(f"Teacher model is not loaded while resolving {label}.")
        return _resolve_component(self._teacher_model, configured_path, candidates, label)

    def _teacher_optional_component(self, configured_path: str | None, candidates: tuple[str, ...]):
        if self._teacher_model is None:
            return None
        return _resolve_optional_component(self._teacher_model, configured_path, candidates)

    def _student_image_inputs(self, image):
        student_inputs = _processor_image_inputs(self._student_processor, image)
        return batch_to_device(student_inputs, self._student_input_device)

    def _student_visual_outputs(self, student_inputs):
        student_vision = self._student_component(
            self.config.distillation.student_vision_path,
            _STUDENT_VISION_CANDIDATES,
            "student vision encoder",
        )
        device = module_device(student_vision)
        vision_inputs = batch_to_device(student_inputs, device)
        vision_kwargs = _build_vision_forward_kwargs(student_vision, vision_inputs)
        outputs = student_vision(**vision_kwargs)
        return _first_tensor(outputs)

    def _student_projector_outputs(self, visual_outputs, student_inputs):
        student_projector = self._student_component(
            self.config.distillation.student_projector_path,
            _STUDENT_PROJECTOR_CANDIDATES,
            "student projector",
        )
        projector_kwargs = _build_projector_forward_kwargs(
            student_projector,
            visual_outputs,
            student_inputs,
        )
        projected = student_projector(**projector_kwargs)
        return _first_tensor(projected)

    def _teacher_text_inputs(self, prompt: str):
        inputs = self._teacher_processor(text=prompt, return_tensors="pt")
        return batch_to_device(inputs, self._teacher_text_device)

    def _splice_visual_embeds(self, teacher_inputs, projected_visual):
        import torch

        teacher_token_embedding = self._teacher_component(
            self.config.distillation.teacher_token_embedding_path,
            _TEACHER_EMBEDDING_CANDIDATES,
            "teacher token embedding",
        )
        input_ids = teacher_inputs["input_ids"]
        text_embeds = teacher_token_embedding(input_ids)
        projected_visual = projected_visual.to(text_embeds.device, dtype=text_embeds.dtype)
        projected_visual = _ensure_batch_sequence(projected_visual)
        projected_visual = self._align_visual_dim(projected_visual, text_embeds.shape[-1])

        placeholder_id = self._visual_placeholder_id()
        if placeholder_id is not None:
            mask = input_ids.eq(placeholder_id)
            if mask.any():
                return _replace_placeholder_embeds(
                    text_embeds=text_embeds,
                    attention_mask=teacher_inputs.get("attention_mask"),
                    placeholder_mask=mask,
                    visual_embeds=projected_visual,
                )

        attention_mask = teacher_inputs.get("attention_mask")
        visual_mask = torch.ones(
            projected_visual.shape[:2],
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
            device=text_embeds.device,
        )
        if attention_mask is None:
            attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.long, device=text_embeds.device)
        switched_embeds = torch.cat([projected_visual, text_embeds], dim=1)
        switched_mask = torch.cat([visual_mask, attention_mask.to(text_embeds.device)], dim=1)
        return switched_embeds, switched_mask

    def _teacher_lm_forward(self, inputs_embeds, attention_mask):
        teacher_lm = self._teacher_component(
            self.config.distillation.teacher_lm_path,
            _TEACHER_LM_CANDIDATES,
            "teacher LLM",
        )
        teacher_lm_head = self._teacher_optional_component(
            self.config.distillation.teacher_lm_head_path,
            _TEACHER_LM_HEAD_CANDIDATES,
        )
        lm_device = module_device(teacher_lm) or self._teacher_text_device
        inputs_embeds = inputs_embeds.to(lm_device)
        attention_mask = attention_mask.to(inputs_embeds.device)
        outputs = teacher_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            return outputs.logits
        hidden_states = _first_tensor(outputs)
        if teacher_lm_head is None:
            raise ValueError("Teacher LLM output has no logits and no teacher_lm_head_path was resolved.")
        return teacher_lm_head(hidden_states)

    def _align_visual_dim(self, visual_embeds, teacher_dim: int):
        import torch

        visual_dim = visual_embeds.shape[-1]
        if visual_dim == teacher_dim:
            return visual_embeds
        if self._aligner is None:
            self._aligner = torch.nn.Linear(visual_dim, teacher_dim, bias=False).to(
                visual_embeds.device,
                dtype=visual_embeds.dtype,
            )
            torch.nn.init.xavier_uniform_(self._aligner.weight)
            self._aligner.requires_grad_(False)
        return self._aligner(visual_embeds)

    def _visual_placeholder_id(self) -> int | None:
        placeholder = self.config.distillation.visual_token_placeholder
        tokenizer = getattr(self._teacher_processor, "tokenizer", self._teacher_processor)
        try:
            token_id = tokenizer.convert_tokens_to_ids(placeholder)
        except Exception:
            return None
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            return None
        return int(token_id)

    def _generate_for_sample_from_projected_visual(
        self,
        *,
        sample: VlmSample,
        prompt: str,
        projected_visual,
        base_row: dict[str, Any] | None = None,
    ):
        teacher_answer = str((base_row or {}).get("teacher_answer") or "").strip()
        full_text = _join_prompt_and_answer(prompt, teacher_answer)
        prompt_teacher_inputs = self._teacher_text_inputs(prompt)
        prompt_embeds, _ = self._splice_visual_embeds(
            teacher_inputs=prompt_teacher_inputs,
            projected_visual=projected_visual,
        )
        teacher_inputs = self._teacher_text_inputs(full_text)
        switched_embeds, attention_mask = self._splice_visual_embeds(
            teacher_inputs=teacher_inputs,
            projected_visual=projected_visual,
        )
        switch_logits = self._teacher_lm_forward(
            inputs_embeds=switched_embeds,
            attention_mask=attention_mask,
        )
        prompt_len = int(prompt_embeds.shape[1])
        teacher_tokens = _extract_teacher_tokens(base_row or {})
        if teacher_tokens:
            prompt_len = int(switch_logits.shape[1]) - len(teacher_tokens)
        answer_len = len(teacher_tokens)
        if answer_len <= 0:
            raise ValueError(f"Switch logits require teacher_tokens for answer-only slicing. id={sample.id}")
        answer_logits = switch_logits[:, prompt_len - 1 : prompt_len - 1 + answer_len, :]
        if int(answer_logits.shape[1]) != answer_len:
            raise ValueError(
                "Switch logits answer slice length mismatch. "
                f"id={sample.id}, answer_logits_len={int(answer_logits.shape[1])}, "
                f"teacher_tokens_len={answer_len}, prompt_len={prompt_len}"
            )
        cached_logits = _compact_adaptive_sequence_logits(
            answer_logits,
            base_k=int(self.config.distillation.dbild_top_k),
            max_cached_logits_vocab=self.config.distillation.max_cached_logits_vocab,
            temperature=float(self.config.distillation.kd_temperature),
        )
        field = self.config.distillation.switch_logits_field
        return {
            field: cached_logits,
            f"{field}_format": "switch_kd",
            f"{field}_prompt_len": 0,
            f"{field}_vocab_size": int(answer_logits.shape[-1]),
            f"{field}_aligned_to_answer": True,
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
            "teacher_tokens": teacher_tokens,
        }

def create_visual_switch_dataset(config: PipelineConfig) -> Path:
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    output_path = resolve_switch_logits_path(config.data)
    completed = _load_completed_ids(output_path, field_name=config.distillation.switch_logits_field)
    if completed["invalid_count"]:
        _rewrite_valid_completed_rows(output_path, field_name=config.distillation.switch_logits_field)
    completed_ids = completed["ids"]
    total = len(samples)
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Switch logits samples: total={total}, completed={len(completed_ids)}, "
        f"completed_valid_count={completed['valid_count']}, "
        f"completed_invalid_count={completed['invalid_count']}, "
        f"first_invalid_row_keys={completed['first_invalid_keys']}, "
        f"pending={len(pending_samples)}, output={output_path}"
    )

    if not pending_samples:
        print("No pending samples. Existing switch logits output is already complete for this manifest.")
        return output_path

    base_rows = _load_switch_base_rows(config)
    rows_by_id = {str(row["id"]): row for row in base_rows}
    distiller = VisualSwitchDistiller(config)
    use_student_visual_cache = bool(config.distillation.switch_cache_student_visual)

    if use_student_visual_cache and not distiller._is_mock_mode():
        cache_dir = _student_visual_cache_dir(config, output_path)
        print(
            f"[switch-logits] student visual cache phase starting: "
            f"pending={len(pending_samples)} cache_dir={cache_dir}"
        )
        distiller.load_student_only()
        cached_now = 0
        for sample in pending_samples:
            cache_path = _student_visual_cache_path(cache_dir, sample)
            if cache_path.exists():
                print(f"[switch-logits][student-cache] sample_id={sample.id} cache_exists path={cache_path}")
                continue
            started = time.perf_counter()
            distiller.generate_student_visual_cache_for_sample(sample, cache_path)
            cached_now += 1
            elapsed = time.perf_counter() - started
            print(
                f"[switch-logits][student-cache] sample_id={sample.id} cached "
                f"elapsed_seconds={elapsed:.2f} cache_path={cache_path}"
            )
        print("[switch-logits] unloading student model before teacher phase.")
        distiller.unload_student()
        print("[switch-logits] student unloaded; gc.collect() and torch.cuda.empty_cache() completed.")
        print("[switch-logits] teacher phase starting.")
        distiller.load_teacher_only()

    completed_now = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in pending_samples:
            started = time.perf_counter()
            row = dict(rows_by_id.get(sample.id, asdict(sample)))
            if not str(row.get("teacher_answer") or "").strip():
                raise ValueError(
                    "Switch logits generation requires teacher_answer so the full "
                    f"prompt-plus-answer sequence can be cached. id={sample.id}, image={sample.image}, "
                    f"row_keys={sorted(row.keys())}"
                )
            _validate_unified_teacher_base_row(
                row,
                require_logits=bool(config.distillation.teacher_logits),
            )
            if use_student_visual_cache and not distiller._is_mock_mode():
                cache_path = _student_visual_cache_path(cache_dir, sample)
                row.update(distiller.generate_for_sample_from_visual_cache(sample, cache_path, base_row=row))
            else:
                row.update(distiller.generate_for_sample(sample, base_row=row))
            if "teacher_tokens" not in row:
                row["teacher_tokens"] = _extract_teacher_tokens(row)
            _validate_switch_logits_row(
                row,
                field_name=config.distillation.switch_logits_field,
                visual_token_placeholder=config.distillation.visual_token_placeholder,
            )
            if completed_now == 0:
                _print_first_switch_logits_debug(row, field_name=config.distillation.switch_logits_field)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed_now += 1
            elapsed = time.perf_counter() - started
            total_done = len(completed_ids) + completed_now
            pending = total - total_done
            print(
                "[switch-logits] "
                f"total={total} completed={total_done} pending={pending} "
                f"current_sample_id={sample.id} elapsed_seconds_per_sample={elapsed:.2f} "
                f"output_path={output_path}"
            )
    return output_path


def _load_completed_ids(path: Path, *, field_name: str) -> dict[str, Any]:
    if not path.exists():
        return {"ids": set(), "valid_count": 0, "invalid_count": 0, "first_invalid_keys": None}

    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid_keys: list[str] | None = None
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        if not _is_valid_switch_logits_row(row, field_name=field_name):
            invalid_count += 1
            if first_invalid_keys is None:
                first_invalid_keys = sorted(str(key) for key in row.keys())
            continue
        completed_ids.add(str(sample_id))
        valid_count += 1
    return {
        "ids": completed_ids,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "first_invalid_keys": first_invalid_keys,
    }


def _rewrite_valid_completed_rows(path: Path, *, field_name: str) -> None:
    valid_rows = [row for row in read_jsonl(path) if _is_valid_switch_logits_row(row, field_name=field_name)]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[switch-logits] pruned invalid existing rows from {path}; "
        f"remaining_valid_rows={len(valid_rows)}"
    )


def _extract_teacher_tokens(row: dict[str, Any]) -> list[int]:
    tokens = row.get("teacher_tokens")
    if isinstance(tokens, list) and (not tokens or not isinstance(tokens[0], list)):
        return [int(value) for value in tokens]
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        return [int(value) for value in tokens[0]]
    generated = row.get("teacher_generated_ids")
    if isinstance(generated, list) and generated and isinstance(generated[0], list):
        return [int(value) for value in generated[0]]
    if isinstance(generated, list):
        return [int(value) for value in generated]
    return []


def _join_prompt_and_answer(prompt: str, answer: str) -> str:
    if not answer:
        return prompt
    separator = "" if prompt.endswith((" ", "\n")) else " "
    return f"{prompt}{separator}{answer}".strip()


def _validate_switch_logits_row(
    row: dict[str, Any],
    *,
    field_name: str,
    visual_token_placeholder: str,
) -> None:
    payload = row.get(field_name)
    if not _is_valid_logits_payload(payload):
        raise ValueError(
            "Switch logits output row is missing a valid logits payload. "
            f"id={row.get('id')}, image={row.get('image')}, row_keys={sorted(row.keys())}"
        )
    shape = payload.get("shape")
    raw_seq_len = int(shape[1])
    teacher_tokens = _extract_teacher_tokens(row)
    answer_len = len(teacher_tokens)
    difference = raw_seq_len - answer_len
    if answer_len > 0 and difference != 0:
        raise ValueError(
            "Switch logits answer-only alignment is invalid. "
            f"id={row.get('id')}, image={row.get('image')}, raw_seq_len={raw_seq_len}, "
            f"teacher_tokens_len={answer_len}, difference={difference}, "
            f"visual_token_placeholder={visual_token_placeholder}, switch_logits_shape={shape}"
        )
    if answer_len > 0 and row.get(f"{field_name}_aligned_to_answer") is not True:
        raise ValueError(f"Switch logits row is missing {field_name}_aligned_to_answer=true. id={row.get('id')}")


def _is_valid_switch_logits_row(row: dict[str, Any], *, field_name: str) -> bool:
    if not _is_valid_logits_payload(row.get(field_name)):
        return False
    teacher_tokens = _extract_teacher_tokens(row)
    if not teacher_tokens:
        return True
    payload = row[field_name]
    try:
        raw_seq_len = int(payload["shape"][1])
    except (TypeError, ValueError, KeyError, IndexError):
        return False
    return raw_seq_len == len(teacher_tokens) and row.get(f"{field_name}_aligned_to_answer") is True


def _print_first_switch_logits_debug(row: dict[str, Any], *, field_name: str) -> None:
    payload = row[field_name]
    shape = payload["shape"]
    teacher_tokens = _extract_teacher_tokens(row)
    effective_len = int(shape[1])
    top_k_first_token = None
    token_k = payload.get("token_k")
    if isinstance(token_k, list) and token_k and isinstance(token_k[0], list) and token_k[0]:
        top_k_first_token = token_k[0][0]
    print("Switch logits first sample debug:")
    print(f"  raw_seq_len: {shape[1]}")
    print("  switch_logits_prompt_len: 0")
    print(f"  teacher_tokens_len: {len(teacher_tokens)}")
    print(f"  raw_seq_len_minus_prompt_len: {effective_len}")
    print(f"  vocab_size: {payload.get('vocab_size')}")
    print(f"  top_k_first_token: {top_k_first_token}")
    print(f"  alignment_validation_passed: {effective_len == len(teacher_tokens) if teacher_tokens else True}")


def _is_valid_logits_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if not all(key in payload for key in ("indices", "values", "vocab_size", "shape")):
        return False
    return _nested_shape(payload.get("indices")) == _nested_shape(payload.get("values")) != ()


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return ()
    first_shape = _nested_shape(value[0])
    for item in value[1:]:
        if _nested_shape(item) != first_shape:
            return ()
    return (len(value), *first_shape)


def _load_switch_base_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    path = resolve_label_path(config.data)
    if path.exists():
        return read_jsonl(path)
    return []


def _validate_unified_teacher_base_row(row: dict[str, Any], *, require_logits: bool) -> None:
    from .teacher_validation import validate_teacher_row

    valid, reason = validate_teacher_row(row, require_teacher_logits=require_logits)
    if not valid:
        raise ValueError(
            "Switch logits generation requires a schema-valid unified teacher row "
            f"from data.label_path. id={row.get('id')}, reason={reason}, row_keys={sorted(row.keys())}"
        )


def _student_visual_cache_dir(config: PipelineConfig, output_path: Path) -> Path:
    configured = config.distillation.student_visual_cache_dir
    if configured is not None:
        return configured
    return output_path.parent / f"{output_path.stem}_student_visual_cache"


def _student_visual_cache_path(cache_dir: Path, sample: VlmSample) -> Path:
    sample_key = quote(str(sample.id), safe="-_.")
    return cache_dir / f"{sample_key}.pt"


def _resolve_component(model, configured_path: str | None, candidates: tuple[str, ...], label: str):
    if configured_path == "__identity__":
        return _identity_projector
    if configured_path:
        return _get_nested_attr(model, configured_path)
    for path in candidates:
        try:
            return _get_nested_attr(model, path)
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not resolve {label}. Set the matching path in distillation config. "
        f"Tried: {', '.join(candidates)}"
    )


def _resolve_optional_component(model, configured_path: str | None, candidates: tuple[str, ...]):
    if configured_path:
        return _get_nested_attr(model, configured_path)
    for path in candidates:
        try:
            return _get_nested_attr(model, path)
        except AttributeError:
            continue
    return None


def _get_nested_attr(obj, path: str):
    parts = path.split(".")
    current = obj
    for index, part in enumerate(parts):
        current = getattr(current, part)
        if index == len(parts) - 1 and _should_invoke_component(current, part):
            current = current()
    return current


def _should_invoke_component(value, name: str) -> bool:
    if not callable(value) or isinstance(value, type):
        return False
    if name not in _INVOKABLE_COMPONENT_METHODS and not name.startswith("get_"):
        return False
    try:
        parameters = inspect.signature(value).parameters
    except (TypeError, ValueError):
        return False
    required = [
        param
        for param in parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and param.default is inspect._empty
    ]
    return len(required) == 0


def _processor_image_inputs(processor, image):
    try:
        return processor(images=image, return_tensors="pt")
    except TypeError:
        pass
    except Exception as exc:
        if "NoneType" not in str(exc):
            raise

    # Qwen2.x-VL processors expect a paired text input so they can place image tokens.
    try:
        return processor(
            text=" ",
            images=image,
            return_tensors="pt",
        )
    except TypeError:
        return processor(
            text=[" "],
            images=[image],
            return_tensors="pt",
        )


def _build_vlm_load_kwargs(
    *,
    requested_device_map: str,
    quantization: str | None,
    torch_dtype: str | None,
    attn_implementation: str,
    BitsAndBytesConfig,
) -> dict[str, Any]:
    import torch

    model_kwargs: dict[str, Any] = {
        "device_map": requested_device_map,
        "trust_remote_code": True,
        "local_files_only": True,
        "offload_buffers": True,
        "attn_implementation": resolve_attn_implementation(attn_implementation),
    }

    mode = (quantization or "none").lower()
    if mode == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif mode == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif torch_dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif torch_dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32

    return model_kwargs


def _build_vision_forward_kwargs(vision_module, student_inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(vision_module.forward).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "hidden_states" in parameters and "grid_thw" in parameters:
        kwargs: dict[str, Any] = {}
        if "pixel_values" in student_inputs:
            kwargs["hidden_states"] = student_inputs["pixel_values"]
        if "image_grid_thw" in student_inputs:
            kwargs["grid_thw"] = student_inputs["image_grid_thw"]
        if kwargs:
            return kwargs

    return {
        key: value
        for key, value in student_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }


def _compact_adaptive_sequence_logits(
    logits,
    *,
    base_k: int,
    max_cached_logits_vocab: int | None,
    temperature: float,
):
    import torch

    logits = logits.detach().float().cpu()
    if logits.ndim != 3:
        raise ValueError(f"Expected switch logits to have shape [batch, seq, vocab], got {tuple(logits.shape)}")

    batch_size, seq_len, vocab_size = logits.shape
    base_k = max(1, int(base_k))
    low_k = max(1, base_k // 4)
    mid_k = base_k
    high_k = base_k * 2
    if max_cached_logits_vocab is not None:
        high_k = min(high_k, int(max_cached_logits_vocab))
    max_k = min(vocab_size, max(low_k, mid_k, high_k))

    low_entropy_threshold = 1.0
    high_entropy_threshold = 2.5
    safe_temperature = max(float(temperature), 1e-6)

    scaled_logits = logits / safe_temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    top_values, top_indices = torch.topk(logits, k=max_k, dim=-1)
    token_k = torch.empty((batch_size, seq_len), dtype=torch.long)
    entropy_weight = torch.empty((batch_size, seq_len), dtype=torch.float32)

    for batch_index in range(batch_size):
        for step_index in range(seq_len):
            entropy_value = float(entropy[batch_index, step_index].item())
            active_k = _adaptive_k(
                entropy_value,
                low_entropy_threshold=low_entropy_threshold,
                high_entropy_threshold=high_entropy_threshold,
                low_k=low_k,
                mid_k=mid_k,
                high_k=high_k,
                max_k=max_k,
            )
            token_k[batch_index, step_index] = active_k
            entropy_weight[batch_index, step_index] = _entropy_to_weight(entropy_value)
            if active_k < max_k:
                top_values[batch_index, step_index, active_k:] = INACTIVE_LOGIT

    return {
        "indices": top_indices.tolist(),
        "values": top_values.tolist(),
        "shape": [batch_size, seq_len, vocab_size],
        "vocab_size": int(vocab_size),
        "adaptive": True,
        "token_k": token_k.tolist(),
        "entropy": entropy.tolist(),
        "entropy_weight": entropy_weight.tolist(),
        "switch_kd": True,
        "k_policy": {
            "low_entropy_threshold": low_entropy_threshold,
            "high_entropy_threshold": high_entropy_threshold,
            "low_k": int(low_k),
            "mid_k": int(mid_k),
            "high_k": int(high_k),
            "max_k": int(max_k),
        },
    }


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


def _build_projector_forward_kwargs(projector_module, visual_outputs, student_inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(projector_module).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "pixel_values" in parameters:
        kwargs: dict[str, Any] = {
            "pixel_values": student_inputs["pixel_values"],
        }
        if "image_grid_thw" in parameters and "image_grid_thw" in student_inputs:
            kwargs["image_grid_thw"] = student_inputs["image_grid_thw"]
        return kwargs

    if "x" in parameters:
        return {"x": visual_outputs}

    if "hidden_states" in parameters:
        return {"hidden_states": visual_outputs}

    return {"x": visual_outputs}


def _first_tensor(outputs):
    import torch

    if torch.is_tensor(outputs):
        return outputs
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)):
        for value in outputs:
            if torch.is_tensor(value):
                return value
    raise ValueError(f"Could not find tensor output in {type(outputs)!r}")


def _ensure_batch_sequence(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4:
        return tensor.flatten(1, 2)
    raise ValueError(f"Expected visual tensor with 2-4 dims, got shape {tuple(tensor.shape)}")


def _replace_placeholder_embeds(text_embeds, attention_mask, placeholder_mask, visual_embeds):
    import torch

    batch_embeds = []
    batch_masks = []
    if attention_mask is None:
        attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.long, device=text_embeds.device)
    for batch_idx in range(text_embeds.shape[0]):
        placeholder_positions = placeholder_mask[batch_idx].nonzero(as_tuple=False).flatten()
        if placeholder_positions.numel() == 0:
            batch_embeds.append(torch.cat([visual_embeds[batch_idx], text_embeds[batch_idx]], dim=0))
            batch_masks.append(
                torch.cat(
                    [
                        torch.ones(visual_embeds.shape[1], dtype=attention_mask.dtype, device=text_embeds.device),
                        attention_mask[batch_idx],
                    ],
                    dim=0,
                )
            )
            continue
        first = int(placeholder_positions[0])
        batch_embeds.append(
            torch.cat(
                [
                    text_embeds[batch_idx, :first],
                    visual_embeds[batch_idx],
                    text_embeds[batch_idx, first + 1 :],
                ],
                dim=0,
            )
        )
        batch_masks.append(
            torch.cat(
                [
                    attention_mask[batch_idx, :first],
                    torch.ones(visual_embeds.shape[1], dtype=attention_mask.dtype, device=text_embeds.device),
                    attention_mask[batch_idx, first + 1 :],
                ],
                dim=0,
            )
        )
    return _pad_sequence(batch_embeds), _pad_sequence(batch_masks, padding_value=0)


def _pad_sequence(items, padding_value=0):
    import torch

    return torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=padding_value)


def _identity_projector(x):
    return x


_INVOKABLE_COMPONENT_METHODS = frozenset(
    {
        "get_input_embeddings",
        "get_output_embeddings",
        "get_decoder",
        "get_encoder",
    }
)

_STUDENT_VISION_CANDIDATES = (
    "vision_tower",
    "model.vision_tower",
    "model.vision_model",
    "vision_model",
    "visual",
    "model.visual",
)

_STUDENT_PROJECTOR_CANDIDATES = (
    "connector",
    "model.connector",
    "multi_modal_projector",
    "model.multi_modal_projector",
    "mm_projector",
    "model.mm_projector",
    "visual_projector",
    "model.visual_projector",
    "projector",
    "model.projector",
)

_TEACHER_LM_CANDIDATES = (
    "language_model",
    "model.language_model",
    "model",
)

_TEACHER_EMBEDDING_CANDIDATES = (
    "get_input_embeddings",
    "language_model.model.embed_tokens",
    "model.language_model.model.embed_tokens",
    "model.embed_tokens",
    "model.model.embed_tokens",
    "language_model.get_input_embeddings",
)

_TEACHER_LM_HEAD_CANDIDATES = (
    "lm_head",
    "language_model.lm_head",
    "model.language_model.lm_head",
)
