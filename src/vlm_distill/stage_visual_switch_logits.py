from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config_schema import (
    PipelineConfig,
    format_prompt,
    resolve_label_path,
    resolve_switch_logits_path,
    resolve_teacher_logits_path,
)
from .data_manifest import VlmSample, read_jsonl, validate_manifest, write_jsonl
from .device_utils import (
    batch_to_device,
    ensure_stage_uses_cuda,
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
        if self._is_mock_mode():
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        student_model_path = resolve_model_path(self.config.student.model_name)
        teacher_model_path = resolve_model_path(self.config.teacher.model_name)
        teacher_requested_device_map = resolve_requested_device_map(
            self.config.teacher.device_map,
            quantization=getattr(self.config.teacher, "quantization", "none"),
            role="teacher",
        )
        self._torch = torch
        self._student_processor = AutoProcessor.from_pretrained(
            student_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._teacher_processor = AutoProcessor.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
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
        components = self._components()
        self._student_input_device = select_model_input_device(
            self._student_model,
            preferred_modules=(components.student_vision, getattr(self._student_model, "visual", None)),
            label="Switch logits student",
        )
        self._teacher_text_device = select_model_input_device(
            self._teacher_model,
            preferred_modules=(components.teacher_token_embedding, components.teacher_lm),
            label="Switch logits teacher",
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

    def generate_for_sample(self, sample: VlmSample) -> dict[str, Any]:
        if self._is_mock_mode():
            return self._mock_generate_for_sample(sample)
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
            teacher_inputs = self._teacher_text_inputs(prompt)
            switched_embeds, attention_mask = self._splice_visual_embeds(
                teacher_inputs=teacher_inputs,
                projected_visual=projected_visual,
            )
            switch_logits = self._teacher_lm_forward(
                inputs_embeds=switched_embeds,
                attention_mask=attention_mask,
            )
            cached_logits = _compact_adaptive_sequence_logits(
                switch_logits,
                base_k=int(self.config.distillation.dbild_top_k),
                max_cached_logits_vocab=self.config.distillation.max_cached_logits_vocab,
                temperature=float(self.config.distillation.kd_temperature),
            )

        field = self.config.distillation.switch_logits_field
        return {
            field: cached_logits,
            f"{field}_format": "switch_kd",
            f"{field}_prompt_len": int(teacher_inputs["input_ids"].shape[1]),
            f"{field}_vocab_size": int(switch_logits.shape[-1]),
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
        }

    def _is_mock_mode(self) -> bool:
        teacher_backend = (self.config.teacher.backend or "").lower()
        student_name = (self.config.student.model_name or "").lower()
        return teacher_backend == "mock" or student_name.startswith("mock-")

    def _mock_generate_for_sample(self, sample: VlmSample) -> dict[str, Any]:
        field = self.config.distillation.switch_logits_field
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=sample.query,
            target_label=sample.target_label,
            target_type=sample.target_type,
            task=sample.task,
        )
        prompt_len = max(1, len(prompt.split()))
        seq_len = max(2, min(6, prompt_len // 2))
        vocab_size = 32
        base_k = int(self.config.distillation.dbild_top_k)
        max_k = min(vocab_size, max(2, min(base_k, 8)))

        indices = []
        values = []
        token_k = []
        entropy = []
        entropy_weight = []
        for step_index in range(seq_len):
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
            "shape": [1, seq_len, vocab_size],
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
            f"{field}_prompt_len": int(prompt_len),
            f"{field}_vocab_size": vocab_size,
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
        }

    def _components(self) -> VSDComponents:
        distill = self.config.distillation
        student_vision = _resolve_component(
            self._student_model,
            distill.student_vision_path,
            _STUDENT_VISION_CANDIDATES,
            "student vision encoder",
        )
        student_projector = _resolve_component(
            self._student_model,
            distill.student_projector_path,
            _STUDENT_PROJECTOR_CANDIDATES,
            "student projector",
        )
        teacher_lm = _resolve_component(
            self._teacher_model,
            distill.teacher_lm_path,
            _TEACHER_LM_CANDIDATES,
            "teacher LLM",
        )
        teacher_token_embedding = _resolve_component(
            self._teacher_model,
            distill.teacher_token_embedding_path,
            _TEACHER_EMBEDDING_CANDIDATES,
            "teacher token embedding",
        )
        teacher_lm_head = _resolve_optional_component(
            self._teacher_model,
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

    def _student_image_inputs(self, image):
        student_inputs = _processor_image_inputs(self._student_processor, image)
        return batch_to_device(student_inputs, self._student_input_device)

    def _student_visual_outputs(self, student_inputs):
        components = self._components()
        device = module_device(components.student_vision)
        vision_inputs = batch_to_device(student_inputs, device)
        vision_kwargs = _build_vision_forward_kwargs(components.student_vision, vision_inputs)
        outputs = components.student_vision(**vision_kwargs)
        return _first_tensor(outputs)

    def _student_projector_outputs(self, visual_outputs, student_inputs):
        components = self._components()
        projector_kwargs = _build_projector_forward_kwargs(
            components.student_projector,
            visual_outputs,
            student_inputs,
        )
        projected = components.student_projector(**projector_kwargs)
        return _first_tensor(projected)

    def _teacher_text_inputs(self, prompt: str):
        inputs = self._teacher_processor(text=prompt, return_tensors="pt")
        return batch_to_device(inputs, self._teacher_text_device)

    def _splice_visual_embeds(self, teacher_inputs, projected_visual):
        import torch

        components = self._components()
        input_ids = teacher_inputs["input_ids"]
        text_embeds = components.teacher_token_embedding(input_ids)
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
        components = self._components()
        lm_device = module_device(components.teacher_lm) or self._teacher_text_device
        inputs_embeds = inputs_embeds.to(lm_device)
        attention_mask = attention_mask.to(inputs_embeds.device)
        outputs = components.teacher_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            return outputs.logits
        hidden_states = _first_tensor(outputs)
        if components.teacher_lm_head is None:
            raise ValueError("Teacher LLM output has no logits and no teacher_lm_head_path was resolved.")
        return components.teacher_lm_head(hidden_states)

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

def create_visual_switch_dataset(config: PipelineConfig) -> Path:
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    output_path = resolve_switch_logits_path(config.data)
    base_rows = _load_switch_base_rows(config)
    rows_by_id = {str(row["id"]): row for row in base_rows}
    distiller = VisualSwitchDistiller(config)
    output_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = rows_by_id.get(sample.id, asdict(sample))
        row.update(distiller.generate_for_sample(sample))
        output_rows.append(row)
    write_jsonl(output_path, output_rows)
    return output_path


def _load_switch_base_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    candidate_paths = (
        resolve_label_path(config.data),
        resolve_teacher_logits_path(config.data),
        resolve_switch_logits_path(config.data),
    )
    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return read_jsonl(path)
    return []


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
