from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config_schema import PipelineConfig
from .data_manifest import VlmSample, read_jsonl, validate_manifest, write_jsonl
from .logits_cache_utils import compact_logits


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

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._torch = torch
        self._student_processor = AutoProcessor.from_pretrained(
            self.config.student.model_name,
            trust_remote_code=True,
        )
        self._teacher_processor = AutoProcessor.from_pretrained(
            self.config.teacher.model_name,
            trust_remote_code=True,
        )
        self._student_model = AutoModelForVision2Seq.from_pretrained(
            self.config.student.model_name,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self._teacher_model = AutoModelForVision2Seq.from_pretrained(
            self.config.teacher.model_name,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def generate_for_sample(self, sample: VlmSample) -> dict[str, Any]:
        if self._student_model is None or self._teacher_model is None:
            self.load()

        import torch
        from PIL import Image

        image_path = self.config.data.image_root / sample.image
        image = Image.open(image_path).convert("RGB")
        prompt = self.config.distillation.prompt_template.format(
            query=sample.query,
            target_label=sample.target_label or "target object",
            task=sample.task,
        )
        with torch.no_grad():
            student_visual = self._student_visual_outputs(image)
            projected_visual = self._student_projector_outputs(student_visual)
            teacher_inputs = self._teacher_text_inputs(prompt)
            switched_embeds, attention_mask = self._splice_visual_embeds(
                teacher_inputs=teacher_inputs,
                projected_visual=projected_visual,
            )
            switch_logits = self._teacher_lm_forward(
                inputs_embeds=switched_embeds,
                attention_mask=attention_mask,
            )
            cached_logits = compact_logits(
                switch_logits,
                self.config.distillation.max_cached_logits_vocab,
            )

        field = self.config.distillation.switch_logits_field
        return {
            field: cached_logits,
            f"{field}_format": "topk" if self.config.distillation.max_cached_logits_vocab else "dense",
            f"{field}_prompt_len": int(teacher_inputs["input_ids"].shape[1]),
            f"{field}_vocab_size": int(switch_logits.shape[-1]),
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

    def _student_visual_outputs(self, image):
        components = self._components()
        student_inputs = self._student_processor(images=image, return_tensors="pt")
        device = _module_device(components.student_vision)
        student_inputs = _move_batch(student_inputs, device)
        outputs = components.student_vision(**student_inputs)
        return _first_tensor(outputs)

    def _student_projector_outputs(self, visual_outputs):
        components = self._components()
        projected = components.student_projector(visual_outputs)
        return _first_tensor(projected)

    def _teacher_text_inputs(self, prompt: str):
        inputs = self._teacher_processor(text=prompt, return_tensors="pt")
        components = self._components()
        device = _module_device(components.teacher_token_embedding)
        return _move_batch(inputs, device)

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
        inputs_embeds = inputs_embeds.to(_module_device(components.teacher_lm))
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
    base_rows = read_jsonl(config.data.distill_path) if config.data.distill_path.exists() else []
    rows_by_id = {str(row["id"]): row for row in base_rows}
    distiller = VisualSwitchDistiller(config)
    output_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = rows_by_id.get(sample.id, asdict(sample))
        row.update(distiller.generate_for_sample(sample))
        output_rows.append(row)
    write_jsonl(config.data.distill_path, output_rows)
    return config.data.distill_path


def _resolve_component(model, configured_path: str | None, candidates: tuple[str, ...], label: str):
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
    return name in _INVOKABLE_COMPONENT_METHODS or name.startswith("get_")


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _move_batch(batch, device):
    if device is None:
        return batch
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


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
