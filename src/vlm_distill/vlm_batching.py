from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EncodedVlmSample:
    model_inputs: dict[str, Any]
    prompt_token_len: int


def load_training_image(image_root: Path, image_path: str, *, resize_mode: str = "original") -> Image.Image:
    path = image_root / image_path
    image = Image.open(path).convert("RGB")
    return _resize_training_image(image, resize_mode)


def encode_vlm_training_sample(
    processor,
    *,
    image: Image.Image,
    prompt: str,
    target: str,
    max_length: int,
    mask_prompt_labels: bool = True,
) -> EncodedVlmSample:
    """Encode one image+prompt+target sample for causal VLM fine-tuning."""
    prompt_text = _build_training_prompt_text(processor, prompt.strip())
    target_text = target.strip()
    separator = "" if prompt_text.endswith((" ", "\n")) else " "
    full_text = f"{prompt_text}{separator}{target_text}".strip()

    common_kwargs = {"return_tensors": "pt", "truncation": True, "max_length": max_length}
    full_inputs = _processor_call(processor, image=image, text=full_text, **common_kwargs)
    prompt_inputs = _processor_call(processor, image=image, text=prompt_text, **common_kwargs)

    prompt_token_len = int(prompt_inputs["input_ids"].shape[1])
    model_inputs = {key: value.squeeze(0) for key, value in full_inputs.items()}
    labels = model_inputs["input_ids"].clone()
    if mask_prompt_labels:
        labels[:prompt_token_len] = -100
    model_inputs["labels"] = labels
    return EncodedVlmSample(model_inputs=model_inputs, prompt_token_len=prompt_token_len)


def build_vlm_data_collator(processor, *, logits_fields=("teacher_logits", "switch_logits")):
    pad_token_id = _resolve_pad_token_id(processor)
    return VlmDataCollator(pad_token_id=pad_token_id, logits_fields=logits_fields)


class VlmDataCollator:
    """Pad multimodal features; keep cached logits as per-sample payloads."""
    _SKIP_KEYS = frozenset({"prompt_token_len"})

    def __init__(
        self,
        pad_token_id: int = 0,
        logits_fields: tuple[str, str] = ("teacher_logits", "switch_logits"),
    ):
        self.pad_token_id = pad_token_id
        self.logits_fields = logits_fields

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        
        logits_payload = {field: [feature.pop(field, None) for feature in features] for field in self.logits_fields}
        
        prompt_token_lens = [int(feature.pop("prompt_token_len", 0)) for feature in features]
        metadata: dict[str, Any] = {}
        for feature in features:
            for key in list(feature.keys()):
                if key.endswith("_prompt_len") or key.endswith("_vocab_size"):
                    metadata.setdefault(key, feature.pop(key))

        tensor_keys = sorted(
            {
                key
                for feature in features
                for key, value in feature.items()
                if key not in self._SKIP_KEYS and torch.is_tensor(value)
            }
        )
        batch: dict[str, Any] = {}
        for key in tensor_keys:
            values = [feature[key] for feature in features]
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=-100,
                )
                continue
            if key in {"input_ids", "attention_mask"}:
                padding_value = self.pad_token_id if key == "input_ids" else 0
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=padding_value,
                )
                continue
            if all(value.shape == values[0].shape for value in values):
                batch[key] = torch.stack(values, dim=0)
                continue
            raise ValueError(
                f"Cannot batch field '{key}' with variable tensor shapes. "
                "Use batch_size=1 or ensure images are resized to the same resolution."
            )

        for field, values in logits_payload.items():
            if any(value is not None for value in values):
                batch[field] = values[0] if len(values) == 1 else values

        batch["prompt_token_len"] = prompt_token_lens[0] if len(prompt_token_lens) == 1 else prompt_token_lens
        batch.update(metadata)
        return batch


def build_supervision_mask(labels):
    import torch

    return (labels != -100).to(dtype=torch.float32)


def _processor_call(processor, *, image: Image.Image, text: str, **kwargs):
    try:
        return processor(images=[image], text=[text], **kwargs)
    except TypeError:
        return processor(text=[text], images=[image], **kwargs)


def _build_training_prompt_text(processor, prompt: str) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _resolve_pad_token_id(processor) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    return int(pad_token_id or 0)


def _resize_training_image(image: Image.Image, resize_mode: str) -> Image.Image:
    mode = _normalize_image_resize_mode(resize_mode)
    if mode == "original":
        return image

    target_height = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
    }[mode]
    width, height = image.size
    if height <= target_height:
        return image

    target_width = round(width * target_height / height)
    return image.resize((target_width, target_height), _pil_lanczos())


def _normalize_image_resize_mode(resize_mode: str | None) -> str:
    mode = (resize_mode or "original").lower()
    aliases = {
        "none": "original",
        "no": "original",
        "off": "original",
        "native": "original",
        "original": "original",
        "480": "480p",
        "480p": "480p",
        "720": "720p",
        "720p": "720p",
        "1080": "1080p",
        "1080p": "1080p",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported training.image_resize={resize_mode!r}. "
            "Use one of: original, 480p, 720p, 1080p."
        )
    return aliases[mode]


def _pil_lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")
