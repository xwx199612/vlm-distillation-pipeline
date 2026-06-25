from __future__ import annotations

import torch
from PIL import Image

from vlm_distill.vlm_batching import encode_vlm_training_sample


class _CanonicalSpanProcessor:
    def __init__(self):
        self.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()
        self._token_map = {
            "<chat>prompt</chat>": [101, 102],
            "<chat>prompt answer</chat>": [101, 102, 5890, 7000],
            "<chat>prompt</chat> answer": [101, 102, 4913, 7000],
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return f"<chat>{messages[0]['content'][1]['text']}</chat>"

    def __call__(self, images=None, text="", return_tensors="pt", truncation=True, max_length=128):
        del images, return_tensors, truncation, max_length
        if isinstance(text, list):
            text = text[0]
        token_ids = self._token_map[text]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 4, 4),
        }


def test_encode_vlm_training_sample_canonical_mode_uses_full_input_answer_span():
    image = Image.new("RGB", (8, 8))
    processor = _CanonicalSpanProcessor()

    encoded = encode_vlm_training_sample(
        processor,
        image=image,
        prompt="prompt",
        target="answer",
        max_length=64,
        canonical_answer_span=True,
    )

    assert encoded.prompt_token_len == 2
    assert encoded.answer_token_ids == [5890, 7000]
    assert encoded.model_inputs["labels"].tolist() == [-100, -100, 5890, 7000]


def test_encode_vlm_training_sample_legacy_mode_keeps_noncanonical_boundary():
    image = Image.new("RGB", (8, 8))
    processor = _CanonicalSpanProcessor()

    encoded = encode_vlm_training_sample(
        processor,
        image=image,
        prompt="prompt",
        target="answer",
        max_length=64,
        canonical_answer_span=False,
    )

    assert encoded.prompt_token_len == 2
    assert encoded.model_inputs["labels"].tolist() == [-100, -100, 4913, 7000]
