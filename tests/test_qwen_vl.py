from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("QWEN_VL_SMOKE_MODEL_PATH"),
    reason="manual Qwen-VL smoke test requires QWEN_VL_SMOKE_MODEL_PATH",
)
def test_qwen_vl_manual_smoke_loads() -> None:
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    import torch

    model_path = Path(os.environ["QWEN_VL_SMOKE_MODEL_PATH"])
    processor = AutoProcessor.from_pretrained(model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForVLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    assert processor is not None
    assert model is not None
