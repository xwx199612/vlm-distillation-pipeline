from __future__ import annotations

from pathlib import Path

from .config_schema import PipelineConfig
from .model_loading import resolve_model_path


def merge_student_adapter(config: PipelineConfig) -> Path:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError as exc:
        raise RuntimeError(
            "Install torch, transformers, and peft to merge a student adapter."
        ) from exc

    base_model_path = resolve_model_path(
        config.student.inference_model_path or config.student.model_name
    )
    adapter_path = config.student.inference_adapter_path or config.student.adapter_dir
    output_path = config.student.merged_model_path or config.student.output_dir / "merged_model"

    resolved_base_path = Path(base_model_path).resolve()
    resolved_output_path = output_path.resolve()
    if resolved_output_path == resolved_base_path:
        raise ValueError(
            "Refusing to overwrite the base model directory while merging the adapter. "
            "Set student.merged_model_path to a different output directory."
        )

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter path is missing adapter_config.json: {adapter_path / 'adapter_config.json'}"
        )

    print(f"base_model_path={base_model_path}")
    print(f"adapter_path={adapter_path}")
    print(f"merged_model_path={output_path}")

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForVLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=config.student.attn_implementation,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_path)
    print(f"OK merged model written: {output_path}")
    return output_path
