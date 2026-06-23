from __future__ import annotations

import re
from typing import Any


def resolve_requested_device_map(device_map: str | None, *, quantization: str, role: str = "teacher") -> str:
    if device_map is None:
        raise ValueError(
            f"{role}.device_map is required for the Hugging Face {role} model. "
            "Set it explicitly in YAML, for example `auto`."
        )

    resolved = device_map.strip()
    if not resolved:
        raise ValueError(
            f"{role}.device_map must be a non-empty string. "
            "Set it explicitly in YAML, for example `auto`."
        )

    if resolved.lower() in {"cpu", "disk"}:
        raise ValueError(
            f"{role}.device_map={device_map!r} is not allowed for the {role} model. "
            "Use a CUDA placement such as `auto`."
        )

    if quantization in {"4bit", "8bit"} and resolved != "auto":
        raise ValueError(
            f"{role}.device_map must be 'auto' when {role}.quantization={quantization!r}. "
            f"Got {device_map!r}."
        )

    allowed_special_values = {"auto", "balanced", "balanced_low_0", "sequential"}
    if resolved in allowed_special_values or resolved == "cuda" or re.fullmatch(r"cuda:\d+", resolved):
        return resolved

    raise ValueError(
        f"Unsupported {role}.device_map={device_map!r}. "
        "Use one of: auto, balanced, balanced_low_0, sequential, cuda, cuda:N."
    )


def normalize_device_location(location: Any):
    import torch

    if isinstance(location, int):
        return torch.device(f"cuda:{location}")
    if isinstance(location, str):
        text = location.lower()
        if text == "cuda" or text.startswith("cuda:") or text == "cpu":
            return torch.device(location)
    return None


def module_device(module):
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return None


def batch_to_device(batch: dict[str, Any], device):
    if device is None:
        return batch
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def select_model_input_device(model, *, preferred_modules=(), label: str = "model"):
    for module in preferred_modules:
        device = module_device(module)
        if device is not None:
            return device

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for location in device_map.values():
            device = normalize_device_location(location)
            if device is not None and device.type == "cuda":
                return device
        raise RuntimeError(f"{label} hf_device_map does not contain a CUDA device: {device_map!r}.")

    if hasattr(model, "device"):
        return model.device

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError(f"Could not determine {label} input device.") from exc


def model_debug_info(model) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    cpu_parts = []
    if device_map:
        cpu_parts = [key for key, value in device_map.items() if str(value).lower() in {"cpu", "disk"}]

    try:
        first_param = next(model.parameters())
        first_param_device = first_param.device
        first_param_dtype = first_param.dtype
    except StopIteration:
        first_param_device = None
        first_param_dtype = None

    return {
        "hf_device_map": device_map,
        "cpu_offload_parts": cpu_parts,
        "first_param_device": first_param_device,
        "first_param_dtype": first_param_dtype,
    }


def print_stage_model_debug(
    *,
    stage_label: str,
    model_path: str,
    quantization_mode: str,
    requested_device_map: str | None,
    model,
    selected_input_device,
) -> None:
    info = model_debug_info(model)
    print(f"{stage_label} resolved model path:", model_path)
    print(f"{stage_label} quantization mode:", quantization_mode)
    print(f"{stage_label} requested device_map:", requested_device_map)
    print(f"{stage_label} hf_device_map:", info["hf_device_map"])
    print(f"{stage_label} CPU/DISK offload parts:", info["cpu_offload_parts"])
    print(f"{stage_label} first parameter device:", info["first_param_device"])
    print(f"{stage_label} first parameter dtype:", info["first_param_dtype"])
    print(f"{stage_label} selected input tensor device:", selected_input_device)


def ensure_stage_uses_cuda(
    *,
    stage_label: str,
    requested_device_map: str | None,
    model,
    selected_input_device,
) -> None:
    info = model_debug_info(model)
    has_cuda_in_map = bool(info["hf_device_map"]) and any(
        (device := normalize_device_location(location)) is not None and device.type == "cuda"
        for location in info["hf_device_map"].values()
    )
    has_cuda_first_param = info["first_param_device"] is not None and info["first_param_device"].type == "cuda"
    has_cuda_input = selected_input_device is not None and selected_input_device.type == "cuda"
    if has_cuda_in_map or has_cuda_first_param or has_cuda_input:
        return
    raise RuntimeError(
        f"{stage_label} model did not place on CUDA. "
        f"requested device_map={requested_device_map!r}, "
        f"hf_device_map={info['hf_device_map']!r}, "
        f"first_param_device={info['first_param_device']!r}, "
        f"selected_input_device={selected_input_device!r}. "
        "Refusing to continue with a CPU-loaded model."
    )
