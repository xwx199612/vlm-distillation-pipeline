from __future__ import annotations

from typing import Any

INACTIVE_LOGIT = -1.0e4


def compact_logits(logits, max_vocab: int | None) -> dict[str, Any] | list[list[list[float]]]:
    import torch

    logits = logits.detach().float().cpu()
    if not max_vocab:
        return logits.tolist()
    top_values, top_indices = torch.topk(logits, k=min(max_vocab, logits.shape[-1]), dim=-1)
    return {
        "indices": top_indices.tolist(),
        "values": top_values.tolist(),
        "shape": list(logits.shape),
        "vocab_size": int(logits.shape[-1]),
    }


def is_compact_logits(cached: Any) -> bool:
    return isinstance(cached, dict) and "indices" in cached and "values" in cached


def cached_vocab_size(cached: Any) -> int | None:
    if is_compact_logits(cached):
        if "vocab_size" in cached:
            return int(cached["vocab_size"])
        shape = cached.get("shape")
        if shape:
            return int(shape[-1])
    if isinstance(cached, list) and cached and isinstance(cached[0], list):
        return len(cached[0][0])
    return None


def cached_token_weight(cached: Any, *, device, dtype):
    import torch

    if not is_compact_logits(cached):
        return None

    values = cached.get("entropy_weight")
    if values is None:
        return None
    return torch.tensor(values, device=device, dtype=dtype)


def compact_logits_to_tensors(cached: Any, *, device, dtype) -> dict[str, Any] | None:
    import torch

    if not is_compact_logits(cached):
        return None

    compact = {
        "indices": torch.tensor(cached["indices"], device=device, dtype=torch.long),
        "values": torch.tensor(cached["values"], device=device, dtype=dtype),
        "shape": tuple(int(value) for value in cached["shape"]),
        "vocab_size": int(cached_vocab_size(cached) or cached["shape"][-1]),
    }
    token_k = cached.get("token_k")
    if token_k is not None:
        compact["token_k"] = torch.tensor(token_k, device=device, dtype=torch.long)
    return compact


def align_compact_reference_to_suffix(
    reference: dict[str, Any],
    *,
    target_shape: tuple[int, ...],
    reference_prompt_len: int | None,
    student_prompt_len: int | None,
    dtype=None,
):
    import torch

    if len(target_shape) != 3:
        raise ValueError(f"Expected target_shape (batch, seq, vocab), got {target_shape}")

    batch_size, seq_len, _ = target_shape
    indices = reference["indices"]
    values = reference["values"]
    token_k = reference.get("token_k")
    k_dim = int(indices.shape[-1])
    out_dtype = dtype or values.dtype

    if indices.shape[0] != batch_size:
        if indices.shape[0] == 1 and batch_size > 1:
            indices = indices.expand(batch_size, -1, -1)
            values = values.expand(batch_size, -1, -1)
            if token_k is not None:
                token_k = token_k.expand(batch_size, -1)
        else:
            indices = indices[:batch_size]
            values = values[:batch_size]
            if token_k is not None:
                token_k = token_k[:batch_size]

    pad_indices = torch.zeros((batch_size, seq_len, k_dim), device=indices.device, dtype=indices.dtype)
    pad_values = torch.full((batch_size, seq_len, k_dim), INACTIVE_LOGIT, device=values.device, dtype=out_dtype)
    pad_token_k = torch.zeros((batch_size, seq_len), device=indices.device, dtype=torch.long)

    if reference_prompt_len is None or student_prompt_len is None:
        copy_len = min(indices.shape[1], seq_len)
        pad_indices[:, :copy_len] = indices[:, :copy_len]
        pad_values[:, :copy_len] = values[:, :copy_len].to(out_dtype)
        if token_k is not None:
            pad_token_k[:, :copy_len] = token_k[:, :copy_len]
        else:
            pad_token_k[:, :copy_len] = k_dim
        return {
            "indices": pad_indices,
            "values": pad_values,
            "token_k": pad_token_k,
            "shape": (batch_size, seq_len, int(reference["vocab_size"])),
            "vocab_size": int(reference["vocab_size"]),
        }

    ref_answer = indices[:, int(reference_prompt_len) :]
    ref_answer_values = values[:, int(reference_prompt_len) :]
    ref_answer_token_k = token_k[:, int(reference_prompt_len) :] if token_k is not None else None
    answer_len = seq_len - int(student_prompt_len)
    if answer_len > 0:
        copy_len = min(ref_answer.shape[1], answer_len)
        start = int(student_prompt_len)
        end = start + copy_len
        pad_indices[:, start:end] = ref_answer[:, :copy_len]
        pad_values[:, start:end] = ref_answer_values[:, :copy_len].to(out_dtype)
        if ref_answer_token_k is not None:
            pad_token_k[:, start:end] = ref_answer_token_k[:, :copy_len]
        else:
            pad_token_k[:, start:end] = k_dim

    return {
        "indices": pad_indices,
        "values": pad_values,
        "token_k": pad_token_k,
        "shape": (batch_size, seq_len, int(reference["vocab_size"])),
        "vocab_size": int(reference["vocab_size"]),
    }


def remap_compact_reference_to_student_vocab(
    reference: dict[str, Any],
    *,
    reference_vocab: int | None,
    student_vocab_size: int,
    shared_token_vocab_size: int | None,
):
    import torch

    if reference_vocab is None or shared_token_vocab_size is None:
        return None

    copy_len = min(
        int(shared_token_vocab_size),
        int(reference_vocab),
        int(student_vocab_size),
    )
    if copy_len <= 0:
        return None

    indices = reference["indices"]
    values = reference["values"].clone()
    token_k = reference.get("token_k")
    valid = indices < copy_len
    remapped_indices = indices.clamp(max=max(copy_len - 1, 0))
    values = values.masked_fill(~valid, INACTIVE_LOGIT)

    if token_k is None:
        token_k = valid.sum(dim=-1, dtype=torch.long)
    else:
        capped_token_k = valid.sum(dim=-1, dtype=torch.long)
        token_k = torch.minimum(token_k, capped_token_k)

    return {
        "indices": remapped_indices,
        "values": values,
        "token_k": token_k,
        "shape": (*reference["shape"][:-1], int(student_vocab_size)),
        "vocab_size": int(student_vocab_size),
    }


def materialize_cached_logits(cached: Any, *, device, dtype, vocab_size: int | None = None):
    import torch

    if is_compact_logits(cached):
        shape = tuple(cached["shape"])
        if vocab_size is None:
            vocab_size = cached_vocab_size(cached)
        if vocab_size is None:
            vocab_size = int(max(cached["indices"][-1])) + 1
        tensor = torch.full(shape, torch.finfo(dtype).min, device=device, dtype=dtype)
        indices = torch.tensor(cached["indices"], device=device)
        values = torch.tensor(cached["values"], device=device, dtype=dtype)
        tensor.scatter_(-1, indices, values)
        return tensor

    tensor = torch.tensor(cached, device=device, dtype=dtype)
    if vocab_size is not None and tensor.shape[-1] != vocab_size:
        tensor = align_reference_logits(tensor, target_shape=(*tensor.shape[:-1], vocab_size), dtype=dtype)
    return tensor


def align_reference_logits(reference, *, target_shape: tuple[int, ...], dtype=None):
    """Pad or truncate reference logits to match student logits shape."""
    import torch

    if len(target_shape) != 3:
        raise ValueError(f"Expected target_shape (batch, seq, vocab), got {target_shape}")

    fill_value = torch.finfo(dtype or reference.dtype).min
    batch_size, seq_len, vocab_size = target_shape
    aligned = reference

    if aligned.shape[0] != batch_size:
        if aligned.shape[0] == 1 and batch_size > 1:
            aligned = aligned.expand(batch_size, -1, -1)
        else:
            aligned = aligned[:batch_size]

    if aligned.shape[-1] < vocab_size:
        pad = torch.full(
            (aligned.shape[0], aligned.shape[1], vocab_size - aligned.shape[-1]),
            fill_value,
            device=aligned.device,
            dtype=aligned.dtype,
        )
        aligned = torch.cat([aligned, pad], dim=-1)
    elif aligned.shape[-1] > vocab_size:
        aligned = aligned[..., :vocab_size]

    if aligned.shape[1] < seq_len:
        pad = torch.full(
            (aligned.shape[0], seq_len - aligned.shape[1], vocab_size),
            fill_value,
            device=aligned.device,
            dtype=aligned.dtype,
        )
        aligned = torch.cat([aligned, pad], dim=1)
    elif aligned.shape[1] > seq_len:
        aligned = aligned[:, :seq_len, :]

    return aligned


def align_reference_logits_to_suffix(
    reference,
    *,
    target_shape: tuple[int, ...],
    reference_prompt_len: int | None,
    student_prompt_len: int | None,
    dtype=None,
):
    """Align cached logits by matching answer suffixes when prompt lengths differ."""
    import torch

    if reference_prompt_len is None or student_prompt_len is None:
        return align_reference_logits(reference, target_shape=target_shape, dtype=dtype)

    ref_answer = reference[:, int(reference_prompt_len) :, :]
    batch_size, seq_len, vocab_size = target_shape
    answer_len = seq_len - int(student_prompt_len)
    if answer_len <= 0:
        return align_reference_logits(reference, target_shape=target_shape, dtype=dtype)

    fill_value = torch.finfo(dtype or reference.dtype).min
    if ref_answer.shape[1] > answer_len:
        ref_answer = ref_answer[:, :answer_len, :]
    elif ref_answer.shape[1] < answer_len:
        pad = torch.full(
            (ref_answer.shape[0], answer_len - ref_answer.shape[1], ref_answer.shape[-1]),
            fill_value,
            device=ref_answer.device,
            dtype=ref_answer.dtype,
        )
        ref_answer = torch.cat([ref_answer, pad], dim=1)

    aligned = torch.full(
        (batch_size, seq_len, ref_answer.shape[-1]),
        fill_value,
        device=reference.device,
        dtype=reference.dtype,
    )
    aligned[:, int(student_prompt_len) : int(student_prompt_len) + ref_answer.shape[1], :] = ref_answer
    return align_reference_logits(aligned, target_shape=target_shape, dtype=dtype)


def align_reference_token_weight_to_suffix(
    reference,
    *,
    target_shape: tuple[int, ...],
    reference_prompt_len: int | None,
    student_prompt_len: int | None,
    dtype=None,
):
    import torch

    if len(target_shape) != 2:
        raise ValueError(f"Expected target_shape (batch, seq), got {target_shape}")

    batch_size, seq_len = target_shape
    aligned = reference

    if aligned.shape[0] != batch_size:
        if aligned.shape[0] == 1 and batch_size > 1:
            aligned = aligned.expand(batch_size, -1)
        else:
            aligned = aligned[:batch_size]

    fill_value = torch.zeros((), device=aligned.device, dtype=dtype or aligned.dtype)
    if reference_prompt_len is None or student_prompt_len is None:
        output = torch.full(
            (batch_size, seq_len),
            fill_value,
            device=aligned.device,
            dtype=dtype or aligned.dtype,
        )
        copy_len = min(aligned.shape[1], seq_len)
        output[:, :copy_len] = aligned[:, :copy_len]
        return output

    ref_answer = aligned[:, int(reference_prompt_len) :]
    answer_len = seq_len - int(student_prompt_len)
    if answer_len <= 0:
        return torch.zeros((batch_size, seq_len), device=aligned.device, dtype=dtype or aligned.dtype)

    if ref_answer.shape[1] > answer_len:
        ref_answer = ref_answer[:, :answer_len]
    elif ref_answer.shape[1] < answer_len:
        pad = torch.zeros(
            (ref_answer.shape[0], answer_len - ref_answer.shape[1]),
            device=aligned.device,
            dtype=dtype or aligned.dtype,
        )
        ref_answer = torch.cat([ref_answer, pad], dim=1)

    output = torch.zeros((batch_size, seq_len), device=aligned.device, dtype=dtype or aligned.dtype)
    output[:, int(student_prompt_len) : int(student_prompt_len) + ref_answer.shape[1]] = ref_answer
    return output


def vocab_sizes_compatible(reference_vocab: int | None, student_vocab: int) -> bool:
    return reference_vocab is not None and reference_vocab == student_vocab
