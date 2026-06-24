from __future__ import annotations

from dataclasses import dataclass

_INACTIVE_LOGIT = -1.0e4


@dataclass
class SwitchKDLossOutput:
    loss: object
    lm_loss: object
    dbild_loss: object
    vsd_loss: object


class SwitchKDLoss:
    """Core Switch-KD objective: LM + DBiLD + optional VSD reference loss.

    The implementation keeps the math framework model-agnostic. VSD logits can come
    from an online visual-switch forward pass or from a precomputed cache.
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        dbild_weight: float = 0.5,
        vsd_weight: float = 0.5,
        temperature: float = 2.0,
        top_k: int = 64,
        top_k_mode: str = "fixed",
        kneedle_candidate_k: int = 256,
        min_top_k: int = 4,
        max_top_k: int | None = None,
        kl_mode: str = "symmetric",
        min_prob: float = 0.0,
    ) -> None:
        self.lm_weight = lm_weight
        self.dbild_weight = dbild_weight
        self.vsd_weight = vsd_weight
        self.temperature = temperature
        self.top_k = top_k
        self.top_k_mode = top_k_mode
        self.kneedle_candidate_k = kneedle_candidate_k
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.kl_mode = kl_mode
        self.min_prob = min_prob
        self._debug_emitted = False

    def __call__(
        self,
        student_logits,
        labels,
        teacher_logits=None,
        switch_logits=None,
        attention_mask=None,
        teacher_token_weight=None,
        switch_token_weight=None,
        sample_weight=None,
    ) -> SwitchKDLossOutput:
        lm_loss = _causal_lm_loss(student_logits, labels)
        zero = student_logits.new_zeros(())
        debug_enabled = not self._debug_emitted

        if debug_enabled:
            print(
                "Switch-KD first loss call:",
                f"teacher_reference={'compact' if isinstance(teacher_logits, dict) else 'dense' if teacher_logits is not None else 'none'}",
                f"switch_reference={'compact' if isinstance(switch_logits, dict) else 'dense' if switch_logits is not None else 'none'}",
                f"student_logits_shape={tuple(student_logits.shape)}",
                f"top_k_mode={self.top_k_mode}",
                f"kl_mode={self.kl_mode}",
                f"dbild_top_k={self.top_k}",
                f"kneedle_candidate_k={self.kneedle_candidate_k}",
            )

        dbild_loss = zero
        if teacher_logits is not None:
            dbild_loss = dynamic_bidirectional_logits_difference(
                student_logits=student_logits,
                reference_logits=teacher_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                top_k_mode=self.top_k_mode,
                kneedle_candidate_k=self.kneedle_candidate_k,
                min_top_k=self.min_top_k,
                max_top_k=self.max_top_k,
                kl_mode=self.kl_mode,
                min_prob=self.min_prob,
                token_weight=teacher_token_weight,
                sample_weight=sample_weight,
                debug_enabled=debug_enabled,
                debug_label="teacher",
            )

        vsd_loss = zero
        if switch_logits is not None:
            vsd_loss = visual_switch_divergence(
                student_logits=student_logits,
                reference_logits=switch_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                top_k_mode=self.top_k_mode,
                kneedle_candidate_k=self.kneedle_candidate_k,
                min_top_k=self.min_top_k,
                max_top_k=self.max_top_k,
                kl_mode=self.kl_mode,
                min_prob=self.min_prob,
                token_weight=switch_token_weight,
                sample_weight=sample_weight,
                debug_enabled=debug_enabled,
                debug_label="switch",
            )

        loss = self.lm_weight * lm_loss + self.dbild_weight * dbild_loss + self.vsd_weight * vsd_loss
        self._debug_emitted = True
        return SwitchKDLossOutput(loss=loss, lm_loss=lm_loss, dbild_loss=dbild_loss, vsd_loss=vsd_loss)


def _emit_reference_debug(
    *,
    label: str,
    compact: bool,
    student_logits_shape,
    reference_logits_shape,
    reference_indices_shape=None,
    gathered_logits_shape=None,
    candidate_count_shape=None,
    candidate_count_max=None,
    student_token_k_stats=None,
    reference_token_k_stats=None,
):
    print(
        f"Switch-KD {label} path:",
        f"reference_kind={'compact' if compact else 'dense'}",
        f"student_logits={tuple(student_logits_shape)}",
        f"reference_logits={tuple(reference_logits_shape)}",
        f"reference_indices={tuple(reference_indices_shape) if reference_indices_shape is not None else None}",
        f"gathered_student_candidate_logits={tuple(gathered_logits_shape) if gathered_logits_shape is not None else None}",
        f"candidate_count_shape={tuple(candidate_count_shape) if candidate_count_shape is not None else None}",
        f"candidate_count_max={candidate_count_max}",
        f"student_token_k_stats={student_token_k_stats}",
        f"reference_token_k_stats={reference_token_k_stats}",
    )


def _build_candidate_union(
    student_top_indices,
    reference_top_indices,
    student_active=None,
    reference_active=None,
):
    import torch

    combined_indices = torch.cat([student_top_indices, reference_top_indices], dim=-1)
    if student_active is None:
        student_active = torch.ones_like(student_top_indices, dtype=torch.bool)
    if reference_active is None:
        reference_active = torch.ones_like(reference_top_indices, dtype=torch.bool)
    combined_active = torch.cat([student_active, reference_active], dim=-1)

    sanitized_indices = torch.where(combined_active, combined_indices, torch.zeros_like(combined_indices))
    sorted_indices, sort_order = sanitized_indices.sort(dim=-1)
    sorted_active = combined_active.gather(dim=-1, index=sort_order)

    is_unique = sorted_active.clone()
    is_unique[..., 1:] = sorted_active[..., 1:] & (
        (~sorted_active[..., :-1]) | (sorted_indices[..., 1:] != sorted_indices[..., :-1])
    )

    candidate_count = is_unique.sum(dim=-1)
    rank = is_unique.cumsum(dim=-1) - 1
    padded_rank = torch.where(is_unique, rank, rank + sorted_indices.shape[-1])
    compact_order = padded_rank.argsort(dim=-1)
    candidate_indices = sorted_indices.gather(dim=-1, index=compact_order)

    position = torch.arange(sorted_indices.shape[-1], device=sorted_indices.device).view(
        *((1,) * (sorted_indices.ndim - 1)),
        -1,
    )
    candidate_active = position < candidate_count.unsqueeze(-1)
    candidate_indices = torch.where(candidate_active, candidate_indices, torch.zeros_like(candidate_indices))
    return candidate_indices, candidate_active, candidate_count


def _kneedle_topk_indices(logits, *, candidate_k, min_top_k, max_top_k):
    import torch

    vocab_size = int(logits.shape[-1])
    candidate_k = min(int(candidate_k), vocab_size)
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")

    top_values, top_indices = torch.topk(logits, k=candidate_k, dim=-1)
    first_value = top_values[..., :1]
    last_value = top_values[..., -1:]
    eps = torch.finfo(top_values.dtype).eps if top_values.is_floating_point() else 1e-6
    denom = (first_value - last_value).clamp_min(eps)
    normalized_values = (top_values - last_value) / denom

    rank = torch.arange(candidate_k, device=logits.device, dtype=top_values.dtype)
    if candidate_k == 1:
        normalized_rank = rank.view(*((1,) * (top_values.ndim - 1)), candidate_k)
    else:
        normalized_rank = (rank / float(candidate_k - 1)).view(*((1,) * (top_values.ndim - 1)), candidate_k)
    knee_score = normalized_values - (1.0 - normalized_rank)

    token_k = knee_score.argmax(dim=-1) + 1
    effective_max_top_k = candidate_k if max_top_k is None else min(candidate_k, int(max_top_k))
    token_k = token_k.clamp(min=int(min_top_k), max=effective_max_top_k)
    top_indices = top_indices[..., :effective_max_top_k]
    active = torch.arange(effective_max_top_k, device=logits.device).view(
        *((1,) * (token_k.ndim)),
        effective_max_top_k,
    ) < token_k.unsqueeze(-1)
    top_indices = torch.where(active, top_indices, torch.zeros_like(top_indices))
    return top_indices, active, token_k


def _token_k_stats(token_k):
    if token_k is None or token_k.numel() == 0:
        return None
    token_k_float = token_k.float()
    return {
        "min": float(token_k_float.min().item()),
        "max": float(token_k_float.max().item()),
        "mean": float(token_k_float.float().mean().item()),
    }


def _apply_candidate_min_prob(
    *,
    scaled_student,
    scaled_reference,
    candidate_active,
    min_prob: float,
):
    import torch.nn.functional as F

    if min_prob <= 0:
        return scaled_student, scaled_reference, candidate_active

    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    reference_log_probs = F.log_softmax(scaled_reference, dim=-1)
    informative = candidate_active & (
        (student_log_probs.exp() > min_prob) | (reference_log_probs.exp() > min_prob)
    )
    fill_value = scaled_student.new_full((), _INACTIVE_LOGIT)
    scaled_student = scaled_student.masked_fill(~informative, fill_value)
    scaled_reference = scaled_reference.masked_fill(~informative, fill_value)
    return scaled_student, scaled_reference, informative


def dynamic_bidirectional_logits_difference(
    student_logits,
    reference_logits,
    attention_mask=None,
    temperature: float = 2.0,
    top_k: int = 64,
    top_k_mode: str = "fixed",
    kneedle_candidate_k: int = 256,
    min_top_k: int = 4,
    max_top_k: int | None = None,
    kl_mode: str = "symmetric",
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
    debug_enabled: bool = False,
    debug_label: str = "reference",
):
    """DBiLD approximation for Switch-KD over cached compact reference logits."""
    import torch
    import torch.nn.functional as F

    if isinstance(reference_logits, dict):
        return _compact_bidirectional_logits_difference(
            student_logits=student_logits,
            reference_logits=reference_logits,
            attention_mask=attention_mask,
            temperature=temperature,
            top_k=top_k,
            top_k_mode=top_k_mode,
            kneedle_candidate_k=kneedle_candidate_k,
            min_top_k=min_top_k,
            max_top_k=max_top_k,
            kl_mode=kl_mode,
            min_prob=min_prob,
            token_weight=token_weight,
            sample_weight=sample_weight,
            debug_enabled=debug_enabled,
            debug_label=debug_label,
        )

    if student_logits.shape != reference_logits.shape:
        raise ValueError(
            "student_logits and reference_logits must have the same shape. "
            f"Got {student_logits.shape} and {reference_logits.shape}."
        )

    vocab_size = student_logits.shape[-1]
    effective_top_k = min(top_k, vocab_size)
    safe_temperature = max(float(temperature), 1e-6)
    student_active = None
    reference_active = None
    student_token_k = None
    reference_token_k = None
    if top_k_mode == "fixed":
        student_top_indices = torch.topk(student_logits, k=effective_top_k, dim=-1).indices
        reference_top_indices = torch.topk(reference_logits, k=effective_top_k, dim=-1).indices
    elif top_k_mode == "kneedle":
        student_top_indices, student_active, student_token_k = _kneedle_topk_indices(
            student_logits,
            candidate_k=kneedle_candidate_k,
            min_top_k=min_top_k,
            max_top_k=max_top_k,
        )
        reference_top_indices, reference_active, reference_token_k = _kneedle_topk_indices(
            reference_logits,
            candidate_k=kneedle_candidate_k,
            min_top_k=min_top_k,
            max_top_k=max_top_k,
        )
    else:
        raise ValueError(f"Unsupported top_k_mode: {top_k_mode}")

    candidate_indices, candidate_active, candidate_count = _build_candidate_union(
        student_top_indices,
        reference_top_indices,
        student_active=student_active,
        reference_active=reference_active,
    )

    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_indices)
    reference_candidate_logits = torch.gather(reference_logits, dim=-1, index=candidate_indices)
    if debug_enabled:
        _emit_reference_debug(
            label=debug_label,
            compact=False,
            student_logits_shape=student_logits.shape,
            reference_logits_shape=reference_logits.shape,
            gathered_logits_shape=student_candidate_logits.shape,
            candidate_count_shape=candidate_count.shape,
            candidate_count_max=int(candidate_count.max().item()) if candidate_count.numel() > 0 else 0,
            student_token_k_stats=_token_k_stats(student_token_k),
            reference_token_k_stats=_token_k_stats(reference_token_k),
        )

    fill_value = student_candidate_logits.new_full((), _INACTIVE_LOGIT)
    scaled_student = (student_candidate_logits / safe_temperature).masked_fill(~candidate_active, fill_value)
    scaled_reference = (reference_candidate_logits / safe_temperature).masked_fill(~candidate_active, fill_value)
    scaled_student, scaled_reference, candidate_active = _apply_candidate_min_prob(
        scaled_student=scaled_student,
        scaled_reference=scaled_reference,
        candidate_active=candidate_active,
        min_prob=min_prob,
    )

    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    reference_log_probs = F.log_softmax(scaled_reference, dim=-1)
    student_region_probs = student_log_probs.exp()
    reference_region_probs = reference_log_probs.exp()

    forward_kl = (reference_region_probs * (reference_log_probs - student_log_probs)).sum(dim=-1)
    reverse_kl = (student_region_probs * (student_log_probs - reference_log_probs)).sum(dim=-1)
    if kl_mode == "symmetric":
        token_loss = 0.5 * (forward_kl + reverse_kl) * (safe_temperature**2)
    elif kl_mode == "reverse":
        token_loss = reverse_kl * (safe_temperature**2)
    else:
        raise ValueError(f"Unsupported kl_mode: {kl_mode}")

    if token_weight is not None:
        token_loss = token_loss * token_weight.to(token_loss.dtype)
    if attention_mask is not None:
        token_loss = token_loss * attention_mask.to(token_loss.dtype)
        normalizer = attention_mask.to(token_loss.dtype)
        if token_weight is not None:
            normalizer = normalizer * token_weight.to(token_loss.dtype)
        loss = token_loss.sum() / normalizer.sum().clamp_min(1.0)
    else:
        loss = token_loss.mean()

    if sample_weight is not None:
        loss = loss * float(sample_weight)
    return loss


def visual_switch_divergence(
    student_logits,
    reference_logits,
    attention_mask=None,
    temperature: float = 2.0,
    top_k: int = 64,
    top_k_mode: str = "fixed",
    kneedle_candidate_k: int = 256,
    min_top_k: int = 4,
    max_top_k: int | None = None,
    kl_mode: str = "symmetric",
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
    debug_enabled: bool = False,
    debug_label: str = "switch",
):
    return dynamic_bidirectional_logits_difference(
        student_logits=student_logits,
        reference_logits=reference_logits,
        attention_mask=attention_mask,
        temperature=temperature,
        top_k=top_k,
        top_k_mode=top_k_mode,
        kneedle_candidate_k=kneedle_candidate_k,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        kl_mode=kl_mode,
        min_prob=min_prob,
        token_weight=token_weight,
        sample_weight=sample_weight,
        debug_enabled=debug_enabled,
        debug_label=debug_label,
    )


def _compact_bidirectional_logits_difference(
    *,
    student_logits,
    reference_logits: dict,
    attention_mask=None,
    temperature: float = 2.0,
    top_k: int = 64,
    top_k_mode: str = "fixed",
    kneedle_candidate_k: int = 256,
    min_top_k: int = 4,
    max_top_k: int | None = None,
    kl_mode: str = "symmetric",
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
    debug_enabled: bool = False,
    debug_label: str = "reference",
):
    import torch
    import torch.nn.functional as F

    reference_indices = reference_logits["indices"]
    reference_values = reference_logits["logits"] if "logits" in reference_logits else reference_logits["values"]
    token_k = reference_logits.get("token_k")

    if reference_indices.ndim != 3 or reference_values.ndim != 3:
        raise ValueError(
            "Compact reference logits must have rank 3 [batch, seq, k]. "
            f"Got indices {tuple(reference_indices.shape)} and values {tuple(reference_values.shape)}."
        )
    if student_logits.shape[:2] != reference_indices.shape[:2]:
        raise ValueError(
            "Compact reference logits must align with student batch/seq dimensions. "
            f"Got student {tuple(student_logits.shape)} and indices {tuple(reference_indices.shape)}."
        )

    safe_temperature = max(float(temperature), 1e-6)
    if token_k is None:
        reference_active = torch.ones_like(reference_indices, dtype=torch.bool)
    else:
        positions = torch.arange(reference_indices.shape[-1], device=reference_indices.device).view(1, 1, -1)
        reference_active = positions < token_k.unsqueeze(-1)

    student_active = None
    student_token_k = None
    if top_k_mode == "fixed":
        effective_top_k = min(top_k, student_logits.shape[-1])
        student_top_indices = torch.topk(student_logits, k=effective_top_k, dim=-1).indices
    elif top_k_mode == "kneedle":
        student_top_indices, student_active, student_token_k = _kneedle_topk_indices(
            student_logits,
            candidate_k=kneedle_candidate_k,
            min_top_k=min_top_k,
            max_top_k=max_top_k,
        )
    else:
        raise ValueError(f"Unsupported top_k_mode: {top_k_mode}")

    candidate_indices, candidate_active, candidate_count = _build_candidate_union(
        student_top_indices,
        reference_indices,
        student_active=student_active,
        reference_active=reference_active,
    )
    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_indices)

    matches = candidate_indices.unsqueeze(-1) == reference_indices.unsqueeze(-2)
    matches = matches & reference_active.unsqueeze(-2)
    reference_fill = reference_values.new_full((), _INACTIVE_LOGIT)
    reference_candidate_logits = torch.where(
        matches.any(dim=-1),
        reference_values.unsqueeze(-2).masked_fill(~matches, reference_fill).max(dim=-1).values,
        reference_fill,
    )

    if debug_enabled:
        _emit_reference_debug(
            label=debug_label,
            compact=True,
            student_logits_shape=student_logits.shape,
            reference_logits_shape=reference_values.shape,
            reference_indices_shape=reference_indices.shape,
            gathered_logits_shape=student_candidate_logits.shape,
            candidate_count_shape=candidate_count.shape,
            candidate_count_max=int(candidate_count.max().item()) if candidate_count.numel() > 0 else 0,
            student_token_k_stats=_token_k_stats(student_token_k),
            reference_token_k_stats=_token_k_stats(token_k),
        )

    fill_value = student_candidate_logits.new_full((), _INACTIVE_LOGIT)
    scaled_student = (student_candidate_logits / safe_temperature).masked_fill(~candidate_active, fill_value)
    scaled_reference = (reference_candidate_logits / safe_temperature).masked_fill(~candidate_active, fill_value)
    scaled_student, scaled_reference, candidate_active = _apply_candidate_min_prob(
        scaled_student=scaled_student,
        scaled_reference=scaled_reference,
        candidate_active=candidate_active,
        min_prob=min_prob,
    )

    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    reference_log_probs = F.log_softmax(scaled_reference, dim=-1)
    student_probs = student_log_probs.exp()
    reference_probs = reference_log_probs.exp()

    forward_kl = (reference_probs * (reference_log_probs - student_log_probs)).sum(dim=-1)
    reverse_kl = (student_probs * (student_log_probs - reference_log_probs)).sum(dim=-1)
    if kl_mode == "symmetric":
        token_loss = 0.5 * (forward_kl + reverse_kl) * (safe_temperature**2)
    elif kl_mode == "reverse":
        token_loss = reverse_kl * (safe_temperature**2)
    else:
        raise ValueError(f"Unsupported kl_mode: {kl_mode}")

    active_tokens = reference_active.any(dim=-1).to(token_loss.dtype)
    token_loss = token_loss * active_tokens

    if token_weight is not None:
        token_loss = token_loss * token_weight.to(token_loss.dtype)
    if attention_mask is not None:
        token_loss = token_loss * attention_mask.to(token_loss.dtype)
        normalizer = attention_mask.to(token_loss.dtype) * active_tokens
        if token_weight is not None:
            normalizer = normalizer * token_weight.to(token_loss.dtype)
        loss = token_loss.sum() / normalizer.sum().clamp_min(1.0)
    else:
        normalizer = active_tokens
        if token_weight is not None:
            normalizer = normalizer * token_weight.to(token_loss.dtype)
        loss = token_loss.sum() / normalizer.sum().clamp_min(1.0)

    if sample_weight is not None:
        loss = loss * float(sample_weight)
    return loss


def _causal_lm_loss(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
