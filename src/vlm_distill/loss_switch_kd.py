from __future__ import annotations

from dataclasses import dataclass

_SHAPE_DEBUG_EMITTED = False
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
        min_prob: float = 0.0,
    ) -> None:
        self.lm_weight = lm_weight
        self.dbild_weight = dbild_weight
        self.vsd_weight = vsd_weight
        self.temperature = temperature
        self.top_k = top_k
        self.min_prob = min_prob

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

        dbild_loss = zero
        if teacher_logits is not None:
            dbild_loss = dynamic_bidirectional_logits_difference(
                student_logits=student_logits,
                reference_logits=teacher_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                min_prob=self.min_prob,
                token_weight=teacher_token_weight,
                sample_weight=sample_weight,
            )

        vsd_loss = zero
        if switch_logits is not None:
            vsd_loss = visual_switch_divergence(
                student_logits=student_logits,
                reference_logits=switch_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                min_prob=self.min_prob,
                token_weight=switch_token_weight,
                sample_weight=sample_weight,
            )

        loss = self.lm_weight * lm_loss + self.dbild_weight * dbild_loss + self.vsd_weight * vsd_loss
        return SwitchKDLossOutput(loss=loss, lm_loss=lm_loss, dbild_loss=dbild_loss, vsd_loss=vsd_loss)


def _emit_shape_debug(
    *,
    student_logits_shape,
    reference_logits_shape,
    student_top_shape,
    reference_top_shape,
    candidate_indices_shape,
    gathered_logits_shape,
):
    global _SHAPE_DEBUG_EMITTED

    if _SHAPE_DEBUG_EMITTED:
        return

    print(
        "Switch-KD shapes:",
        f"student_logits={tuple(student_logits_shape)}",
        f"reference_logits={tuple(reference_logits_shape)}",
        f"student_top={tuple(student_top_shape)}",
        f"reference_top={tuple(reference_top_shape)}",
        f"candidate_indices={tuple(candidate_indices_shape)}",
        f"gathered_logits={tuple(gathered_logits_shape)}",
    )
    _SHAPE_DEBUG_EMITTED = True


def _build_candidate_union(student_top_indices, reference_top_indices, reference_active=None):
    import torch

    combined_indices = torch.cat([student_top_indices, reference_top_indices], dim=-1)
    student_active = torch.ones_like(student_top_indices, dtype=torch.bool)
    if reference_active is None:
        combined_active = torch.ones_like(combined_indices, dtype=torch.bool)
    else:
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
    return candidate_indices, candidate_active


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
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
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
            min_prob=min_prob,
            token_weight=token_weight,
            sample_weight=sample_weight,
        )

    if student_logits.shape != reference_logits.shape:
        raise ValueError(
            "student_logits and reference_logits must have the same shape. "
            f"Got {student_logits.shape} and {reference_logits.shape}."
        )

    vocab_size = student_logits.shape[-1]
    effective_top_k = min(top_k, vocab_size)
    safe_temperature = max(float(temperature), 1e-6)
    student_top_indices = torch.topk(student_logits, k=effective_top_k, dim=-1).indices
    reference_top_indices = torch.topk(reference_logits, k=effective_top_k, dim=-1).indices
    candidate_indices, candidate_active = _build_candidate_union(student_top_indices, reference_top_indices)

    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_indices)
    reference_candidate_logits = torch.gather(reference_logits, dim=-1, index=candidate_indices)
    _emit_shape_debug(
        student_logits_shape=student_logits.shape,
        reference_logits_shape=reference_logits.shape,
        student_top_shape=student_top_indices.shape,
        reference_top_shape=reference_top_indices.shape,
        candidate_indices_shape=candidate_indices.shape,
        gathered_logits_shape=student_candidate_logits.shape,
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
    token_loss = 0.5 * (forward_kl + reverse_kl) * (safe_temperature**2)

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
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
):
    return dynamic_bidirectional_logits_difference(
        student_logits=student_logits,
        reference_logits=reference_logits,
        attention_mask=attention_mask,
        temperature=temperature,
        top_k=top_k,
        min_prob=min_prob,
        token_weight=token_weight,
        sample_weight=sample_weight,
    )


def _compact_bidirectional_logits_difference(
    *,
    student_logits,
    reference_logits: dict,
    attention_mask=None,
    temperature: float = 2.0,
    top_k: int = 64,
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
):
    import torch
    import torch.nn.functional as F

    global _SHAPE_DEBUG_EMITTED

    reference_indices = reference_logits["indices"]
    reference_values = reference_logits["values"]
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
    effective_top_k = min(top_k, student_logits.shape[-1])
    student_top_indices = torch.topk(student_logits, k=effective_top_k, dim=-1).indices
    if token_k is None:
        reference_active = torch.ones_like(reference_indices, dtype=torch.bool)
    else:
        positions = torch.arange(reference_indices.shape[-1], device=reference_indices.device).view(1, 1, -1)
        reference_active = positions < token_k.unsqueeze(-1)

    candidate_indices, candidate_active = _build_candidate_union(
        student_top_indices,
        reference_indices,
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

    _emit_shape_debug(
        student_logits_shape=student_logits.shape,
        reference_logits_shape=reference_values.shape,
        student_top_shape=student_top_indices.shape,
        reference_top_shape=reference_indices.shape,
        candidate_indices_shape=candidate_indices.shape,
        gathered_logits_shape=student_candidate_logits.shape,
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
    token_loss = 0.5 * (forward_kl + reverse_kl) * (safe_temperature**2)

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
