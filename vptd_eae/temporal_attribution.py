"""Independent role-level implementation of temporal evidence attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TemporalAttributionOutput:
    """Reconstructed role target and diagnostics for candidate arguments."""

    target_probs: Tensor
    temporal_direction: Tensor
    attributed_shift: Tensor
    consensus_gate: Tensor
    explained_ratio: Tensor
    projection_scale: Tensor
    support_weight: Tensor
    refutation_weight: Tensor
    accepted_video_fraction: Tensor


def _validate_inputs(
    student_logits: Tensor,
    static_teacher_logits: Tensor,
    video_teacher_logits: Tensor,
    role_mask: Tensor | None,
) -> Tensor:
    if student_logits.ndim < 2:
        raise ValueError("student_logits must have shape [..., num_roles]")
    if static_teacher_logits.shape != student_logits.shape:
        raise ValueError("static_teacher_logits must match student_logits")
    if video_teacher_logits.ndim != student_logits.ndim + 1:
        raise ValueError("video_teacher_logits must have shape [num_videos, ..., num_roles]")
    if video_teacher_logits.shape[1:] != student_logits.shape:
        raise ValueError("video_teacher_logits trailing dimensions must match student_logits")
    if video_teacher_logits.shape[0] < 1:
        raise ValueError("at least one video hypothesis is required")

    if role_mask is None:
        return torch.ones_like(student_logits, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(role_mask, student_logits.shape)
    except RuntimeError as exc:
        raise ValueError("role_mask must broadcast to student_logits") from exc
    mask = mask.to(device=student_logits.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("every candidate requires at least one valid role")
    return mask


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    return F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def _center(log_probs: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(log_probs.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    safe = log_probs.masked_fill(~mask, 0.0)
    mean = (safe * valid).sum(dim=-1, keepdim=True) / count
    return (safe - mean) * valid


def _js_divergence(log_p: Tensor, log_q: Tensor, mask: Tensor) -> Tensor:
    log_p = log_p.masked_fill(~mask, -torch.inf)
    log_q = log_q.masked_fill(~mask, -torch.inf)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    p = log_p.exp().masked_fill(~mask, 0.0)
    q = log_q.exp().masked_fill(~mask, 0.0)
    kl_pm = (p * (log_p - log_m).masked_fill(~mask, 0.0)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m).masked_fill(~mask, 0.0)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _consensus_direction(
    directions: Tensor,
    role_mask: Tensor,
    consensus_ratio: float,
    minimum_cosine: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not 0.5 < consensus_ratio <= 1.0:
        raise ValueError("consensus_ratio must be in (0.5, 1.0]")
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in [-1, 1]")

    video_count = directions.shape[0]
    if video_count == 1:
        gate = torch.ones(directions.shape[1:-1], dtype=torch.bool, device=directions.device)
        fraction = torch.ones_like(gate, dtype=directions.dtype)
        return directions[0], gate, fraction

    expanded_mask = role_mask.unsqueeze(0).to(directions.dtype)
    # The median prevents one high-magnitude hallucinated video from changing
    # the consensus direction selected by the other video hypotheses.
    reference = (directions * expanded_mask).median(dim=0).values
    dot = (directions * reference.unsqueeze(0) * expanded_mask).sum(dim=-1)
    direction_norm = torch.linalg.vector_norm(directions * expanded_mask, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference * role_mask, dim=-1).unsqueeze(0)
    cosine = dot / (direction_norm * reference_norm + eps)
    accepted = (cosine >= minimum_cosine) & (direction_norm > eps) & (reference_norm > eps)

    required = math.ceil(video_count * consensus_ratio)
    gate = accepted.sum(dim=0) >= required
    accepted_fraction = accepted.to(directions.dtype).mean(dim=0)
    weights = accepted.to(directions.dtype).unsqueeze(-1)
    aggregate = (directions * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
    aggregate = aggregate * gate.unsqueeze(-1) * role_mask
    return aggregate, gate, accepted_fraction


def build_temporal_target(
    student_logits: Tensor,
    static_teacher_logits: Tensor,
    video_teacher_logits: Tensor,
    *,
    role_mask: Tensor | None = None,
    consensus_ratio: float = 2.0 / 3.0,
    minimum_consensus_cosine: float = 0.5,
    positive_support_cap: float = 0.8,
    projection_ridge: float = 1e-3,
    maximum_logit_shift: float = 4.0,
    eps: float = 1e-8,
) -> TemporalAttributionOutput:
    """Build an EAE student-anchored target from temporal evidence.

    The last axis is the event-specific role support, including ``NONE``. The
    first axis of ``video_teacher_logits`` indexes generated video hypotheses.
    """

    if not 0.0 <= positive_support_cap <= 1.0:
        raise ValueError("positive_support_cap must be in [0, 1]")
    if projection_ridge <= 0:
        raise ValueError("projection_ridge must be positive")

    mask = _validate_inputs(student_logits, static_teacher_logits, video_teacher_logits, role_mask)
    student_logp = _masked_log_softmax(student_logits, mask)
    static_logp = _masked_log_softmax(static_teacher_logits.detach(), mask)
    video_mask = mask.unsqueeze(0).expand_as(video_teacher_logits)
    video_logp = _masked_log_softmax(video_teacher_logits.detach(), video_mask)

    centered_student = _center(student_logp.detach(), mask)
    centered_static = _center(static_logp, mask)
    centered_videos = _center(video_logp, video_mask)

    per_video_directions = centered_videos - centered_static.unsqueeze(0)
    temporal_direction, gate, accepted_fraction = _consensus_direction(
        per_video_directions,
        mask,
        consensus_ratio,
        minimum_consensus_cosine,
        eps,
    )

    privileged_teacher = centered_static + temporal_direction
    complete_correction = (privileged_teacher - centered_student) * mask
    dot = (complete_correction * temporal_direction).sum(dim=-1, keepdim=True)
    direction_energy = temporal_direction.square().sum(dim=-1, keepdim=True)
    projection_scale = dot.clamp_min(0.0) / (direction_energy + projection_ridge)
    projected_shift = projection_scale * temporal_direction
    shift_budget = torch.linalg.vector_norm(projected_shift, dim=-1, keepdim=True)

    support = temporal_direction.clamp_min(0.0)
    refutation = temporal_direction.clamp_max(0.0)
    support_score = (complete_correction * support).sum(dim=-1, keepdim=True).clamp_min(0.0)
    refutation_score = (complete_correction * refutation).sum(dim=-1, keepdim=True).clamp_min(0.0)
    score_total = support_score + refutation_score + eps
    support_weight = torch.minimum(
        support_score / score_total,
        torch.full_like(support_score, positive_support_cap),
    )
    refutation_weight = refutation_score / score_total

    support_unit = support / (torch.linalg.vector_norm(support, dim=-1, keepdim=True) + eps)
    refutation_unit = refutation / (torch.linalg.vector_norm(refutation, dim=-1, keepdim=True) + eps)
    attributed_shift = shift_budget * (
        support_weight * support_unit + refutation_weight * refutation_unit
    )
    attributed_shift = attributed_shift * gate.unsqueeze(-1) * mask
    attributed_shift = attributed_shift.clamp(-maximum_logit_shift, maximum_logit_shift)

    target_logits = (centered_student + attributed_shift).masked_fill(~mask, -torch.inf)
    target_probs = F.softmax(target_logits, dim=-1).detach()
    correction_norm = torch.linalg.vector_norm(complete_correction, dim=-1)
    explained_ratio = torch.linalg.vector_norm(attributed_shift, dim=-1) / (correction_norm + eps)
    explained_ratio = explained_ratio.clamp(0.0, 1.0) * gate.to(explained_ratio.dtype)

    return TemporalAttributionOutput(
        target_probs=target_probs,
        temporal_direction=temporal_direction.detach(),
        attributed_shift=attributed_shift.detach(),
        consensus_gate=gate.detach(),
        explained_ratio=explained_ratio.detach(),
        projection_scale=projection_scale.squeeze(-1).detach(),
        support_weight=support_weight.squeeze(-1).detach(),
        refutation_weight=refutation_weight.squeeze(-1).detach(),
        accepted_video_fraction=accepted_fraction.detach(),
    )


def compute_temporal_distillation_loss(
    student_logits: Tensor,
    static_teacher_logits: Tensor,
    video_teacher_logits: Tensor,
    *,
    labels: Tensor | None = None,
    role_mask: Tensor | None = None,
    distillation_weight: float = 1.0,
    static_anchor_weight: float = 0.1,
    supervised_weight: float = 1.0,
    ignore_index: int = -100,
    **target_kwargs: float,
) -> tuple[Tensor, dict[str, Tensor], TemporalAttributionOutput]:
    """Combine ACE supervision with temporal attribution distillation."""

    mask = _validate_inputs(student_logits, static_teacher_logits, video_teacher_logits, role_mask)
    output = build_temporal_target(
        student_logits,
        static_teacher_logits,
        video_teacher_logits,
        role_mask=mask,
        **target_kwargs,
    )
    student_logp = _masked_log_softmax(student_logits, mask)
    target_logp = output.target_probs.clamp_min(1e-12).log()
    distillation_loss = _js_divergence(target_logp, student_logp, mask).mean()

    static_logp = _masked_log_softmax(static_teacher_logits.detach(), mask)
    anchor_per_item = _js_divergence(static_logp, student_logp, mask)
    anchor_loss = ((1.0 - output.explained_ratio).detach() * anchor_per_item).mean()

    supervised_loss = student_logits.sum() * 0.0
    if labels is not None:
        if labels.shape != student_logits.shape[:-1]:
            raise ValueError("labels must match student_logits without the role dimension")
        supervised_loss = F.cross_entropy(
            student_logits.masked_fill(~mask, -1e9).reshape(-1, student_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    loss = (
        distillation_weight * distillation_loss
        + static_anchor_weight * anchor_loss
        + supervised_weight * supervised_loss
    )
    metrics = {
        "loss": loss.detach(),
        "loss_temporal": distillation_loss.detach(),
        "loss_static_anchor": anchor_loss.detach(),
        "loss_supervised": supervised_loss.detach(),
        "consensus_coverage": output.consensus_gate.to(student_logits.dtype).mean(),
        "accepted_video_fraction": output.accepted_video_fraction.mean(),
        "explained_ratio": output.explained_ratio.mean(),
        "projection_scale": output.projection_scale.mean(),
        "support_weight": output.support_weight.mean(),
        "refutation_weight": output.refutation_weight.mean(),
    }
    return loss, metrics, output
