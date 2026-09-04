"""Standalone VPTD-EAE components for event argument extraction."""

from .temporal_attribution import (
    TemporalAttributionOutput,
    build_temporal_target,
    compute_temporal_distillation_loss,
)
from .role_support import ROLE_TO_INDEX, ROLE_VOCABULARY, build_role_mask, roles_for_event

__all__ = [
    "TemporalAttributionOutput",
    "build_temporal_target",
    "compute_temporal_distillation_loss",
    "ROLE_TO_INDEX",
    "ROLE_VOCABULARY",
    "build_role_mask",
    "roles_for_event",
]
