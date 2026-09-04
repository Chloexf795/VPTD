"""Event-specific role supports for the standalone VPTD-EAE package."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .event_schema import EVENT_SCHEMAS, normalize_event_type


NONE_ROLE = "NONE"


def build_role_vocabulary() -> tuple[str, ...]:
    roles = {NONE_ROLE}
    for schema in EVENT_SCHEMAS.values():
        roles.update(schema)
    return tuple(sorted(roles))


ROLE_VOCABULARY = build_role_vocabulary()
ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLE_VOCABULARY)}


def roles_for_event(event_type: str) -> tuple[str, ...]:
    normalized = normalize_event_type(event_type)
    roles = EVENT_SCHEMAS.get(normalized, ())
    return tuple([*roles, NONE_ROLE])


def build_role_mask(event_types: Sequence[str], *, device: torch.device | str | None = None) -> Tensor:
    """Return ``[batch, global_roles]`` mask for heterogeneous event types."""

    mask = torch.zeros((len(event_types), len(ROLE_VOCABULARY)), dtype=torch.bool, device=device)
    for row, event_type in enumerate(event_types):
        roles = roles_for_event(event_type)
        if len(roles) == 1:
            raise ValueError(f"unknown or unsupported event type: {event_type!r}")
        for role in roles:
            mask[row, ROLE_TO_INDEX[role]] = True
    return mask


def role_labels_to_indices(labels: Sequence[str], *, device: torch.device | str | None = None) -> Tensor:
    try:
        values = [ROLE_TO_INDEX[label] for label in labels]
    except KeyError as exc:
        raise ValueError(f"unknown EAE role: {exc.args[0]!r}") from exc
    return torch.tensor(values, dtype=torch.long, device=device)
