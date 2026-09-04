"""Exact EAE Precision/Recall/F1 and directional-role diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


DEFAULT_DIRECTIONAL_PAIRS = (
    ("Agent", "Victim"),
    ("Attacker", "Target"),
    ("Giver", "Recipient"),
    ("Origin", "Destination"),
    ("Buyer", "Seller"),
)


def _safe_prf(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "predicted": predicted, "gold": gold}


def _argument_identity(sample_id: str, argument: Mapping[str, Any]) -> tuple[Any, ...]:
    if "start" in argument and "end" in argument:
        start = int(argument["start"])
        end = int(argument["end"])
        if end <= start:
            raise ValueError(f"argument end must exceed start in {sample_id!r}")
        return (sample_id, "span", start, end)
    entity = str(argument.get("entity", argument.get("text", ""))).strip().casefold()
    if not entity:
        raise ValueError(f"argument needs offsets or entity text in {sample_id!r}")
    return (sample_id, "entity", entity)


def _flatten(records: Iterable[Mapping[str, Any]], include_role: bool) -> set[tuple[Any, ...]]:
    flattened: set[tuple[Any, ...]] = set()
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("every metric record requires sample_id")
        for argument in record.get("arguments", []):
            key = _argument_identity(sample_id, argument)
            if include_role:
                role = str(argument.get("role", "")).strip().casefold()
                if not role:
                    raise ValueError(f"argument role is missing in {sample_id!r}")
                key += (role,)
            flattened.add(key)
    return flattened


def evaluate_arguments(
    gold_records: Iterable[Mapping[str, Any]],
    predicted_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute micro argument identification and role classification P/R/F1."""

    gold_records = list(gold_records)
    predicted_records = list(predicted_records)
    gold_ids = _flatten(gold_records, include_role=False)
    pred_ids = _flatten(predicted_records, include_role=False)
    gold_roles = _flatten(gold_records, include_role=True)
    pred_roles = _flatten(predicted_records, include_role=True)

    by_role: dict[str, dict[str, float | int]] = {}
    roles = sorted({key[-1] for key in gold_roles | pred_roles})
    for role in roles:
        role_gold = {key for key in gold_roles if key[-1] == role}
        role_pred = {key for key in pred_roles if key[-1] == role}
        by_role[role] = _safe_prf(len(role_gold & role_pred), len(role_pred), len(role_gold))
    return {
        "argument_identification": _safe_prf(len(gold_ids & pred_ids), len(pred_ids), len(gold_ids)),
        "argument_classification": _safe_prf(len(gold_roles & pred_roles), len(pred_roles), len(gold_roles)),
        "by_role": by_role,
    }


def _role_index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], str]:
    index: dict[tuple[Any, ...], str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        for argument in record.get("arguments", []):
            index[_argument_identity(sample_id, argument)] = str(argument["role"])
    return index


def evaluate_directional_corrections(
    gold_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    temporal_records: Sequence[Mapping[str, Any]],
    *,
    directional_pairs: Sequence[tuple[str, str]] = DEFAULT_DIRECTIONAL_PAIRS,
) -> dict[str, Any]:
    """Count corrected, remaining, and newly introduced role reversals."""

    opposite = {left.casefold(): right.casefold() for left, right in directional_pairs}
    opposite.update({right.casefold(): left.casefold() for left, right in directional_pairs})
    gold = {key: value.casefold() for key, value in _role_index(gold_records).items()}
    baseline = {key: value.casefold() for key, value in _role_index(baseline_records).items()}
    temporal = {key: value.casefold() for key, value in _role_index(temporal_records).items()}
    totals = defaultdict(int)
    by_pair: dict[str, dict[str, int | float]] = {}

    for left, right in directional_pairs:
        roles = {left.casefold(), right.casefold()}
        eligible = corrected = remaining = introduced = 0
        for identity, gold_role in gold.items():
            if gold_role not in roles:
                continue
            baseline_role = baseline.get(identity)
            temporal_role = temporal.get(identity)
            if baseline_role == opposite[gold_role]:
                eligible += 1
                if temporal_role == gold_role:
                    corrected += 1
                elif temporal_role == opposite[gold_role]:
                    remaining += 1
            elif baseline_role == gold_role and temporal_role == opposite[gold_role]:
                introduced += 1
        by_pair[f"{left}/{right}"] = {
            "baseline_reversals": eligible,
            "corrected": corrected,
            "remaining": remaining,
            "introduced": introduced,
            "correction_rate": corrected / eligible if eligible else 0.0,
        }
        totals["baseline_reversals"] += eligible
        totals["corrected"] += corrected
        totals["remaining"] += remaining
        totals["introduced"] += introduced
    eligible = totals["baseline_reversals"]
    return {"overall": {**dict(totals), "correction_rate": totals["corrected"] / eligible if eligible else 0.0}, "by_pair": by_pair}
