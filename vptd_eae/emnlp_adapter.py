"""Adapters between common generative EAE results and VPTD-EAE metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .event_schema import map_visual_role
from .result_parsing import extract_arguments


def _sample_id(item: Mapping[str, Any], index: int) -> str:
    return str(item.get("sample_id") or item.get("sentence_id") or item.get("id") or f"row-{index}")


def convert_emnlp_results(
    result_items: Sequence[Mapping[str, Any]],
    *,
    mode: str = "text",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert EMNLP-style result JSON to gold/pred metric records."""

    if mode not in {"text", "multi"}:
        raise ValueError("mode must be text or multi")
    gold_records: list[dict[str, Any]] = []
    predicted_records: list[dict[str, Any]] = []
    for index, item in enumerate(result_items):
        sample_id = _sample_id(item, index)
        event_type = str(item.get("event_type", ""))
        gold_arguments = []
        for argument in item.get("ground_truth", []):
            entity = argument.get("entity", argument.get("text", ""))
            role = argument.get("role", "")
            if entity and role:
                gold_arguments.append({"entity": entity, "role": role})

        raw_predictions = extract_arguments(item.get("output", ""))
        predictions = []
        for argument in raw_predictions:
            entity = argument.get("entity", "")
            role = argument.get("role", "")
            if not entity or not role:
                continue
            predictions.append({"entity": entity, "role": map_visual_role(event_type, role)})

        gold_records.append({"sample_id": sample_id, "arguments": gold_arguments})
        predicted_records.append({"sample_id": sample_id, "arguments": predictions})
    return gold_records, predicted_records
