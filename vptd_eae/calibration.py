"""Development-set calibration for the non-NONE role margin."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import evaluate_arguments


def apply_score_threshold(records: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    output = []
    for record in records:
        kept = []
        for argument in record.get("arguments", []):
            if "score" not in argument:
                raise ValueError("every candidate argument requires score = best_role_logit - NONE_logit")
            if float(argument["score"]) >= threshold:
                kept.append(dict(argument))
        output.append({"sample_id": record["sample_id"], "arguments": kept})
    return output


def tune_global_threshold(
    gold_records: Sequence[Mapping[str, Any]],
    scored_records: Sequence[Mapping[str, Any]],
    *,
    metric: str = "argument_classification",
) -> dict[str, Any]:
    if metric not in {"argument_identification", "argument_classification"}:
        raise ValueError("unsupported metric")
    scores = sorted({float(a["score"]) for record in scored_records for a in record.get("arguments", [])})
    thresholds = [math.nextafter(scores[-1], math.inf), *scores] if scores else [math.inf]
    best = None
    best_key = None
    for threshold in thresholds:
        predictions = apply_score_threshold(scored_records, threshold)
        result = evaluate_arguments(gold_records, predictions)[metric]
        key = (float(result["f1"]), float(result["recall"]), float(result["precision"]), threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": threshold, "metrics": result, "points": len(thresholds)}
    assert best is not None
    return best
