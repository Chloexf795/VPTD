#!/usr/bin/env python3
"""Evaluate EMNLP-style EAE results with VPTD diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vptd_eae.data import read_json_or_jsonl
from vptd_eae.emnlp_adapter import convert_emnlp_results
from vptd_eae.metrics import evaluate_arguments, evaluate_directional_corrections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--mode", choices=("text", "multi"), default="text")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    items = read_json_or_jsonl(args.result)
    gold, predictions = convert_emnlp_results(items, mode=args.mode)
    report = {"metrics": evaluate_arguments(gold, predictions)}
    if args.baseline_result:
        baseline_items = read_json_or_jsonl(args.baseline_result)
        baseline_gold, baseline_predictions = convert_emnlp_results(baseline_items, mode=args.mode)
        if baseline_gold != gold:
            raise ValueError("baseline and temporal result files do not contain the same ordered gold samples")
        report["directional_corrections"] = evaluate_directional_corrections(
            gold, baseline_predictions, predictions
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
