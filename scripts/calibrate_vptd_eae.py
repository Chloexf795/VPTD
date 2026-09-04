#!/usr/bin/env python3
"""Tune the non-NONE argument margin on ACE dev predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vptd_eae.calibration import apply_score_threshold, tune_global_threshold
from vptd_eae.data import read_json_or_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--scored-predictions", required=True, type=Path)
    parser.add_argument("--output-predictions", required=True, type=Path)
    parser.add_argument("--output-calibration", required=True, type=Path)
    args = parser.parse_args()

    gold = read_json_or_jsonl(args.gold)
    scored = read_json_or_jsonl(args.scored_predictions)
    result = tune_global_threshold(gold, scored)
    predictions = apply_score_threshold(scored, float(result["threshold"]))
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.output_predictions.open("w", encoding="utf-8") as handle:
        write_jsonl(predictions, handle)
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    args.output_calibration.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
