#!/usr/bin/env python3
"""Build leakage-safe VPTD-EAE records from processed ACE and SWiG files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vptd_eae.converters import convert_ace_records, convert_swig_records, load_swig_mapping
from vptd_eae.data import attach_event_type_prototypes, read_json_or_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace", required=True, type=Path, help="ACE processed JSON used by ACEConverter")
    parser.add_argument("--swig", required=True, type=Path, help="SWiG processed JSON used by SwigConverter")
    parser.add_argument("--mapping", required=True, type=Path, help="Existing ace_sr_mapping.txt")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--prototypes-per-event", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_ace = read_json_or_jsonl(args.ace)
    raw_swig = read_json_or_jsonl(args.swig)
    ace_events = convert_ace_records(raw_ace)
    swig_frames = convert_swig_records(raw_swig, load_swig_mapping(args.mapping))
    records, stats = attach_event_type_prototypes(
        ace_events,
        swig_frames,
        split=args.split,
        image_root=args.image_root,
        prototypes_per_event=args.prototypes_per_event,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        write_jsonl(records, handle)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.to_dict()
    args.stats.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
