#!/usr/bin/env python3
"""Audit normalized inputs without images, GPUs or changes to the source data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vptd_eae.data import read_json_or_jsonl
from vptd_eae.datasets import audit_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.data.resolve():
        parser.error('Audit output must not overwrite its source')
    report = audit_inputs(read_json_or_jsonl(args.data))
    report.update(source=str(args.data.resolve()), source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'conflicts' and not k.endswith('_sample_ids')}))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
