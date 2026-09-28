#!/usr/bin/env python3
"""Export audited training JSONL to the pinned Vision-OPD Parquet contract."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vptd_eae.data import read_json_or_jsonl
from vptd_eae.datasets import image_checksums, audit_inputs, require_unambiguous_inputs
from vptd_eae.opd import build_opd_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--mode', choices=['ace-opsd', 'swig-opsd', 'ace-static-opd'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-ambiguous-ace', action='store_true', help='Explicit legacy reproduction only')
    args = parser.parse_args()
    rows = read_json_or_jsonl(args.data)
    input_audit = audit_inputs(rows)
    require_unambiguous_inputs(input_audit, allow_ambiguous_ace=args.allow_ambiguous_ace)
    records = build_opd_records(rows, args.mode)
    import pyarrow as pa
    import pyarrow.parquet as pq
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, args.output)
    restored = pq.read_table(args.output).to_pylist()
    if restored != records:
        raise RuntimeError('Parquet round trip changed training records')
    report = {'mode': args.mode, 'input_rows': len(rows), 'exported_rows': len(records),
              'input_audit': input_audit, 'ambiguity_override': args.allow_ambiguous_ace,
              'excluded_without_prototype': len(rows) - len(records),
              'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'parquet_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'image_sha256': image_checksums(i['path'] for r in records for i in r['images'] + r.get('bbox_images', [])),
              'input': str(args.data.resolve())}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
