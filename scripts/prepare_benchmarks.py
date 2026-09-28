#!/usr/bin/env python3
"""Prepare local ACE/SWiG splits, training-only prototypes, and auditable manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vptd_eae.data import read_json_or_jsonl, write_jsonl
from vptd_eae.datasets import normalize_ace, normalize_swig, audit_splits, digest
from vptd_eae.converters import load_swig_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ('ace', 'swig'):
        for split in ('train', 'dev', 'test'):
            parser.add_argument(f'--{dataset}-{split}', type=Path)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--prototypes-per-event', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-missing-images', action='store_true', help='Audit only; image runs will remain blocked')
    parser.add_argument('--allow-overlap', action='store_true', help='Explicitly retain cross-split repeated content and flag manifest')
    parser.add_argument('--drop-train-overlap', action='store_true', help='Remove train content also found in dev/test; preserve evaluation splits')
    parser.add_argument('--allow-ambiguous-ace', action='store_true', help='Legacy reproduction only; retain and record ambiguous event inputs')
    args = parser.parse_args()
    if args.prototypes_per_event < 1:
        parser.error('--prototypes-per-event must be positive')
    if args.allow_overlap and args.drop_train_overlap:
        parser.error('Choose only one overlap policy')
    splits, sources, audits = {}, {}, {}
    mapping = load_swig_mapping(args.mapping) if args.mapping else None
    for dataset in ('ace', 'swig'):
        subset = {}
        for split in ('train', 'dev', 'test'):
            path = getattr(args, f'{dataset}_{split}')
            if path is None:
                continue
            raw = read_json_or_jsonl(path)
            rows = normalize_ace(raw, split) if dataset == 'ace' else normalize_swig(raw, split, args.image_root, mapping)
            subset[split] = rows
            splits[f'{dataset}_{split}'] = rows
            sources[f'{dataset}_{split}'] = {'path': str(path.resolve()),
                                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        before = audit_splits(subset)
        removed = []
        if args.drop_train_overlap and 'train' in subset:
            def content_key(row):
                return row['sentence'].strip().casefold() if dataset == 'ace' else Path(row['images'][0]).name
            held_out = {content_key(r) for s in ('dev', 'test') for r in subset.get(s, [])}
            removed = [r['sample_id'] for r in subset['train'] if content_key(r) in held_out]
            subset['train'] = [r for r in subset['train'] if content_key(r) not in held_out]
            splits[f'{dataset}_train'] = subset['train']
        audits[dataset] = audit_splits(subset)
        audits[dataset]['original_overlap_pairs'] = before['overlap_pairs']
        audits[dataset]['removed_train_sample_ids'] = removed
    if not splits:
        parser.error('Provide at least one --ace-* or --swig-* input')
    pool = {}
    for row in splits.get('swig_train', []):
        pool.setdefault(row['event_type'], {})[row['images'][0]] = row
    coverage = {}
    for name, rows in splits.items():
        if not name.startswith('ace_'):
            continue
        for row in rows:
            candidates = list(pool.get(row['event_type'], {}).values())
            selected = sorted(candidates, key=lambda p: digest([args.seed, row['sample_id'], Path(p['images'][0]).name]))[:args.prototypes_per_event]
            row['visual_prototypes'] = [{'image': p['images'][0], 'source_sample_id': p['sample_id'],
                                         'source_split': 'train', 'event_type': p['event_type']} for p in selected]
            row['alignment'] = {'kind': 'event_type_prototype', 'instance_aligned': False,
                                'label_source': 'ACE', 'visual_source': 'SWiG-train'}
        coverage[name] = {'total': len(rows), 'paired': sum(bool(r['visual_prototypes']) for r in rows)}
    missing = sum(s['missing_images'] for a in audits.values() for s in a['splits'].values())
    overlaps = sum(sum(a['overlap_pairs'].values()) for a in audits.values())
    ambiguous = any(not s['input_audit']['ready'] for a in audits.values() for s in a['splits'].values())
    non_ace_conflicts = any(c['dataset'] != 'ace' for a in audits.values() for s in a['splits'].values()
                            for c in s['input_audit']['conflicts'])
    manifest = {'schema_version': 2, 'sources': sources, 'audits': audits, 'prototype_coverage': coverage,
                'seed': args.seed, 'prototypes_per_event': args.prototypes_per_event,
                'image_root': str(args.image_root.resolve()) if args.image_root else None,
                'missing_images': missing, 'cross_split_overlaps': overlaps,
                'image_runs_ready': missing == 0, 'overlap_override': args.allow_overlap,
                'input_runs_ready': not ambiguous, 'ambiguity_override': args.allow_ambiguous_ace,
                'protocol_note': 'ACE surface EAE; SWiG ACE-mapped grounding. Not standard ACE span or full SWiG situation recognition.'}
    if args.mapping:
        manifest['mapping_sha256'] = hashlib.sha256(args.mapping.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (missing and not args.allow_missing_images) or (overlaps and not args.allow_overlap) or non_ace_conflicts or (ambiguous and not args.allow_ambiguous_ace):
        (args.output_dir / 'audit.failed.json').write_text(json.dumps(manifest, indent=2) + '\n')
        raise ValueError(f'Preparation blocked: missing images={missing}, cross-split overlaps={overlaps}, ambiguous inputs={ambiguous}. See audit.failed.json.')
    for name, rows in splits.items():
        path = args.output_dir / f'{name}.jsonl'
        with path.open('w', encoding='utf-8') as stream:
            write_jsonl(rows, stream)
        sources[name]['prepared_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': {k: len(v) for k, v in splits.items()}, 'missing_images': missing,
                      'cross_split_overlaps': overlaps, 'prototype_coverage': coverage}, indent=2))


if __name__ == '__main__':
    main()
