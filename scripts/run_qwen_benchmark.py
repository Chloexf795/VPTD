#!/usr/bin/env python3
"""Run or resume Qwen predictions and score only complete, identity-checked runs."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vptd_eae.data import read_json_or_jsonl
from vptd_eae.datasets import image_checksums, audit_inputs, require_unambiguous_inputs
from vptd_eae.benchmark import prediction_identity, score_predictions
from vptd_eae.qwen import QwenAPI, QwenTransformers, messages_for, output_schema


def generate_predictions(backend, rows, workers):
    """Keep at most workers requests in flight; only the caller writes results."""
    if workers == 1:
        for row in rows:
            yield row, backend.generate(row)
        return
    pending_rows = iter(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit_next():
            row = next(pending_rows, None)
            if row is not None:
                active[pool.submit(backend.generate, row)] = row

        for _ in range(workers):
            submit_next()
        try:
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    row = active.pop(future)
                    yield row, future.result()
                    submit_next()
        finally:
            for future in active:
                future.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--revision', help='Hugging Face revision; pin a commit for publication')
    parser.add_argument('--backend', choices=['transformers', 'api'], default='transformers')
    parser.add_argument('--base-url', help='Explicit Qwen/vLLM /v1 endpoint; key from QWEN_API_KEY')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--workers', type=int, default=1, help='Concurrent API requests (1–32); Transformers uses 1')
    parser.add_argument('--limit', type=int, help='Diagnostic subset; never writes full-dataset metrics')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-ambiguous-ace', action='store_true', help='Explicit legacy reproduction only; recorded in run configuration')
    parser.add_argument('--output-policy', choices=['legacy', 'schema-v1'], default='legacy', help='schema-v1 requires a vLLM API backend; no posthoc filtering')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='vLLM API parameter, >=1; choose on train before dev comparisons')
    args = parser.parse_args()
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1:
        parser.error('--repetition-penalty must be finite and >=1')
    if args.backend != 'api' and (args.output_policy != 'legacy' or args.repetition_penalty != 1.0):
        parser.error('Output constraints and repetition penalty currently require --backend api')
    if not 1 <= args.workers <= 32 or args.backend == 'transformers' and args.workers != 1:
        parser.error('--workers must be 1–32 for API and 1 for Transformers')
    if args.limit is not None and args.limit < 1 or args.max_tokens < 1 or args.max_input_tokens < 1:
        parser.error('Token budgets and --limit must be positive')
    rows = read_json_or_jsonl(args.data)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Dataset must be nonempty with unique sample IDs')
    if len({(r['dataset'], r['protocol'], r['split']) for r in rows}) != 1:
        raise ValueError('Use a single dataset, split, and protocol per run')
    selected = rows[:args.limit] if args.limit else rows
    predictions_path = args.output_dir / 'predictions.jsonl'
    if args.score_only:
        report = score_predictions(rows, read_json_or_jsonl(predictions_path))
        report['run'] = json.loads((args.output_dir / 'run.json').read_text())
        (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    input_audit = audit_inputs(rows)
    require_unambiguous_inputs(input_audit, allow_ambiguous_ace=args.allow_ambiguous_ace)
    # Inspect all selected images before any model loading or paid API request.
    for row in selected:
        messages_for(row, api=False)
        if args.output_policy == 'schema-v1':
            output_schema(row)
    if args.dry_run:
        print(json.dumps({'data': str(args.data.resolve()), 'samples': len(rows), 'selected': len(selected),
                          'model': args.model, 'backend': args.backend,
                          'protocol': rows[0]['protocol'], 'image_preflight': 'passed'}, indent=2))
        return
    if args.backend == 'api' and not args.base_url:
        parser.error('--backend api requires --base-url')
    config = {k: getattr(args, k) for k in ('model', 'revision', 'backend', 'base_url', 'device',
                                             'max_tokens', 'max_input_tokens', 'workers', 'output_policy', 'repetition_penalty')}
    config.update(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  allow_ambiguous_ace=args.allow_ambiguous_ace,
                  seed=42, temperature=0, protocol=rows[0]['protocol'], split=rows[0]['split'],
                  image_sha256=image_checksums(p for r in rows for p in r['images']))
    saved_path = args.output_dir / 'run.json'
    previous = []
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        old_config = dict(saved['config'])
        # Historical runs need explicit reproduction only when the audit fails.
        old_config.setdefault('allow_ambiguous_ace', not input_audit['ready'])
        old_config.setdefault('output_policy', 'legacy')
        old_config.setdefault('repetition_penalty', 1.0)
        if not args.resume or old_config != config:
            raise ValueError('Run directory exists; use --resume with identical settings or a new output directory')
        if predictions_path.exists():
            previous = read_json_or_jsonl(predictions_path)
    elif predictions_path.exists():
        raise ValueError('Predictions without run.json cannot be resumed')
    expected = {r['sample_id']: r for r in rows}
    done = set()
    for pred in previous:
        sid = pred['sample_id']
        if sid in done or sid not in expected or pred.get('status') != 'ok' or pred['input_sha256'] != prediction_identity(expected[sid]):
            raise ValueError('Cannot resume duplicate, failed, foreign, or stale predictions')
        done.add(sid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not saved_path.exists():
        versions = {}
        for package in ('torch', 'transformers', 'openai', 'vllm'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        saved = {'config': config, 'versions': versions, 'expected_samples': len(rows),
                 'input_audit': input_audit,
                 'data': str(args.data.resolve()), 'created_at_unix': time.time()}
        saved_path.write_text(json.dumps(saved, indent=2) + '\n')
    elif 'input_audit' not in saved:
        (args.output_dir / 'resume_input_audit.json').write_text(json.dumps(
            {'input_audit': input_audit, 'ambiguity_override': args.allow_ambiguous_ace}, indent=2)+'\n')
    pending = [r for r in selected if r['sample_id'] not in done]
    if pending:
        backend_type = QwenAPI if args.backend == 'api' else QwenTransformers
        backend = backend_type(**vars(args))
        with predictions_path.open('a', encoding='utf-8') as stream:
            for row, generated in generate_predictions(backend, pending, args.workers):
                # Transport failures stop the run. They are never stored as empty answers.
                prediction = {'sample_id': row['sample_id'], 'input_sha256': prediction_identity(row),
                              'status': 'ok', **generated}
                stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                stream.flush()
                previous.append(prediction)
                print(f'{len(previous)}/{len(rows)} {row["sample_id"]}', flush=True)
    if len(previous) == len(rows):
        report = score_predictions(rows, previous)
        report['run'] = saved
        (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['metrics'], indent=2))
    else:
        print(f'Diagnostic subset: {len(previous)}/{len(rows)}; full-dataset metrics not produced. Resume without --limit.')


if __name__ == '__main__':
    main()
