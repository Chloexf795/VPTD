"""Offline paired diagnostics; never alter predictions, gold, or official metrics.

Run from repository root: python3 scripts/analyze_static50.py
Bootstrap intervals resample sentence/image clusters, not training seeds or documents.
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vptd_eae.benchmark import parse_predictions, match_boxes, iou, score_predictions
from vptd_eae.datasets import decode_json
from vptd_eae.metrics import evaluate_arguments, _safe_prf
from vptd_eae.event_schema import map_visual_role

OUT = ROOT / 'outputs/error_analysis_static50'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'outputs/gpu_20260923'
AFTER = ROOT / 'outputs/gpu_20260925'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def counts(row, args, role=True):
    if row['dataset'] == 'swig':
        return match_boxes(row['arguments'], args, role)
    key = 'argument_classification' if role else 'argument_identification'
    m = evaluate_arguments([row], [{'sample_id': row['sample_id'], 'arguments': args}])[key]
    return m['tp'], m['predicted'], m['gold']


def total(cs):
    return _safe_prf(*(sum(x[i] for x in cs) for i in range(3)))


def paired_stats(before, after):
    b, a = total(before), total(after)
    return {'before': b, 'after': a, 'delta_f1_pp': 100*(a['f1']-b['f1']),
            'delta_tp': a['tp']-b['tp'],
            'delta_fp': (a['predicted']-a['tp'])-(b['predicted']-b['tp'])}


def decode(row, pred):
    try:
        if pred['finish_reason'] != 'stop':
            return [], pred['finish_reason']
        return parse_predictions(pred['output'], row), None
    except ValueError as e:
        return [], str(e)


def pairs(args):
    return {(a['entity'].strip().casefold(), a['role'].strip().casefold()) for a in args}


def schema(row):
    if row['dataset'] == 'ace':
        prompt = row['prompt'].split('## Your Task')[-1]
        roles = re.search(r'\*\*Argument Roles\*\*: ([^\n]+)', prompt).group(1).split(',')
        entities = json.loads(re.search(r'\*\*Event Mentions\*\*: (\[[^\n]*\])', prompt).group(1))
        candidates = {x.get('entity', x.get('text', '')).casefold() for x in entities}
    else:
        roles = row['prompt'].split('Allowed roles: ')[1].split(',')
        candidates = set()
    return {map_visual_role(row['event_type'], r.strip()).casefold() for r in roles}, candidates


def bootstrap(before, after, rows):
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        key = re.sub(r'\W+', ' ', row['sentence'].casefold()).strip() if row['dataset']=='ace' else row['images'][0]
        clusters[key].append(i)
    b, a = (np.asarray([np.asarray(c)[indices].sum(axis=0) for indices in clusters.values()]) for c in (before, after))
    rng = np.random.default_rng(42)
    ds = []
    for _ in range(2000):
        indices = rng.integers(0, len(b), len(b))
        bs, ass = b[indices].sum(axis=0), a[indices].sum(axis=0)
        ds.append(100*(2*ass[0]/max(1, ass[1]+ass[2])-2*bs[0]/max(1, bs[1]+bs[2])))
    return {'resamples': 2000, 'seed': 42, 'unit': 'normalized sentence for ACE; image for SWiG; not document or training seed',
            'clusters': len(clusters),
            'percentile_95_delta_f1_pp': np.quantile(ds, [.025, .975]).tolist()}


report = {}
manifest = {}
for dataset in ('ace', 'swig'):
    gold_path = BASE / f'data_processed/{dataset}_dev.jsonl'
    pred_paths = [BASE / f'qwen_baseline/{dataset}_dev/predictions.jsonl',
                  AFTER / f'qwen_static50/{dataset}_dev/predictions.jsonl']
    rows = read(gold_path)
    raw_lists = [read(p) for p in pred_paths]
    for p in [gold_path, *pred_paths]:
        manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for path, preds in zip(pred_paths, raw_lists):
        original = json.loads(path.with_name('metrics.json').read_text())
        assert score_predictions(rows, preds) == {k:v for k,v in original.items() if k != 'run'}
    raw = [{p['sample_id']:p for p in ps} for ps in raw_lists]
    sample_results = []
    roles = defaultdict(lambda: [[], []])
    events = defaultdict(lambda: [[], []])
    cohorts = defaultdict(lambda: [[], []])
    errors = [Counter(), Counter()]
    diagnostic = [Counter(), Counter()]
    confusion = [Counter(), Counter()]
    transitions = Counter()
    pair_changes = Counter()
    all_counts = [[[], []], [[], []]]
    examples = []
    for row in rows:
        sid = row['sample_id']
        parsed, reasons = zip(*(decode(row, p[sid]) for p in raw))
        allowed, candidates = schema(row)
        sample_cs = [counts(row, p) for p in parsed]
        for v in range(2):
            all_counts[0][v].append(counts(row, parsed[v], False))
            all_counts[1][v].append(sample_cs[v])
            events[row['event_type']][v].append(sample_cs[v])
            cohort = 'gold_empty' if not row['arguments'] else 'gold_nonempty'
            cohorts[cohort][v].append(sample_cs[v])
            if dataset == 'ace':
                cohorts['with_prototype' if row['visual_prototypes'] else 'without_prototype'][v].append(sample_cs[v])
            if not any(reasons):
                cohorts['valid_in_both'][v].append(sample_cs[v])
            errors[v][reasons[v] or 'valid'] += 1
            diagnostic[v]['valid_empty_outputs'] += int(not reasons[v] and not parsed[v])
            diagnostic[v]['empty_on_nonempty_gold'] += int(not reasons[v] and not parsed[v] and bool(row['arguments']))
            diagnostic[v]['schema_disallowed_predictions'] += sum(a['role'].casefold() not in allowed for a in parsed[v])
            if dataset == 'ace':
                gs, ps = pairs(row['arguments']), pairs(parsed[v])
                gold_entities = {e for e,r in gs}
                for entity, role in ps-gs:
                    if entity in gold_entities:
                        diagnostic[v]['fp_gold_entity_wrong_role'] += 1
                        true_roles = {r for e,r in gs if e == entity}
                        if len(true_roles) == 1:
                            confusion[v][f'{next(iter(true_roles))} -> {role}'] += 1
                    elif entity in candidates:
                        diagnostic[v]['fp_candidate_entity_not_gold'] += 1
                    else:
                        diagnostic[v]['fp_entity_outside_candidates'] += 1
            else:
                for arg in parsed[v]:
                    nearby = [g for g in row['arguments'] if iou(arg['bbox'],g['bbox']) >= .5]
                    if not nearby:
                        diagnostic[v]['prediction_has_no_gold_iou50_edge'] += 1
                    elif not any(g['role'].casefold() == arg['role'].casefold() for g in nearby):
                        diagnostic[v]['prediction_has_only_wrong_role_iou50_edges'] += 1
                        true_roles = {g['role'].casefold() for g in nearby}
                        if len(true_roles)==1:
                            confusion[v][f'{next(iter(true_roles))} -> {arg["role"].casefold()}'] += 1
                    else:
                        diagnostic[v]['prediction_has_correct_role_iou50_edge'] += 1
        union_roles = {a['role'].casefold() for args in [row['arguments'], *parsed] for a in args}
        for role in union_roles:
            gold = {**row, 'arguments': [a for a in row['arguments'] if a['role'].casefold()==role]}
            for v in range(2):
                roles[role][v].append(counts(gold,[a for a in parsed[v] if a['role'].casefold()==role]))
        transitions[f'{"invalid" if reasons[0] else "valid"} -> {"invalid" if reasons[1] else "valid"}'] += 1
        if dataset == 'ace':
            g,b,a = pairs(row['arguments']),pairs(parsed[0]),pairs(parsed[1])
            pair_changes['new_correct_pairs'] += len((a&g)-(b&g))
            pair_changes['lost_correct_pairs'] += len((b&g)-(a&g))
            pair_changes['new_false_positive_pairs'] += len((a-g)-(b-g))
            pair_changes['removed_false_positive_pairs'] += len((b-g)-(a-g))
        stats=paired_stats([sample_cs[0]],[sample_cs[1]])
        sample_results.append({'sample_id':sid,'event_type':row['event_type'],**stats,
                               'before_error':reasons[0],'after_error':reasons[1]})
        examples.append({'sample_id':sid,'sentence':row.get('sentence'), 'images':row['images'],
            'event_type':row['event_type'],'gold':row['arguments'], 'before_arguments':parsed[0], 'after_arguments':parsed[1],
            'before_error':reasons[0],'after_error':reasons[1], **stats,
            'before_raw':raw[0][sid]['output'],'after_raw':raw[1][sid]['output']})
    groups=lambda x:{k:{'samples':len(v[0]),**paired_stats(*v)} for k,v in sorted(x.items())}
    overall={key:{**paired_stats(*all_counts[i]), 'paired_bootstrap':bootstrap(*all_counts[i], rows)}
             for i,key in enumerate(('argument_identification','argument_classification'))}
    by_role, by_event = groups(roles),groups(events)
    for version in ('before','after'):
        assert sum(x[version]['tp'] for x in by_role.values()) == overall['argument_classification'][version]['tp']
        assert sum(x[version]['predicted'] for x in by_role.values()) == overall['argument_classification'][version]['predicted']
        assert sum(x[version]['gold'] for x in by_role.values()) == overall['argument_classification'][version]['gold']
    if dataset=='ace':
        for v,version in enumerate(('before','after')):
            m=overall['argument_classification'][version]
            assert sum(diagnostic[v][k] for k in ('fp_gold_entity_wrong_role','fp_candidate_entity_not_gold','fp_entity_outside_candidates'))==m['predicted']-m['tp']
    top=lambda v:[{'gold_to_predicted':k,'count':n} for k,n in v.most_common(15)]
    report[dataset]={'samples':len(rows),'overall':overall,'by_role':by_role,'by_event':by_event,
        'cohorts':groups(cohorts),'output_errors':[dict(x) for x in errors],
        'validity_transitions':dict(transitions),'diagnostics':[dict(x) for x in diagnostic],
        'confusions':[top(x) for x in confusion],'ace_pair_changes':dict(pair_changes),
        'missing_trigger_field':sum(not r.get('trigger') for r in rows) if dataset=='ace' else None,
        'improved_samples':sum(x['delta_f1_pp']>0 for x in sample_results),
        'regressed_samples':sum(x['delta_f1_pp']<0 for x in sample_results)}
    if dataset=='ace':
        prompt_groups = defaultdict(list)
        for r in rows:
            prompt_groups[r['prompt']].append(r)
        contradictory = [g for g in prompt_groups.values() if len({tuple(sorted(pairs(r['arguments']))) for r in g})>1]
        report[dataset]['identical_prompt_conflicts']={'groups':len(contradictory),'samples':sum(map(len,contradictory))}
        (OUT/'ace_conflicting_inputs.json').write_text(json.dumps([
            {'prompt_sha256':hashlib.sha256(g[0]['prompt'].encode()).hexdigest(),
             'sentence':g[0]['sentence'],'event_type':g[0]['event_type'],
             'samples':[{'sample_id':r['sample_id'],'gold':r['arguments']} for r in g]}
            for g in contradictory],ensure_ascii=False,indent=2)+'\n')
    selected={'largest_regressions':sorted(examples,key=lambda x:(x['delta_f1_pp'],-x['delta_fp']))[:8],
        'largest_improvements':sorted(examples,key=lambda x:(-x['delta_f1_pp'],-x['delta_tp']))[:8],
        'most_added_false_positives':sorted(examples,key=lambda x:-x['delta_fp'])[:8],
        'new_invalid_outputs':[x for x in examples if x['after_error'] and not x['before_error']][:8]}
    (OUT/f'{dataset}_examples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    (OUT/f'{dataset}_samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sample_results))
    print(dataset,json.dumps({k:report[dataset][k] for k in ('overall','output_errors','diagnostics','ace_pair_changes','validity_transitions')},ensure_ascii=False))
(OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
source_audit = {}
for split in ('train', 'dev'):
    source = ROOT.parent / f'EMNLP2026/dataset/sft_ace_{split}.json'
    if not source.exists():
        continue
    records = json.loads(source.read_text())
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r['instruction']+'\n'+r.get('input','')].append((i, decode_json(r['output'])))
    conflicting = [g for g in groups.values() if len({tuple(sorted(set(
        (a.get('entity',a.get('text','')).strip().casefold(),a['role'].strip().casefold())
        for a in args))) for _,args in g})>1]
    source_audit[split]={'file':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'samples':len(records),'identical_raw_input_conflict_groups':len(conflicting),
        'affected_rows':sum(map(len,conflicting)),'examples':conflicting[:3]}
    manifest[str(source)]=source_audit[split]['sha256']
(OUT/'source_export_audit.json').write_text(json.dumps(source_audit,ensure_ascii=False,indent=2)+'\n')
(OUT/'input_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
