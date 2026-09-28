"""Audited surface EAE and mapped visual grounding metrics (no LLM judge)."""
from __future__ import annotations

from .datasets import check_box, decode_json, digest
from .event_schema import map_visual_role
from .metrics import evaluate_arguments, _safe_prf


def prediction_identity(row: dict) -> str:
    return digest(row)


def parse_answer(text: str) -> list[dict]:
    try:
        result = decode_json(text)
    except (TypeError, ValueError) as exc:
        raise ValueError('Model output is not a JSON array') from exc
    if not isinstance(result, list) or any(not isinstance(x, dict) for x in result):
        raise ValueError('Model output must be a JSON array of argument objects')
    return result


def parse_predictions(text: str, row: dict) -> list[dict]:
    result = []
    for arg in parse_answer(text):
        role = arg.get('role')
        if not isinstance(role, str) or not role.strip():
            raise ValueError('Predicted argument requires a role')
        if row['dataset'] == 'ace':
            entity = arg.get('entity', arg.get('text'))
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError('Predicted ACE argument requires entity text')
            # Preserve the existing EMNLP adapter's prediction-role aliases.
            item = {'role': map_visual_role(row['event_type'], role), 'entity': entity.strip()}
            if item not in result:
                result.append(item)
        else:
            boxes = arg.get('bboxes', arg.get('bbox'))
            if not isinstance(boxes, list):
                raise ValueError('Predicted SWiG argument requires bbox/bboxes')
            if boxes and isinstance(boxes[0], (int, float)):
                boxes = [boxes]
            for box in boxes:
                item = {'role': map_visual_role(row['event_type'], role), 'bbox': check_box(box)}
                if item not in result:
                    result.append(item)
    return result


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def match_boxes(gold, predictions, include_role: bool) -> tuple[int, int, int]:
    # Maximum-cardinality bipartite matching: a gold box can be credited only once.
    def unique(items):
        found = {}
        for x in items:
            key = tuple(x['bbox']) + ((x['role'].casefold(),) if include_role else ())
            found[key] = x
        return list(found.values())
    gold, predictions = unique(gold), unique(predictions)
    edges = [[j for j, g in enumerate(gold) if iou(p['bbox'], g['bbox']) >= 0.5
              and (not include_role or p['role'].casefold() == g['role'].casefold())] for p in predictions]
    matched = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    count = sum(augment(i, set()) for i in range(len(predictions)))
    return count, len(predictions), len(gold)


def score_predictions(records: list[dict], predictions: list[dict]) -> dict:
    if not records:
        raise ValueError('Cannot score an empty dataset')
    expected = {r['sample_id']: r for r in records}
    actual = {r['sample_id']: r for r in predictions}
    if len(expected) != len(records) or len(actual) != len(predictions) or expected.keys() != actual.keys():
        raise ValueError('Scoring requires unique IDs and exactly the complete expected sample set')
    if len({(r['dataset'], r['protocol']) for r in records}) != 1:
        raise ValueError('Score each dataset and protocol separately')
    gold, parsed, invalid, truncated = [], [], 0, 0
    for row in records:
        pred = actual[row['sample_id']]
        if pred.get('status') != 'ok' or pred.get('input_sha256') != prediction_identity(row):
            raise ValueError(f'Failed or stale prediction: {row["sample_id"]}')
        try:
            if pred.get('finish_reason', 'stop') != 'stop':
                truncated += pred.get('finish_reason') == 'length'
                raise ValueError('Token-budget exhaustion is an invalid model output, not a skipped sample')
            arguments = parse_predictions(pred['output'], row)
        except ValueError:
            arguments = []
            invalid += 1
        gold.append({'sample_id': row['sample_id'], 'arguments': row['arguments']})
        parsed.append({'sample_id': row['sample_id'], 'arguments': arguments})
    if records[0]['dataset'] == 'ace':
        metrics = evaluate_arguments(gold, parsed)
    else:
        metrics = {}
        for key, include_role in [('argument_identification', False), ('argument_classification', True)]:
            counts = [match_boxes(g['arguments'], p['arguments'], include_role) for g, p in zip(gold, parsed)]
            totals = [sum(c[i] for c in counts) for i in range(3)]
            metrics[key] = _safe_prf(*totals)
    return {'protocol': records[0]['protocol'], 'samples': len(records), 'complete': True,
            'invalid_outputs': invalid, 'truncated_outputs': truncated,
            'truncation_policy': 'count-as-invalid-with-full-gold-denominator', 'metrics': metrics}
