"""Import local ACE/SWiG exports without mixing their supervision contracts."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .converters import convert_ace_records, convert_swig_records
from .event_schema import map_visual_role, normalize_event_type, roles_for_event_type


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def image_checksums(paths) -> dict[str, str]:
    """Bind a run to the actual local image bytes, not just mutable filenames."""
    return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(set(paths))}


def ace_roles(event: str) -> tuple[str, ...]:
    if event == 'CONTACT||MEET':
        return ('Entity', 'Place')
    if event == 'TRANSACTION||TRANSFER|MONEY':
        return ('Giver', 'Recipient', 'Money', 'Beneficiary', 'Place')
    return roles_for_event_type(event)


def decode_json(text: str):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def field(prompt: str, label: str) -> str:
    # Match the final concrete input, not examples earlier in the instruction.
    matches = re.findall(r'(?:\*\*)?' + re.escape(label) + r'(?:\*\*)?:\s*([^\n]+)', prompt)
    if not matches:
        raise ValueError(f'Missing {label!r} in source prompt')
    return matches[-1].strip()


def check_box(box) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'Expected xyxy box: {box!r}')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in box):
        raise ValueError(f'Non-numeric box: {box!r}')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError(f'Box outside normalized 0..1000 coordinates: {box!r}')
    return list(box)


def normalize_ace(records: list[dict], split: str) -> list[dict]:
    output = []
    if not records:
        raise ValueError('ACE input is empty')
    alpaca = 'instruction' in records[0]
    if not alpaca:
        records = convert_ace_records(records)
    for index, row in enumerate(records):
        if alpaca:
            prompt = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
            event = normalize_event_type(field(prompt, 'Event Type'))
            sentence = field(prompt, 'Input Sentence')
            arguments = decode_json(row['output'])
            # Repair the final input's empty/wrong schema using a fixed ontology,
            # independent of this sample's gold. Keep all candidate entities intact.
            matches = list(re.finditer(r'\*\*Argument Roles\*\*:[^\n]*', prompt))
            roles = ace_roles(event)
            if matches and roles:
                last = matches[-1]
                prompt = prompt[:last.start()] + '**Argument Roles**: ' + ', '.join(roles) + prompt[last.end():]
            protocol = 'ace-export-surface-eae-given-event-and-entities-schema-v1'
            trigger = ''  # The source export does not contain an instance trigger.
        else:
            event, sentence, trigger = row['event_type'], row['sentence'], row['trigger']
            arguments = row['arguments']
            annotations = row['source_annotations']
            source_trigger = annotations['trigger']
            prompt = ('Extract the arguments of the specified event from the sentence. '
                      'Return only a JSON array of {"role": "...", "entity": "exact text span"}.\n'
                      f'Event type: {event}\nTrigger: {trigger}\nSentence: {sentence}')
            if 'start' in source_trigger and 'end' in source_trigger:
                prompt += '\nTrigger source position (offset unit as supplied): ' + json.dumps(
                    {k: source_trigger[k] for k in ('start', 'end', 'offset_unit') if k in source_trigger})
            prompt += '\nAllowed roles: ' + ', '.join(ace_roles(event))
            if row['entity_mentions']:
                prompt += '\nCandidate entities: ' + json.dumps(row['entity_mentions'], ensure_ascii=False)
            protocol = 'ace-processed-surface-eae-given-trigger-v2'
        if not isinstance(arguments, list):
            raise ValueError(f'ACE row {index}: gold must be a list')
        gold = []
        for argument in arguments:
            entity = argument.get('entity', argument.get('text', ''))
            role = argument.get('role', '')
            if not isinstance(entity, str) or not entity.strip() or not isinstance(role, str) or not role.strip():
                raise ValueError(f'ACE row {index}: invalid argument')
            gold.append({'role': role.strip(), 'entity': entity.strip()})
        output.append(dict(sample_id=f'ace:{split}:{index}:{digest(prompt)[:12]}', split=split,
                           dataset='ace', protocol=protocol, prompt=prompt, sentence=sentence,
                           trigger=trigger, event_type=event, arguments=gold, images=[],
                           visual_prototypes=[]))
        if not alpaca:
            output[-1]['source_annotations'] = row['source_annotations']
    if not output:
        raise ValueError('ACE input contains no event mentions')
    return output


def normalize_swig(records, split: str, image_root: Path | None = None, mapping=None) -> list[dict]:
    if not records:
        raise ValueError('SWiG input is empty')
    alpaca = isinstance(records, list) and 'instruction' in records[0]
    if not alpaca and isinstance(records, dict):
        if mapping is None:
            raise ValueError('Processed SWiG requires --mapping')
        records = convert_swig_records(records, mapping)
    output = []
    for index, row in enumerate(records):
        if alpaca:
            event = normalize_event_type(field(row['instruction'], 'Event Type'))
            paths = row.get('images', [])
            if len(paths) != 1:
                raise ValueError('SWiG rows must contain exactly one image')
            path = paths[0]
            raw = decode_json(row['output'])
            if isinstance(raw, dict) and isinstance(raw.get('objects'), list):
                raw = [{'role': x['category'], 'bboxes': [x['bbox']]} for x in raw['objects']]
            if not isinstance(raw, list):
                raise ValueError(f'Unsupported SWiG answer at row {index}')
        else:
            event = normalize_event_type(row['event_type'])
            path = row['image']
            raw = [{'role': role, 'bboxes': boxes} for role, boxes in row['bounding_boxes'].items()]
        image = str((image_root / Path(path).name).resolve()) if image_root else str(Path(path).resolve())
        gold = []
        for argument in raw:
            role = map_visual_role(event, argument['role'])
            boxes = argument.get('bboxes', argument.get('bbox', []))
            if boxes and isinstance(boxes[0], (int, float)):
                boxes = [boxes]
            for box in boxes:
                if box == []:
                    continue
                item = {'role': role, 'bbox': check_box(box)}
                if item not in gold:
                    gold.append(item)
        roles = sorted({map_visual_role(event, r) for r in roles_for_event_type(event)})
        # The local ACE-mapped SWiG export labels the transferred visual object as
        # Instrument. Keep this visual schema separate from ACE's Money role.
        if event == 'TRANSACTION||TRANSFER|MONEY':
            roles = ['Giver', 'Recipient', 'Instrument', 'Money', 'Beneficiary', 'Place']
        if not roles:
            raise ValueError(f'No configured SWiG role schema for {event}')
        unknown = {x['role'] for x in gold} - set(roles)
        if unknown:
            # Schemas are fixed for all rows of the event, never inferred from row gold.
            raise ValueError(f'Gold roles outside SWiG schema for {event}: {sorted(unknown)}')
        prompt = ('Locate the arguments of the specified event in this image. Return only a JSON array '
                  'of {"role": "...", "bbox": [x1,y1,x2,y2]}. Coordinates must be normalized '
                  'to 0..1000. Omit absent roles.\n'
                  f'Event type: {event}\nAllowed roles: {", ".join(roles)}')
        output.append(dict(sample_id=f'swig:{split}:{index}:{Path(path).name}', split=split,
                           dataset='swig', protocol='swig-ace-mapped-grounding-iou50-given-event',
                           event_type=event, prompt=prompt, arguments=gold, images=[image],
                           visual_prototypes=[]))
    return output


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    signatures = {}
    reports = {}
    for name, rows in splits.items():
        # Sentence-level audit catches same sentence with different event prompts.
        values = [digest(r['sentence'].strip().casefold()) if r['dataset'] == 'ace'
                  else Path(r['images'][0]).name for r in rows]
        signatures[name] = set(values)
        missing = sorted({p for r in rows for p in r.get('images', []) if not Path(p).is_file()})
        reports[name] = {'rows': len(rows), 'repeated_content_rows': len(values) - len(set(values)),
                         'missing_images': len(missing), 'missing_image_examples': missing[:5],
                         'events': dict(Counter(r['event_type'] for r in rows))}
        reports[name]['input_audit'] = audit_inputs(rows)
        reports[name]['ace_gold_roles_outside_prompt_schema'] = dict(Counter(
            f'{r["event_type"]}/{a["role"]}' for r in rows if r['dataset'] == 'ace'
            and ace_roles(r['event_type']) for a in r['arguments'] if a['role'] not in ace_roles(r['event_type'])))
    names = list(splits)
    overlaps = {f'{a}:{b}': len(signatures[a] & signatures[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    return {'splits': reports, 'overlap_pairs': overlaps}


def audit_inputs(rows: list[dict]) -> dict:
    """Detect contradictory targets for identical visible inputs, without rewriting gold."""
    groups = defaultdict(list)
    missing, unlocated = [], []
    datasets = {}
    for row in rows:
        # Hidden IDs, gold and prototype retrieval metadata cannot distinguish
        # inputs actually supplied to the student at evaluation time.
        signature = digest([row['dataset'], row['prompt'], row.get('images', [])])
        datasets[signature] = row['dataset']
        gold = []
        for arg in row['arguments']:
            identity = (('bbox', *arg['bbox']) if 'bbox' in arg else
                        ('span', arg['start'], arg['end']) if 'start' in arg and 'end' in arg else
                        ('entity', arg.get('entity', arg.get('text', '')).strip().casefold()))
            gold.append((*identity, arg['role'].strip().casefold()))
        groups[signature].append((row['sample_id'], tuple(sorted(set(gold)))))
        if row['dataset'] == 'ace':
            trigger = row.get('trigger') or ''
            if not isinstance(trigger, str) or not trigger.strip():
                missing.append(row['sample_id'])
            elif 'sentence' in row:
                position = row.get('source_annotations', {}).get('trigger', {})
                located = all(type(position.get(k)) is int for k in ('start', 'end')) and 0 <= position['start'] < position['end']
                occurrences = re.findall(r'(?<!\w)' + re.escape(trigger.strip()) + r'(?!\w)', row['sentence'], re.IGNORECASE)
                if len(occurrences) != 1 and not located:
                    unlocated.append(row['sample_id'])
    conflicts = [{'input_sha256': k, 'dataset': datasets[k], 'sample_ids': [sid for sid, _ in values],
                  'gold_variants': len({gold for _, gold in values})}
                 for k, values in sorted(groups.items()) if len({gold for _, gold in values}) > 1]
    return {'version': 1, 'rows': len(rows), 'conflicting_groups': len(conflicts),
            'conflicting_rows': sum(len(c['sample_ids']) for c in conflicts), 'conflicts': conflicts,
            'missing_ace_trigger_rows': len(missing), 'missing_ace_trigger_sample_ids': missing,
            'unlocated_ace_trigger_rows': len(unlocated), 'unlocated_ace_trigger_sample_ids': unlocated,
            'ready': bool(rows) and not conflicts and not missing and not unlocated}


def require_unambiguous_inputs(report: dict, *, allow_ambiguous_ace: bool = False):
    if not report['rows']:
        raise ValueError('No input rows to audit')
    non_ace_conflicts = any(c['dataset'] != 'ace' for c in report['conflicts'])
    if not report['ready'] and (not allow_ambiguous_ace or non_ace_conflicts):
        raise ValueError('Ambiguous inputs: '
                         f'{report["conflicting_groups"]} conflicting input groups; '
                         f'{report["missing_ace_trigger_rows"]} ACE rows missing trigger. '
                         f'{report.get("unlocated_ace_trigger_rows", 0)} ACE rows lack an unambiguous trigger location. '
                         'Restore source event annotations; --allow-ambiguous-ace is for explicit legacy reproduction only.')
