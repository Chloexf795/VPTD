"""Dataset bridge to the pinned, unmodified Vision-OPD/verl runtime.

Source: https://github.com/VisionOPD/Vision-OPD/blob/c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471/verl/trainer/ppo/ray_trainer.py
"""
from __future__ import annotations

import json
from pathlib import Path


def build_opd_records(rows: list[dict], mode: str) -> list[dict]:
    if mode not in {'ace-opsd', 'swig-opsd', 'ace-static-opd'}:
        raise ValueError(f'Unknown OPD mode: {mode}')
    output = []
    for index, row in enumerate(rows):
        if row['split'] != 'train' or row['dataset'] != mode.split('-')[0]:
            raise ValueError('Training requires train rows of the selected dataset')
        images = row['images']
        teacher_images = [p['image'] for p in row.get('visual_prototypes', [])] if mode == 'ace-static-opd' else []
        if mode == 'ace-static-opd' and not teacher_images:
            # Coverage is explicitly reported by the exporting CLI.
            continue
        for path in images + teacher_images:
            if not Path(path).is_file():
                raise FileNotFoundError(f'Required training image is missing: {path}')
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
        prompt = '\n'.join(['<image>'] * len(images) + [row['prompt']])
        answer = json.dumps(row['arguments'], ensure_ascii=False)
        record = {'data_source': 'vptd_' + row['dataset'],
                  'prompt': [{'role': 'user', 'content': prompt}],
                  'images': [{'path': p} for p in images],
                  'ability': 'event_argument_extraction',
                  'reward_model': {'style': 'none', 'ground_truth': answer},
                  'extra_info': {'index': index, 'sample_id': row['sample_id'], 'answer': answer,
                                 'question': row['prompt'], 'split': 'train'}}
        if teacher_images:
            teacher_prompt = (row['prompt'] + '\nThe following images are same-event-type visual prototypes, '
                              'NOT photographs of the sentence. Use them only to clarify role relations. '
                              'Extract entities exclusively from the sentence; do not invent participants.\n'
                              + '\n'.join(['<image>'] * len(teacher_images)))
            record['teacher_prompt'] = [{'role': 'user', 'content': teacher_prompt}]
            record['bbox_images'] = [{'path': p} for p in teacher_images]
        output.append(record)
    if not output:
        raise ValueError('No eligible training rows for selected mode')
    return output
