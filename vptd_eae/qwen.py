"""Qwen inference via Transformers or an explicitly configured compatible endpoint.

Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
from pathlib import Path
import re


def output_schema(row: dict) -> dict:
    """Gold-free JSON Schema for the pinned vLLM structured-output API.

    https://docs.vllm.ai/en/v0.18.0/features/structured_outputs/
    Does not claim to enforce semantic relevance, unique boxes or x1 < x2.
    """
    prompt = row['prompt']
    role_lines = re.findall(r'(?:\*\*Argument Roles\*\*|Allowed roles):\s*([^\n]+)', prompt)
    if not role_lines:
        raise ValueError('Structured output requires an explicit allowed-role list in the input')
    roles = sorted({r.strip() for r in role_lines[-1].split(',') if r.strip()})
    if not roles:
        raise ValueError('Structured output requires nonempty allowed roles')
    properties = {'role': {'type': 'string', 'enum': roles}}
    if row['dataset'] == 'ace':
        matches = re.findall(r'(?:\*\*Event Mentions\*\*|Candidate entities):\s*(\[[^\n]*\])', prompt)
        if not matches:
            raise ValueError('Structured ACE output requires explicit candidate entities; never derive them from gold')
        mentions = json.loads(matches[-1])
        entities = sorted({m.get('entity', m.get('text', '')).strip() for m in mentions})
        if any(not e for e in entities):
            raise ValueError('Invalid candidate entity text')
        if not entities:
            return {'type': 'array', 'maxItems': 0, 'items': {'type': 'object'}}
        properties['entity'] = {'type': 'string', 'enum': entities}
    elif row['dataset'] == 'swig':
        properties['bbox'] = {'type': 'array', 'minItems': 4, 'maxItems': 4,
                              'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}}
    else:
        raise ValueError(f'Unsupported dataset: {row["dataset"]}')
    return {'type': 'array', 'items': {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}}


def messages_for(row: dict, *, api: bool) -> list[dict]:
    content = []
    for filename in row['images']:
        path = Path(filename)
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        if api:
            mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
            url = f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
            content.append({'type': 'image_url', 'image_url': {'url': url}})
        else:
            content.append({'type': 'image', 'url': str(path.resolve())})
    content.append({'type': 'text', 'text': row['prompt']})
    return [{'role': 'user', 'content': content}]


class QwenAPI:
    def __init__(self, model: str, base_url: str, *, max_tokens: int, timeout: float,
                 output_policy: str = 'legacy', repetition_penalty: float = 1.0, **kwargs):
        from openai import OpenAI
        if not base_url:
            raise ValueError('--base-url is required for API inference')
        if output_policy not in {'legacy', 'schema-v1'} or not math.isfinite(repetition_penalty) or repetition_penalty < 1:
            raise ValueError('Invalid output policy or repetition penalty (must be finite and >=1)')
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get('QWEN_API_KEY', 'EMPTY'),
                             timeout=timeout, max_retries=2)
        self.model, self.max_tokens = model, max_tokens
        self.output_policy, self.repetition_penalty = output_policy, repetition_penalty

    def generate(self, row: dict) -> dict:
        extra = {}
        if self.output_policy == 'schema-v1':
            extra['structured_outputs'] = {'json': output_schema(row)}
        if self.repetition_penalty != 1.0:
            extra['repetition_penalty'] = self.repetition_penalty
        response = self.client.chat.completions.create(
            model=self.model, messages=messages_for(row, api=True), temperature=0,
            max_tokens=self.max_tokens, seed=42, **({'extra_body': extra} if extra else {}))
        choice = response.choices[0]
        if choice.finish_reason not in {'stop', 'length'}:
            raise RuntimeError(f'Incomplete model response: finish_reason={choice.finish_reason}')
        output = choice.message.content
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError('Empty model response')
        return {'output': output, 'finish_reason': choice.finish_reason,
                'usage': response.usage.model_dump() if response.usage else None,
                'served_model': response.model}


class QwenTransformers:
    def __init__(self, model: str, *, max_tokens: int, max_input_tokens: int, device: str,
                 revision: str | None = None, **kwargs):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.manual_seed(42)
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, revision=revision, dtype='auto', device_map=device)
        self.model.eval()
        self.max_tokens, self.max_input_tokens = max_tokens, max_input_tokens

    def generate(self, row: dict) -> dict:
        import torch
        inputs = self.processor.apply_chat_template(messages_for(row, api=False), tokenize=True,
                  add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.model.device)
        length = inputs['input_ids'].shape[1]
        if length > self.max_input_tokens:
            raise ValueError(f'Input has {length} tokens, above limit {self.max_input_tokens}')
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        answer = generated[0, length:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(answer) == self.max_tokens and answer[-1].item() not in eos
        output = self.processor.decode(answer, skip_special_tokens=True)
        if not output.strip():
            raise RuntimeError('Empty model response')
        return {'output': output, 'finish_reason': 'length' if truncated else 'stop',
                'usage': {'prompt_tokens': length, 'completion_tokens': len(answer)},
                'served_model': self.model.config._name_or_path}
