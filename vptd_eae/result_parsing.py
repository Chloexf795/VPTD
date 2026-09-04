from __future__ import annotations

import ast
import json
from typing import Any


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.split("</think>")[-1].strip()
    if text.lower().startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    first = text.find("[")
    last = text.rfind("]")
    candidate = text[first : last + 1] if first >= 0 and last > first else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return []


def extract_arguments(output: Any) -> list[dict[str, Any]]:
    parsed = _parse_json_value(output)
    if isinstance(parsed, dict):
        parsed = parsed.get("arguments", parsed.get("pred", [parsed]))
    if not isinstance(parsed, list):
        return []
    arguments = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        entity = str(item.get("entity", item.get("text", ""))).strip()
        if role and entity:
            arguments.append({"role": role, "entity": entity})
    return arguments
