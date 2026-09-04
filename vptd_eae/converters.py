"""converters for the ACE and SWiG processed schemas."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .event_schema import normalize_event_type


def convert_ace_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten processed ACE sentences into one record per event mention."""

    converted = []
    for sentence_record in records:
        sentence = str(sentence_record.get("sentence", ""))
        for event in sentence_record.get("event_mentions", []):
            arguments = []
            for argument in event.get("arguments", []):
                role = str(argument.get("role", "")).strip()
                entity = str(argument.get("text", argument.get("entity", ""))).strip()
                if role and entity:
                    arguments.append({"role": role, "entity": entity})
            converted.append(
                {
                    "doc_id": sentence_record.get("doc_id", "doc"),
                    "sent_id": sentence_record.get("sent_id", len(converted)),
                    "sentence": sentence,
                    "event_type": normalize_event_type(str(event.get("event_type", ""))),
                    "trigger": str(event.get("trigger", {}).get("text", event.get("trigger", ""))),
                    "arguments": arguments,
                }
            )
    return converted


def load_swig_mapping(path: str | Path) -> dict[tuple[str, str], tuple[str, str]]:
    mapping = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"expected four tab-separated fields at {path}:{line_number}")
            verb, swig_role, ace_event, ace_role = parts[:4]
            mapping[(verb, swig_role)] = (normalize_event_type(ace_event), ace_role)
    return mapping


def convert_swig_records(
    records: Mapping[str, Mapping[str, Any]],
    mapping: Mapping[tuple[str, str], tuple[str, str]],
) -> list[dict[str, Any]]:
    """Map SWiG roles to ACE roles and normalize boxes to the 0-1000 scale."""

    converted = []
    for image_name, frame in records.items():
        width = float(frame.get("width", 0))
        height = float(frame.get("height", 0))
        verb = str(frame.get("verb", ""))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size for {image_name!r}")
        grouped: dict[str, list[list[int]]] = defaultdict(list)
        event_type = None
        for swig_role, box in frame.get("bb", {}).items():
            mapped = mapping.get((verb, swig_role))
            if mapped is None or not isinstance(box, list) or len(box) != 4 or -1 in box:
                continue
            event_type, ace_role = mapped
            grouped[ace_role].append(
                [
                    round(float(box[0]) / width * 1000),
                    round(float(box[1]) / height * 1000),
                    round(float(box[2]) / width * 1000),
                    round(float(box[3]) / height * 1000),
                ]
            )
        if event_type and grouped:
            converted.append(
                {
                    "image": image_name,
                    "verb": verb,
                    "event_type": event_type,
                    "bounding_boxes": dict(grouped),
                }
            )
    return converted
