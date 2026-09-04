"""Event-type prototype construction using ACE/SWiG EAE formats.

ACE and SWiG are not instance-paired. SWiG boxes therefore remain visual
prototypes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class PrototypeBuildStats:
    ace_events: int
    paired_events: int
    swig_frames: int
    eligible_swig_frames: int
    event_type_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ace_events": self.ace_events,
            "paired_events": self.paired_events,
            "unpaired_events": self.ace_events - self.paired_events,
            "swig_frames": self.swig_frames,
            "eligible_swig_frames": self.eligible_swig_frames,
            "pair_coverage": self.paired_events / self.ace_events if self.ace_events else 0.0,
            "event_type_counts": self.event_type_counts,
        }


def read_json_or_jsonl(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            records.append(value)
        return records


def write_jsonl(records: Iterable[Mapping[str, Any]], output: TextIO) -> int:
    count = 0
    for record in records:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        count += 1
    return count


def _stable_rank(sample_id: str, frame_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}:{frame_id}".encode()).hexdigest()


def _validate_ace_event(event: Mapping[str, Any]) -> None:
    required = ("sentence", "event_type", "trigger", "arguments")
    missing = [key for key in required if key not in event]
    if missing:
        raise ValueError(f"converted ACE event is missing {missing}: {event!r}")
    if not isinstance(event["arguments"], list):
        raise ValueError("converted ACE arguments must be a list")
    for argument in event["arguments"]:
        if not argument.get("role") or not (argument.get("entity") or argument.get("text")):
            raise ValueError(f"invalid ACE argument: {argument!r}")


def _normalize_swig_frame(frame: Mapping[str, Any], image_root: str | Path | None) -> dict[str, Any] | None:
    event_type = str(frame.get("event_type", "")).strip()
    image = str(frame.get("image", "")).strip()
    boxes = frame.get("bounding_boxes", {})
    if not event_type or not image or not isinstance(boxes, dict) or not boxes:
        return None
    image_path = str(Path(image_root) / image) if image_root else image
    return {
        "frame_id": image,
        "image": image_path,
        "verb": str(frame.get("verb", "")),
        "event_type": event_type,
        "bounding_boxes": boxes,
    }


def attach_event_type_prototypes(
    ace_events: Sequence[Mapping[str, Any]],
    swig_frames: Sequence[Mapping[str, Any]],
    *,
    split: str,
    image_root: str | Path | None = None,
    prototypes_per_event: int = 3,
    seed: int = 42,
) -> tuple[Iterator[dict[str, Any]], PrototypeBuildStats]:
    """Attach deterministic, type-matched SWiG prototypes to ACE events."""

    if prototypes_per_event < 1:
        raise ValueError("prototypes_per_event must be positive")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible = 0
    for frame in swig_frames:
        normalized = _normalize_swig_frame(frame, image_root)
        if normalized is not None:
            by_type[normalized["event_type"]].append(normalized)
            eligible += 1

    output = []
    paired = 0
    counts: Counter[str] = Counter()
    for index, source in enumerate(ace_events):
        _validate_ace_event(source)
        doc_id = source.get("doc_id", "doc")
        sent_id = source.get("sent_id", index)
        sample_id = f"{doc_id}:{sent_id}:{index}"
        event_type = str(source["event_type"])
        candidates = sorted(
            by_type.get(event_type, []),
            key=lambda item: _stable_rank(sample_id, item["frame_id"], seed),
        )
        selected = candidates[:prototypes_per_event]
        if selected:
            paired += 1
            counts[event_type] += 1
        output.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sentence": source["sentence"],
                "event_type": event_type,
                "trigger": source["trigger"],
                "arguments": list(source["arguments"]),
                "visual_prototypes": selected,
                "alignment": {
                    "kind": "event_type_prototype",
                    "instance_aligned": False,
                    "label_source": "ACE",
                    "visual_source": "SWiG",
                },
            }
        )
    stats = PrototypeBuildStats(
        ace_events=len(ace_events),
        paired_events=paired,
        swig_frames=len(swig_frames),
        eligible_swig_frames=eligible,
        event_type_counts=dict(sorted(counts.items())),
    )
    return iter(output), stats
