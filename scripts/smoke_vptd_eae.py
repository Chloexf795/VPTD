#!/usr/bin/env python3
"""Run a small numerical test of the migrated temporal objective."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vptd_eae.temporal_attribution import compute_temporal_distillation_loss


def main() -> int:
    # Role order: Attacker, Target, Instrument, NONE.
    student = torch.tensor([[0.1, 1.2, -0.2, 0.0]], requires_grad=True)
    static_teacher = torch.tensor([[0.0, 1.3, -0.1, 0.0]])
    video_teacher = torch.tensor(
        [
            [[1.5, -0.2, -0.1, 0.0]],
            [[1.3, -0.1, -0.2, 0.0]],
            [[1.6, -0.3, -0.1, 0.0]],
        ]
    )
    loss, metrics, output = compute_temporal_distillation_loss(
        student,
        static_teacher,
        video_teacher,
        labels=torch.tensor([0]),
    )
    loss.backward()
    report = {
        "loss": float(loss.detach()),
        "consensus_gate": bool(output.consensus_gate.item()),
        "student_probs": torch.softmax(student.detach(), dim=-1).tolist()[0],
        "temporal_target_probs": output.target_probs.tolist()[0],
        "gradient_norm": float(student.grad.norm()),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
