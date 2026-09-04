import unittest

import torch

from vptd_eae.temporal_attribution import build_temporal_target, compute_temporal_distillation_loss


class TemporalAttributionTests(unittest.TestCase):
    def test_consistent_video_hypotheses_correct_role_direction(self):
        student = torch.tensor([[0.0, 1.0, -0.2, 0.0]], requires_grad=True)
        static = torch.tensor([[0.0, 1.1, -0.1, 0.0]])
        videos = torch.tensor(
            [
                [[1.4, -0.2, -0.1, 0.0]],
                [[1.3, -0.1, -0.2, 0.0]],
                [[1.5, -0.3, -0.1, 0.0]],
            ]
        )
        output = build_temporal_target(student, static, videos)
        original = torch.softmax(student.detach(), dim=-1)
        self.assertTrue(output.consensus_gate.item())
        self.assertGreater(output.target_probs[0, 0], original[0, 0])
        self.assertLess(output.target_probs[0, 1], original[0, 1])
        self.assertGreater(output.refutation_weight.item(), 0)

    def test_conflicting_hypotheses_close_gate(self):
        student = torch.zeros(1, 3)
        static = torch.zeros(1, 3)
        videos = torch.tensor(
            [
                [[1.0, -1.0, 0.0]],
                [[-1.0, 0.0, 1.0]],
                [[0.0, 1.0, -1.0]],
            ]
        )
        output = build_temporal_target(student, static, videos)
        self.assertFalse(output.consensus_gate.item())
        self.assertTrue(torch.allclose(output.attributed_shift, torch.zeros_like(output.attributed_shift)))

    def test_loss_backpropagates_and_respects_role_mask(self):
        student = torch.tensor([[0.0, 1.0, -0.2, 9.0]], requires_grad=True)
        static = torch.tensor([[0.0, 1.1, -0.1, 9.0]])
        videos = torch.tensor(
            [
                [[1.4, -0.2, -0.1, 9.0]],
                [[1.3, -0.1, -0.2, 9.0]],
                [[1.5, -0.3, -0.1, 9.0]],
            ]
        )
        mask = torch.tensor([[True, True, True, False]])
        loss, metrics, output = compute_temporal_distillation_loss(
            student, static, videos, labels=torch.tensor([0]), role_mask=mask
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertEqual(output.target_probs[0, 3], 0)
        self.assertEqual(metrics["consensus_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
