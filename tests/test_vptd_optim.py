import copy
import unittest

import torch

from vptd_eae.optim import LowMemoryAdamW


class OptimizerTests(unittest.TestCase):
    def test_resume_preserves_adam_state_and_disables_foreach(self):
        initial = torch.nn.Parameter(torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64))
        old = torch.optim.AdamW([initial], lr=2e-6, foreach=True)
        initial.grad = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
        old.step()
        saved = copy.deepcopy(old.state_dict())
        actual = torch.nn.Parameter(initial.detach().clone())
        expected = torch.nn.Parameter(initial.detach().clone())
        optimizer = LowMemoryAdamW([actual], lr=0.1)
        reference = torch.optim.AdamW([expected], lr=0.1, foreach=False)
        optimizer.load_state_dict(saved)
        reference.load_state_dict(copy.deepcopy(saved))
        reference.param_groups[0]['foreach'] = False
        self.assertIs(saved['param_groups'][0]['foreach'], True)
        self.assertIs(optimizer.param_groups[0]['foreach'], False)
        self.assertEqual(optimizer.param_groups[0]['lr'], 2e-6)
        for parameter in (actual, expected):
            parameter.grad = torch.tensor([-0.2, 0.5, 0.3], dtype=torch.float64)
        optimizer.step()
        reference.step()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[actual][key], reference.state[expected][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
