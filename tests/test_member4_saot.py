import unittest

import torch

from losses.transport_alignment import (
    fused_gromov_wasserstein_plan,
    transport_plan_alignment_loss,
)
from losses.structural_distillation import (
    structural_calibration_loss,
    cross_task_plan_distillation_loss,
)
from models.saot import SAOTModel


class Member4SAOTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x1 = torch.randn(5, 4)
        self.x2 = torch.randn(4, 4)
        self.a1 = torch.tensor(
            [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
            dtype=torch.float32,
        )
        self.a2 = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            dtype=torch.float32,
        )

    def test_fgw_plan_has_marginals_and_alignment_is_zero_for_same_plan(self):
        pi, graph_cost = fused_gromov_wasserstein_plan(
            self.x1, self.x2, self.a1, self.a2, sigma=0.5, epsilon=0.2
        )
        self.assertTrue(torch.allclose(pi.sum(1), torch.full((5,), 1 / 5), atol=2e-3))
        self.assertTrue(torch.allclose(pi.sum(0), torch.full((4,), 1 / 4), atol=2e-3))
        self.assertAlmostEqual(float(transport_plan_alignment_loss(pi, pi)), 0.0, places=6)
        self.assertEqual(tuple(graph_cost.shape), (5, 4))

    def test_structural_and_distillation_losses_are_nonnegative(self):
        pi, graph_cost = fused_gromov_wasserstein_plan(
            self.x1, self.x2, self.a1, self.a2, sigma=0.5, epsilon=0.2
        )
        z1, z2 = self.x1[:, :2], self.x2[:, :2]
        representation_cost = torch.cdist(z1, z2).pow(2)
        self.assertGreaterEqual(float(structural_calibration_loss(graph_cost, representation_cost, pi)), 0.0)
        self.assertGreaterEqual(float(cross_task_plan_distillation_loss(pi, pi.detach())), 0.0)

    def test_teacher_snapshot_is_deep_copy(self):
        model = SAOTModel(in_dim=4, hidden_dim=8, out_dim=6)
        teacher = model.snapshot_teacher()
        before = {k: v.clone() for k, v in teacher.items()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        self.assertTrue(all(torch.equal(before[k], teacher[k]) for k in teacher))


if __name__ == "__main__":
    unittest.main()
