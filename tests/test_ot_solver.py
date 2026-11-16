import unittest

import numpy as np

from models.ot_solver import (
    fused_gromov_wasserstein,
    pairwise_squared_euclidean,
    representation_transport_plan,
    sinkhorn,
)


class OTSolverTests(unittest.TestCase):
    def test_sinkhorn_preserves_marginals(self):
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = sinkhorn(cost, epsilon=0.05, tolerance=1e-10)
        np.testing.assert_allclose(result.plan.sum(axis=1), [0.5, 0.5], atol=1e-9)
        np.testing.assert_allclose(result.plan.sum(axis=0), [0.5, 0.5], atol=1e-9)
        self.assertGreater(result.plan[0, 0], result.plan[0, 1])

    def test_representation_plan_recovers_permutation(self):
        x = np.eye(4)
        permutation = np.array([2, 0, 3, 1])
        y = x[permutation]
        result = representation_transport_plan(x, y, epsilon=0.02)
        expected = np.argsort(permutation)
        np.testing.assert_array_equal(np.argmax(result.plan, axis=1), expected)

    def test_fgw_sigma_changes_plan(self):
        x1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        x2 = x1[[1, 2, 0]]
        feature_cost = pairwise_squared_euclidean(x1, x2)
        c1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        c2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        structure_only = fused_gromov_wasserstein(
            feature_cost, c1, c2, sigma=0.0, epsilon=0.03, relaxation=0.5
        )
        feature_only = fused_gromov_wasserstein(
            feature_cost, c1, c2, sigma=1.0, epsilon=0.03, relaxation=0.5
        )
        self.assertGreater(np.linalg.norm(structure_only.plan - feature_only.plan), 1e-3)
        np.testing.assert_allclose(feature_only.plan.sum(axis=1), 1 / 3, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
