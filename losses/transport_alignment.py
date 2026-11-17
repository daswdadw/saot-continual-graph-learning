"""Optimal transport primitives used by SAOT (Equations 5--9)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def _uniform_marginals(n: int, m: int, reference: Tensor) -> Tuple[Tensor, Tensor]:
    return (torch.full((n,), 1.0 / n, device=reference.device, dtype=reference.dtype),
            torch.full((m,), 1.0 / m, device=reference.device, dtype=reference.dtype))


def _sinkhorn(cost: Tensor, epsilon: float = 0.05, iterations: int = 80,
              row_mass: Optional[Tensor] = None, col_mass: Optional[Tensor] = None) -> Tensor:
    """Entropic OT plan, computed in log space for numerical stability."""
    n, m = cost.shape
    if row_mass is None or col_mass is None:
        row_mass, col_mass = _uniform_marginals(n, m, cost)
    log_mu, log_nu = row_mass.clamp_min(1e-12).log(), col_mass.clamp_min(1e-12).log()
    log_k = -cost / max(float(epsilon), 1e-6)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(log_k + u[:, None], dim=0)
    return (log_k + u[:, None] + v[None, :]).exp()


def _gw_linearized_cost(a1: Tensor, a2: Tensor, pi: Tensor) -> Tensor:
    """Linearized GW cost ``L(A1,A2) ⊗ pi`` from Equation 4."""
    # C_ijkl = |A1_ik - A2_jl|; contraction over k,l with pi_kl.
    cost_tensor = (a1[:, :, None, None] - a2[None, None, :, :]).abs()
    return torch.einsum("ikjl,kl->ij", cost_tensor, pi)


def fused_gromov_wasserstein_plan(
    x1: Tensor, x2: Tensor, a1: Tensor, a2: Tensor, sigma: float = 0.5,
    epsilon: float = 0.05, iterations: int = 12, sinkhorn_iterations: int = 80,
) -> Tuple[Tensor, Tensor]:
    """Compute the paper's fused graph-space plan ``pi_G``.

    Returns ``(pi_G, C_G)`` where ``C_G`` is the fused node-pair cost. The
    graph term is optimized by the standard entropic conditional-gradient
    (linearized GW) update; all operations remain differentiable with respect
    to node features and therefore support SAOT training.
    """
    if x1.ndim != 2 or x2.ndim != 2 or a1.shape != (x1.shape[0], x1.shape[0]) or a2.shape != (x2.shape[0], x2.shape[0]):
        raise ValueError("features and adjacency shapes are inconsistent")
    sigma = float(sigma)
    feature_cost = torch.cdist(x1, x2).pow(2)
    row, col = _uniform_marginals(x1.shape[0], x2.shape[0], feature_cost)
    pi = row[:, None] * col[None, :]
    for _ in range(iterations):
        # Equation (3): c_A(A1_ik, A2_jl) = |A1_ik - A2_jl|.
        # The contraction is kept explicit so weighted/non-binary adjacency
        # matrices follow the paper's absolute-difference definition too.
        graph_cost = _gw_linearized_cost(a1, a2, pi)
        cost = sigma * feature_cost + (1.0 - sigma) * graph_cost
        pi = _sinkhorn(cost, epsilon, sinkhorn_iterations, row, col)
    return pi, cost


def representation_transport_plan(z1: Tensor, z2: Tensor, epsilon: float = 0.05,
                                  iterations: int = 80) -> Tensor:
    return _sinkhorn(torch.cdist(z1, z2).pow(2), epsilon, iterations)


def transport_plan_alignment_loss(pi_g: Tensor, pi_z: Tensor, measure: str = "frobenius") -> Tensor:
    """Equation 9: discrepancy between graph and representation plans."""
    if pi_g.shape != pi_z.shape:
        raise ValueError("transport plans must have the same shape")
    if measure == "kl":
        p, q = pi_g.clamp_min(1e-8), pi_z.clamp_min(1e-8)
        return (p * (p.log() - q.log()) - p + q).sum()
    if measure != "frobenius":
        raise ValueError("measure must be 'frobenius' or 'kl'")
    return (pi_g - pi_z).pow(2).mean()
