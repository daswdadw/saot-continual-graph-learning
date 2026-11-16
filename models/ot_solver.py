"""Optimal-transport solvers used by the SAOT reproduction.

The module is deliberately NumPy-only so the member-2 deliverable can be run
without the full SAOT training stack.  It provides:

* a log-domain Sinkhorn solver for representation-space Wasserstein OT; and
* an entropic fixed-point solver for graph-space fused
  Gromov-Wasserstein (FGW) transport.

For two graphs, ``sigma`` is the feature weight used in the SAOT paper:

    sigma * <M, pi> + (1-sigma) * sum |C1_ik-C2_jl| pi_ij pi_kl.

The absolute structural loss matches Equation (5) in the paper.  A squared
loss is also provided because it admits a lower-memory contraction and is
useful for larger sampled graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class OTResult:
    """Result returned by :func:`sinkhorn`."""

    plan: Array
    cost: float
    converged: bool
    iterations: int
    marginal_error: float
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class FGWResult:
    """Result returned by :func:`fused_gromov_wasserstein`."""

    plan: Array
    objective: float
    feature_term: float
    structure_term: float
    converged: bool
    iterations: int
    plan_change: float
    marginal_error: float
    objective_history: tuple[float, ...]


def _as_float_matrix(value: Array, name: str) -> Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def _probability_vector(value: Array | None, size: int, name: str) -> Array:
    if value is None:
        return np.full(size, 1.0 / size, dtype=np.float64)
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if np.any(vector <= 0) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite, strictly positive masses")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return vector / total


def pairwise_squared_euclidean(x: Array, y: Array) -> Array:
    """Return the pairwise squared Euclidean cost matrix."""

    x = _as_float_matrix(x, "x")
    y = _as_float_matrix(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    cost = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * x @ y.T
    )
    return np.maximum(cost, 0.0)


def pairwise_cosine_cost(x: Array, y: Array, eps: float = 1e-12) -> Array:
    """Return ``1 - cosine_similarity`` for every cross-set pair."""

    x = _as_float_matrix(x, "x")
    y = _as_float_matrix(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    x_norm = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    y_norm = np.maximum(np.linalg.norm(y, axis=1, keepdims=True), eps)
    cosine = (x / x_norm) @ (y / y_norm).T
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def _logsumexp(matrix: Array, axis: int) -> Array:
    maximum = np.max(matrix, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(matrix - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def sinkhorn(
    cost_matrix: Array,
    source_mass: Array | None = None,
    target_mass: Array | None = None,
    *,
    epsilon: float = 0.05,
    max_iter: int = 2_000,
    tolerance: float = 1e-9,
) -> OTResult:
    """Solve entropy-regularized discrete OT in the log domain.

    The returned objective history records the unregularized transport cost.
    The transport plan always uses the requested marginals up to the reported
    numerical error.
    """

    cost = _as_float_matrix(cost_matrix, "cost_matrix")
    if np.any(cost < 0):
        raise ValueError("cost_matrix must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    n, m = cost.shape
    a = _probability_vector(source_mass, n, "source_mass")
    b = _probability_vector(target_mass, m, "target_mass")
    log_a = np.log(a)
    log_b = np.log(b)
    log_kernel = -cost / epsilon
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)
    history: list[float] = []
    converged = False
    marginal_error = float("inf")

    for iteration in range(1, max_iter + 1):
        old_log_u = log_u.copy()
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel + log_u[:, None], axis=0)

        if iteration == 1 or iteration % 10 == 0 or iteration == max_iter:
            log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            plan = np.exp(np.clip(log_plan, -745.0, 709.0))
            marginal_error = max(
                float(np.max(np.abs(plan.sum(axis=1) - a))),
                float(np.max(np.abs(plan.sum(axis=0) - b))),
            )
            history.append(float(np.sum(plan * cost)))
            dual_change = float(np.max(np.abs(log_u - old_log_u)))
            if marginal_error <= tolerance and dual_change <= max(tolerance, 1e-12):
                converged = True
                break

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(np.clip(log_plan, -745.0, 709.0))
    marginal_error = max(
        float(np.max(np.abs(plan.sum(axis=1) - a))),
        float(np.max(np.abs(plan.sum(axis=0) - b))),
    )
    return OTResult(
        plan=plan,
        cost=float(np.sum(plan * cost)),
        converged=converged or marginal_error <= tolerance,
        iterations=iteration,
        marginal_error=marginal_error,
        objective_history=tuple(history),
    )


def _absolute_structure_cost(c1: Array, c2: Array, plan: Array) -> tuple[Array, float]:
    # tensor[i, j, k, l] = |c1[i,k] - c2[j,l]|.
    tensor = np.abs(c1[:, None, :, None] - c2[None, :, None, :])
    local = np.einsum("ijkl,kl->ij", tensor, plan, optimize=True)
    objective = float(np.sum(local * plan))
    return local, objective


def _squared_structure_cost(
    c1: Array, c2: Array, plan: Array, source_mass: Array, target_mass: Array
) -> tuple[Array, float]:
    # Efficient contraction for (c1_ik - c2_jl)^2.
    constant_1 = (c1 * c1) @ source_mass
    constant_2 = (c2 * c2) @ target_mass
    local = constant_1[:, None] + constant_2[None, :] - 2.0 * c1 @ plan @ c2.T
    local = np.maximum(local, 0.0)
    objective = float(np.sum(local * plan))
    return local, objective


def fused_gromov_wasserstein(
    feature_cost: Array,
    structure_1: Array,
    structure_2: Array,
    source_mass: Array | None = None,
    target_mass: Array | None = None,
    *,
    sigma: float = 0.5,
    epsilon: float = 0.03,
    max_iter: int = 200,
    sinkhorn_max_iter: int = 2_000,
    tolerance: float = 1e-8,
    structure_loss: Literal["absolute", "square"] = "absolute",
    relaxation: float = 1.0,
) -> FGWResult:
    """Compute an entropic fused Gromov-Wasserstein transport plan.

    This solves a regularized fixed-point approximation.  Each outer step
    contracts the four-index structural loss under the current plan, then
    obtains a new feasible coupling through Sinkhorn.  ``relaxation`` can be
    reduced below 1 for difficult cases.
    """

    feature_cost = _as_float_matrix(feature_cost, "feature_cost")
    c1 = _as_float_matrix(structure_1, "structure_1")
    c2 = _as_float_matrix(structure_2, "structure_2")
    n, m = feature_cost.shape
    if c1.shape != (n, n) or c2.shape != (m, m):
        raise ValueError("structure matrices must match the feature-cost dimensions")
    if not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must be in (0, 1]")
    if structure_loss not in {"absolute", "square"}:
        raise ValueError("structure_loss must be 'absolute' or 'square'")

    a = _probability_vector(source_mass, n, "source_mass")
    b = _probability_vector(target_mass, m, "target_mass")
    plan = np.outer(a, b)
    history: list[float] = []
    converged = False
    plan_change = float("inf")

    for iteration in range(1, max_iter + 1):
        if structure_loss == "absolute":
            local_structure, _ = _absolute_structure_cost(c1, c2, plan)
        else:
            local_structure, _ = _squared_structure_cost(c1, c2, plan, a, b)
        local_cost = sigma * feature_cost + (1.0 - sigma) * 2.0 * local_structure
        update = sinkhorn(
            local_cost,
            a,
            b,
            epsilon=epsilon,
            max_iter=sinkhorn_max_iter,
            tolerance=min(tolerance * 0.1, 1e-10),
        ).plan
        candidate = (1.0 - relaxation) * plan + relaxation * update
        plan_change = float(np.linalg.norm(candidate - plan, ord=1))
        plan = candidate

        if structure_loss == "absolute":
            _, structure_term = _absolute_structure_cost(c1, c2, plan)
        else:
            _, structure_term = _squared_structure_cost(c1, c2, plan, a, b)
        feature_term = float(np.sum(feature_cost * plan))
        objective = sigma * feature_term + (1.0 - sigma) * structure_term
        history.append(objective)
        if plan_change <= tolerance:
            converged = True
            break

    marginal_error = max(
        float(np.max(np.abs(plan.sum(axis=1) - a))),
        float(np.max(np.abs(plan.sum(axis=0) - b))),
    )
    return FGWResult(
        plan=plan,
        objective=objective,
        feature_term=feature_term,
        structure_term=structure_term,
        converged=converged,
        iterations=iteration,
        plan_change=plan_change,
        marginal_error=marginal_error,
        objective_history=tuple(history),
    )


def representation_transport_plan(
    embeddings_1: Array,
    embeddings_2: Array,
    source_mass: Array | None = None,
    target_mass: Array | None = None,
    *,
    metric: Literal["cosine", "sqeuclidean"] = "cosine",
    epsilon: float = 0.03,
    max_iter: int = 2_000,
    tolerance: float = 1e-9,
) -> OTResult:
    """Compute SAOT's representation-space plan ``pi_Z``."""

    if metric == "cosine":
        cost = pairwise_cosine_cost(embeddings_1, embeddings_2)
    elif metric == "sqeuclidean":
        cost = pairwise_squared_euclidean(embeddings_1, embeddings_2)
    else:
        raise ValueError("metric must be 'cosine' or 'sqeuclidean'")
    return sinkhorn(
        cost,
        source_mass,
        target_mass,
        epsilon=epsilon,
        max_iter=max_iter,
        tolerance=tolerance,
    )
