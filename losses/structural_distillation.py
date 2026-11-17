"""Structural calibration and cross-task plan distillation losses."""

from __future__ import annotations

import torch
from torch import Tensor


def structural_calibration_loss(graph_cost: Tensor, representation_cost: Tensor,
                                pi_g: Tensor, measure: str = "frobenius") -> Tensor:
    """Equation 10: calibrate representation pairwise costs against FGW costs."""
    if graph_cost.shape != representation_cost.shape or graph_cost.shape != pi_g.shape:
        raise ValueError("cost matrices and plan must have the same shape")
    if measure == "kl":
        p, q = graph_cost.clamp_min(1e-8), representation_cost.clamp_min(1e-8)
        return (pi_g * (p * (p.log() - q.log()) - p + q)).sum()
    return (pi_g * (graph_cost - representation_cost).pow(2)).sum() / pi_g.sum().clamp_min(1e-8)


def cross_task_plan_distillation_loss(teacher_plan: Tensor, student_plan: Tensor,
                                      measure: str = "frobenius") -> Tensor:
    """Equation 12: distill teacher ``pi_Z`` into the current task encoder."""
    if teacher_plan.shape != student_plan.shape:
        raise ValueError("teacher and student plans must have the same shape")
    teacher_plan = teacher_plan.detach()
    if measure == "kl":
        p, q = teacher_plan.clamp_min(1e-8), student_plan.clamp_min(1e-8)
        return (p * (p.log() - q.log()) - p + q).sum()
    return (teacher_plan - student_plan).pow(2).mean()


def saot_total_loss(l_mat: Tensor, l_str: Tensor, l_skd: Tensor,
                    alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    return l_mat + float(alpha) * l_str + float(beta) * l_skd
