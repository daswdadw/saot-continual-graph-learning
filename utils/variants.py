"""Ablation variant definitions used by the experiment scripts."""

from __future__ import annotations

ABLATION_VARIANTS = [
    "w/ Cosine",
    "w/ MSE",
    "w/o Optimal Transport",
    "w/o Cross-task Distillation",
    "SAOT (ours)",
]


def variant_to_overrides(variant: str) -> dict:
    table = {
        "w/ Cosine": {"distillation_measure": "cosine"},
        "w/ MSE": {"distillation_measure": "mse"},
        "w/o Optimal Transport": {"use_ot": False, "ot_replacement": "graph_barlow_twins"},
        "w/o Cross-task Distillation": {"use_cross_task_distillation": False},
        "SAOT (ours)": {"use_ot": True, "use_cross_task_distillation": True},
    }
    if variant not in table:
        raise ValueError(f"unknown ablation variant: {variant}")
    return dict(table[variant])


def validate_single_component_change(variant: str) -> None:
    variant_to_overrides(variant)
