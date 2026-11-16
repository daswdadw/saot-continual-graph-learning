"""Vanilla sequential GCN baseline (fine-tuning, no forgetting control).

Definition
----------
A two-layer GCN encoder plus one linear head is trained task by task on the
current task's training nodes only.  Nothing protects earlier knowledge, so this
is the "lower bound" baseline every CGL method must beat.

Input / output
--------------
Input: a global ``torch_geometric.data.Data`` (``x``, ``edge_index``, ``y``) and
a :class:`evaluation.protocol.TaskSplit`.  Output: one performance matrix per
scenario, built by the runner from :meth:`predict_logits`.

Run
---
::

    python experiments/main_baselines.py --config configs/local.yaml \
        --dataset CoraFull-CL --seeds 42 43 44 --models gcn_sequential

Memory
------
Full-graph training on CoraFull-CL (19,793 nodes, hidden 512) stays well below
a few GB of GPU memory; no neighbour sampling is needed.

Scenario rules
--------------
``class-il`` restricts logits to every class seen so far, ``task-il`` restricts
them to the current task's classes.  Both use the same trained model, so one run
yields both matrices.
"""

from __future__ import annotations

from typing import Any

from evaluation.protocol import SequentialNodeClassifier

__all__ = ["GCNSequentialMethod"]


class GCNSequentialMethod(SequentialNodeClassifier):
    """Sequential fine-tuning of a shared GCN encoder and linear head.

    The whole behaviour (current-task cross entropy, per-task Adam reset and
    scenario-dependent label masking) comes from
    :class:`evaluation.protocol.SequentialNodeClassifier`; this class only
    documents and names the baseline.
    """

    name = "gcn_sequential"


def build_method(**kwargs: Any) -> GCNSequentialMethod:
    """Factory used by the unified runner."""

    return GCNSequentialMethod(**kwargs)
