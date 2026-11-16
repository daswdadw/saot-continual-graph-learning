"""Graph Auto-Encoder (GAE) baseline for continual node classification.

Why GAE and not DGI
-------------------
Both are self-supervised graph baselines of the SAOT paper, and the assignment
only requires one.  GAE (Kipf & Welling, 2016) is much cheaper to implement and
far more stable in a continual setting: it needs a GCN encoder, an inner-product
decoder and a negative-edge BCE, with no readout network, no bilinear
discriminator and no EMA target encoder.  DGI's discriminator compares node
embeddings against a global summary, and that summary shifts as tasks arrive, so
it also introduces extra hyper-parameters (corruption rate, readout choice).
``METHOD_REGISTRY`` in ``experiments/main_baselines.py`` can gain a ``dgi``
entry later without touching this file's interface.

Protocol
--------
For each task:

1. **self-supervised pre-training** -- reconstruct the current task's edges with
   ``sigmoid(z_i^T z_j)``, using the task's positive edges plus the same number
   of deterministically sampled negative pairs, for ``pretrain_epochs`` epochs;
2. **linear probe** -- freeze the encoder and train the shared linear head on
   the current task's labels for ``probe_epochs`` epochs (``joint_finetune=True``
   instead fine-tunes encoder and head together).

Input / output
--------------
Same as every baseline: ``prepare(graph, split, seed)`` then ``fit_task(t)``;
``predict_logits()`` returns ``[num_nodes, num_classes]`` for the evaluator.

Run
---
::

    python experiments/main_baselines.py --config configs/local.yaml \
        --dataset CoraFull-CL --seeds 42 43 44 --models gae

Memory
------
Only one full-graph forward per epoch, so CoraFull-CL fits in a few GB; the
negative-edge BCE never materialises an ``N x N`` adjacency.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from evaluation.protocol import (
    SequentialNodeClassifier,
    restricted_cross_entropy,
)

__all__ = ["GAEMethod"]


class GAEMethod(SequentialNodeClassifier):
    """GAE pre-training followed by a linear probe, repeated per task.

    Args:
        pretrain_epochs: Self-supervised epochs per task (defaults to
            ``epochs_per_task``).
        probe_epochs: Supervised epochs for the classifier (defaults to
            ``epochs_per_task``).
        negative_ratio: Negative pairs sampled per positive edge.
        joint_finetune: If true, the probe phase also updates the encoder.
        **kwargs: Forwarded to :class:`SequentialNodeClassifier`.
    """

    name = "gae"

    def __init__(
        self,
        *,
        pretrain_epochs: int | None = None,
        probe_epochs: int | None = None,
        negative_ratio: float = 1.0,
        joint_finetune: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if pretrain_epochs is not None and (
            isinstance(pretrain_epochs, bool)
            or not isinstance(pretrain_epochs, int)
            or pretrain_epochs < 1
        ):
            raise ValueError(f"pretrain_epochs must be a positive integer, got {pretrain_epochs!r}.")
        if probe_epochs is not None and (
            isinstance(probe_epochs, bool) or not isinstance(probe_epochs, int) or probe_epochs < 1
        ):
            raise ValueError(f"probe_epochs must be a positive integer, got {probe_epochs!r}.")
        if float(negative_ratio) <= 0.0:
            raise ValueError(f"negative_ratio must be positive, got {negative_ratio}.")
        self.pretrain_epochs = int(pretrain_epochs or self.epochs_per_task)
        self.probe_epochs = int(probe_epochs or self.epochs_per_task)
        self.negative_ratio = float(negative_ratio)
        self.joint_finetune = bool(joint_finetune)

    def fit_task(self, task_index: int) -> dict[str, Any]:
        self._validate_task_index(task_index)
        edges = self.split.task_edge_index(self.graph.edge_index, task_index)
        if edges.size(1) == 0:
            raise RuntimeError(
                f"task {task_index} has no intra-task edge; GAE cannot reconstruct it."
            )
        task_nodes = self.split.task_to_node_ids[task_index].to(self.device)
        generator = torch.Generator().manual_seed(self.seed * 10_000 + task_index * 17 + 1)
        num_negative = max(1, int(round(edges.size(1) * self.negative_ratio)))

        ssl_losses = self._pretrain(edges, task_nodes, num_negative, generator)
        allowed = self.allowed_classes(task_index)
        probe_losses = self._probe(task_index, allowed)

        train_accuracy = self._train_accuracy(task_index, allowed)
        info: dict[str, Any] = {
            "task": task_index,
            "epochs": self.pretrain_epochs + self.probe_epochs,
            "pretrain_epochs": self.pretrain_epochs,
            "probe_epochs": self.probe_epochs,
            "final_loss": probe_losses[-1] if probe_losses else float("nan"),
            "mean_loss": float(sum(probe_losses) / len(probe_losses)) if probe_losses else float("nan"),
            "ssl_final_loss": ssl_losses[-1] if ssl_losses else float("nan"),
            "ssl_mean_loss": float(sum(ssl_losses) / len(ssl_losses)) if ssl_losses else float("nan"),
            "train_accuracy": train_accuracy,
            "allowed_classes": list(allowed),
        }
        self.tasks_trained = max(self.tasks_trained, task_index + 1)
        self.history.append(info)
        return info

    def _pretrain(
        self,
        edges: Tensor,
        task_nodes: Tensor,
        num_negative: int,
        generator: torch.Generator,
    ) -> list[float]:
        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        positives = torch.ones(edges.size(1), device=self.device)
        negatives_target = torch.zeros(num_negative, device=self.device)
        losses: list[float] = []

        self.train()
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad(set_to_none=True)
            embeddings = self.encoder(
                self.graph.x,
                self.graph.edge_index,
                edge_weight=getattr(self.graph, "edge_weight", None),
            )
            source, target = self._negative_pairs(task_nodes, num_negative, generator)
            positive_scores = (
                embeddings.index_select(0, edges[0]) * embeddings.index_select(0, edges[1])
            ).sum(dim=1)
            negative_scores = (
                embeddings.index_select(0, source) * embeddings.index_select(0, target)
            ).sum(dim=1)
            loss = F.binary_cross_entropy_with_logits(
                positive_scores, positives
            ) + F.binary_cross_entropy_with_logits(negative_scores, negatives_target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    def _negative_pairs(
        self, task_nodes: Tensor, count: int, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        picks = torch.randint(0, task_nodes.numel(), (2 * count,), generator=generator)
        pairs = task_nodes.index_select(0, picks.to(task_nodes.device))
        return pairs[:count], pairs[count:]

    def _probe(self, task_index: int, allowed: list[int]) -> list[float]:
        train_ids = self._train_ids(task_index)
        targets = self.graph.y.index_select(0, train_ids)
        losses: list[float] = []

        if self.joint_finetune:
            optimizer = torch.optim.Adam(
                self._trainable_parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self.train()
            for _ in range(self.probe_epochs):
                optimizer.zero_grad(set_to_none=True)
                logits = self._forward_logits().index_select(0, train_ids)
                loss = restricted_cross_entropy(logits, targets, allowed)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return losses

        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(
                self.graph.x,
                self.graph.edge_index,
                edge_weight=getattr(self.graph, "edge_weight", None),
            ).detach()
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        for _ in range(self.probe_epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = self.classifier(embeddings).index_select(0, train_ids)
            loss = restricted_cross_entropy(logits, targets, allowed)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses


def build_method(**kwargs: Any) -> GAEMethod:
    """Factory used by the unified runner."""

    return GAEMethod(**kwargs)
