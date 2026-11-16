"""ER-GNN style replay baseline for continual node classification.

Protocol
--------
The model is a two-layer GCN encoder plus one linear head (same backbone as
every other baseline).  For task ``t`` the loss is

``L = CE(current task train nodes) + replay_weight * CE(replay memory)``

Replay nodes come from member 3's :class:`data.replay_buffer.ReplayBuffer`, which
stores ``(task_id, global_node_id)`` records and never exceeds its budget.  After
training task ``t`` finishes, ``replay_budget`` training nodes of that task are
inserted, so the total capacity is ``replay_budget * num_tasks`` (the paper's
"200 samples per task" for CoraFull-CL).  ``replay_budget=0`` disables replay and
reduces exactly to ``gcn_sequential``.

Class-IL samples are scored over all classes seen so far; Task-IL samples are
grouped by their originating task and scored over that task's classes only
(multi-head semantics), because a replayed old sample cannot be expressed in the
current task's label space.

Input / output
--------------
``prepare(graph, split, seed=...)`` then ``fit_task(t)``; ``predict_logits()``
returns ``[num_nodes, num_classes]``.  All sampling seeds derive from the run
seed and the task index, so a run is reproducible.

Run
---
::

    python experiments/main_baselines.py --config configs/local.yaml \
        --dataset CoraFull-CL --seeds 42 43 44 --models replay

Memory
------
One full-graph forward per epoch plus ``replay_batch`` rows of cross entropy;
CoraFull-CL needs only a few GB.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from data.replay_buffer import ReplayBuffer
from evaluation.protocol import SequentialNodeClassifier, restricted_cross_entropy

__all__ = ["ReplayMethod"]


class ReplayMethod(SequentialNodeClassifier):
    """Sequential fine-tuning with a fixed-budget node replay memory.

    Args:
        replay_budget: Nodes stored per task (0 disables replay).
        replay_batch: Replay nodes drawn per epoch; defaults to ``replay_budget``.
        replay_weight: Weight of the replay cross entropy.
        **kwargs: Forwarded to :class:`SequentialNodeClassifier`.
    """

    name = "replay"

    def __init__(
        self,
        *,
        replay_budget: int = 200,
        replay_batch: int | None = None,
        replay_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(replay_budget, bool) or not isinstance(replay_budget, int):
            raise TypeError(f"replay_budget must be an integer, got {replay_budget!r}.")
        if replay_budget < 0:
            raise ValueError(f"replay_budget must be non-negative, got {replay_budget}.")
        if replay_batch is not None:
            if isinstance(replay_batch, bool) or not isinstance(replay_batch, int):
                raise TypeError(f"replay_batch must be an integer, got {replay_batch!r}.")
            if replay_batch < 0:
                raise ValueError(f"replay_batch must be non-negative, got {replay_batch}.")
        if float(replay_weight) < 0.0:
            raise ValueError(f"replay_weight must be non-negative, got {replay_weight}.")

        self.replay_budget = replay_budget
        self.replay_batch = replay_budget if replay_batch is None else replay_batch
        self.replay_weight = float(replay_weight)
        self.buffer: ReplayBuffer | None = None
        self._epoch = 0

    # ------------------------------------------------------------------ setup
    def prepare(self, graph, split, *, seed: int) -> None:
        super().prepare(graph, split, seed=seed)
        self.buffer = ReplayBuffer(budget=self.replay_budget * split.num_tasks, seed=seed)
        self._epoch = 0

    # --------------------------------------------------------------- training
    def _loss_terms(
        self, logits: Tensor, task_index: int, allowed: Sequence[int]
    ) -> tuple[Tensor, dict[str, float]]:
        current_loss, extras = super()._loss_terms(logits, task_index, allowed)
        if self.replay_batch == 0 or self.buffer is None or len(self.buffer) == 0:
            return current_loss, {**extras, "replay_loss": 0.0, "replay_nodes": 0}

        self._epoch += 1
        count = min(self.replay_batch, len(self.buffer))
        nodes, task_ids = self.buffer.sample(
            count,
            balanced=True,
            replace=False,
            seed=self._sample_seed(task_index),
            return_task_ids=True,
        )
        nodes = nodes.to(self.device)
        replay_logits = logits.index_select(0, nodes)
        replay_targets = self.graph.y.index_select(0, nodes)

        if self.scenario == "class-il":
            replay_loss = restricted_cross_entropy(replay_logits, replay_targets, allowed)
        else:
            replay_loss = self._task_il_replay_loss(replay_logits, replay_targets, task_ids)

        total = current_loss + self.replay_weight * replay_loss
        extras = {
            **extras,
            "replay_loss": float(replay_loss.detach()),
            "replay_nodes": int(nodes.numel()),
        }
        return total, extras

    def _task_il_replay_loss(
        self, logits: Tensor, targets: Tensor, task_ids: Sequence[Any]
    ) -> Tensor:
        flags = torch.as_tensor([int(task_id) for task_id in task_ids], dtype=torch.long)
        losses: list[Tensor] = []
        for task_id in sorted({int(value) for value in task_ids}):
            index = (flags == task_id).nonzero(as_tuple=True)[0].to(self.device)
            losses.append(
                restricted_cross_entropy(
                    logits.index_select(0, index),
                    targets.index_select(0, index),
                    self.split.task_to_classes[task_id],
                )
            )
        return torch.stack(losses).mean()

    def _sample_seed(self, task_index: int) -> int:
        return self.seed * 1000 + task_index * 100 + self._epoch

    def fit_task(self, task_index: int) -> dict[str, Any]:
        self._epoch = 0
        info = super().fit_task(task_index)
        if self.replay_budget > 0:
            self._store_task_samples(task_index)
        info["buffer_size"] = len(self.buffer) if self.buffer is not None else 0
        return info

    def _store_task_samples(self, task_index: int) -> None:
        pool = self.split.task_to_train_ids[task_index]
        count = min(self.replay_budget, pool.numel())
        if count == 0:
            return
        generator = torch.Generator().manual_seed(self.seed * 7777 + task_index)
        picks = torch.randperm(pool.numel(), generator=generator)[:count]
        self.buffer.add(task_id=task_index, node_ids=pool.index_select(0, picks))

    # -------------------------------------------------------------- artifacts
    def checkpoint_payload(self) -> dict[str, Any]:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "replay_budget": self.replay_budget,
                "replay_batch": self.replay_batch,
                "replay_weight": self.replay_weight,
                "buffer_size": len(self.buffer) if self.buffer is not None else 0,
            }
        )
        return payload


def build_method(**kwargs: Any) -> ReplayMethod:
    """Factory used by the unified runner."""

    return ReplayMethod(**kwargs)
