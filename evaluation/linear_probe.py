"""Frozen-encoder linear-probe evaluation used by the SAOT paper protocol."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn


def _encode_task(model: nn.Module, task: Dict[str, Tensor]) -> Tensor:
    with torch.no_grad():
        return model(task["x"], edge_index=task["edge_index"], sparse=True).detach()


def _fit_probe(
    features: Tensor,
    labels: Tensor,
    class_ids: List[int],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> nn.Module:
    if not class_ids:
        raise ValueError("linear probe needs at least one class")
    torch.manual_seed(seed)
    lookup = {int(label): index for index, label in enumerate(class_ids)}
    target = torch.tensor([lookup[int(label)] for label in labels.tolist()], dtype=torch.long, device=features.device)
    probe = nn.Linear(features.shape[1], len(class_ids), device=features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    for _ in range(max(1, int(epochs))):
        logits = probe(features)
        loss = nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return probe.eval()


def evaluate_frozen_encoder(
    model: nn.Module,
    tasks: List[Dict[str, Tensor]],
    checkpoint_task: int,
    *,
    scenario: str = "class-il",
    epochs: int = 100,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> Dict[int, float]:
    """Evaluate all tasks seen at a checkpoint with a frozen encoder.

    Class-IL fits one probe over all seen training nodes. Task-IL fits one
    independent probe per task, because the task identity is available at
    inference. Returned keys are task indices and values are test accuracy.
    """
    if checkpoint_task < 0 or checkpoint_task >= len(tasks):
        raise ValueError("checkpoint_task is outside the task sequence")
    model.eval()
    embeddings = [_encode_task(model, task) for task in tasks[: checkpoint_task + 1]]
    scenario = scenario.lower()
    scores: Dict[int, float] = {}
    if scenario == "task-il":
        for task_id, (task, features) in enumerate(zip(tasks[: checkpoint_task + 1], embeddings)):
            train_mask = task["train_mask"].bool()
            test_mask = task["test_mask"].bool()
            class_ids = [int(value) for value in task["classes"].tolist()]
            probe = _fit_probe(features[train_mask], task["y"][train_mask].long(), class_ids, epochs=epochs, learning_rate=learning_rate, seed=seed + task_id)
            lookup = {label: index for index, label in enumerate(class_ids)}
            target = torch.tensor([lookup[int(value)] for value in task["y"][test_mask].tolist()], device=features.device)
            scores[task_id] = float((probe(features[test_mask]).argmax(dim=1) == target).float().mean().item())
        return scores
    if scenario != "class-il":
        raise ValueError("scenario must be 'class-il' or 'task-il'")
    all_features, all_labels = [], []
    seen_classes: List[int] = []
    for task, features in zip(tasks[: checkpoint_task + 1], embeddings):
        train_mask = task["train_mask"].bool()
        all_features.append(features[train_mask])
        all_labels.append(task["y"][train_mask].long())
        for value in task["classes"].tolist():
            if int(value) not in seen_classes:
                seen_classes.append(int(value))
    probe = _fit_probe(torch.cat(all_features), torch.cat(all_labels), seen_classes, epochs=epochs, learning_rate=learning_rate, seed=seed + checkpoint_task)
    lookup = {label: index for index, label in enumerate(seen_classes)}
    for task_id, (task, features) in enumerate(zip(tasks[: checkpoint_task + 1], embeddings)):
        test_mask = task["test_mask"].bool()
        target = torch.tensor([lookup[int(value)] for value in task["y"][test_mask].tolist()], device=features.device)
        scores[task_id] = float((probe(features[test_mask]).argmax(dim=1) == target).float().mean().item())
    return scores
