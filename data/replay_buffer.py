"""Fixed-budget historical subgraph replay used by SAOT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class ReplaySubgraph:
    task_id: int
    x: Tensor
    edge_index: Tensor
    y: Optional[Tensor] = None
    eval_mask: Optional[Tensor] = None


def induced_edge_index(edge_index: Tensor, nodes: Tensor) -> Tensor:
    if nodes.ndim != 1:
        raise ValueError("nodes must be a one-dimensional index tensor")
    if nodes.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    # The sampled nodes are global IDs. A dictionary-free sort/search mapping
    # avoids allocating a vector up to the largest global node ID and remains
    # correct when an edge references a node outside the sampled range.
    nodes = nodes.long().cpu()
    edges = edge_index.long().cpu()
    order = torch.argsort(nodes)
    sorted_nodes = nodes[order]
    positions = torch.searchsorted(sorted_nodes, edges)
    valid_position = positions < sorted_nodes.numel()
    safe_positions = positions.clamp_max(sorted_nodes.numel() - 1)
    valid_source = valid_position[0] & (sorted_nodes[safe_positions[0]] == edges[0])
    valid_target = valid_position[1] & (sorted_nodes[safe_positions[1]] == edges[1])
    keep = valid_source & valid_target
    local_source = order[safe_positions[0]]
    local_target = order[safe_positions[1]]
    return torch.stack((local_source[keep], local_target[keep]), dim=0)


class ReplayBuffer:
    def __init__(self, budget_per_task: int = 200, seed: int = 42):
        if budget_per_task < 1:
            raise ValueError("budget_per_task must be positive")
        self.budget_per_task = int(budget_per_task)
        self.generator = torch.Generator().manual_seed(seed)
        self._items: Dict[int, ReplaySubgraph] = {}

    def add(self, task_id: int, x: Tensor, edge_index: Tensor,
            y: Optional[Tensor] = None, eval_mask: Optional[Tensor] = None) -> ReplaySubgraph:
        n = x.shape[0]
        k = min(self.budget_per_task, n)
        perm = torch.randperm(n, generator=self.generator)[:k]
        perm, _ = torch.sort(perm)
        sampled_edges = induced_edge_index(edge_index.cpu(), perm.cpu())
        item = ReplaySubgraph(task_id, x[perm].detach().cpu(), sampled_edges,
                              None if y is None else y[perm].detach().cpu(),
                              None if eval_mask is None else eval_mask[perm].detach().cpu())
        self._items[task_id] = item
        return item

    def items(self) -> List[ReplaySubgraph]:
        return [self._items[k] for k in sorted(self._items)]

    def __len__(self) -> int:
        return len(self._items)

    def state_dict(self) -> Dict:
        return {"budget_per_task": self.budget_per_task, "items": self.items()}
