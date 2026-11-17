"""SAOT encoder and model-facing utilities.

The implementation follows Section 4 of the paper and deliberately keeps the
encoder independent of PyG: dense adjacency matrices and ``edge_index`` pairs
are both accepted, which makes the loss code reusable for sampled subgraphs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def edge_index_to_adjacency(edge_index: Tensor, num_nodes: int, device=None) -> Tensor:
    """Convert a ``[2, E]`` edge index to a symmetric dense adjacency matrix."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    device = device or edge_index.device
    a = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
    a[edge_index[0].long(), edge_index[1].long()] = 1.0
    a[edge_index[1].long(), edge_index[0].long()] = 1.0
    a.fill_diagonal_(0)
    return a


def normalize_adjacency(adjacency: Tensor) -> Tensor:
    """Return GCN normalization ``D^-1/2 (A+I) D^-1/2``."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    a = adjacency.float()
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    degree = (a + eye).sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * (a + eye) * inv_sqrt[None, :]


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return self.linear(normalize_adjacency(adjacency) @ x)


class SparseGCNLayer(nn.Module):
    """GCN layer for sampled graphs represented by ``edge_index``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        n = x.shape[0]
        src, dst = edge_index.long()
        self_loops = torch.arange(n, device=x.device)
        src = torch.cat([src, self_loops])
        dst = torch.cat([dst, self_loops])
        degree = torch.bincount(src, minlength=n).to(x.dtype).clamp_min(1.0)
        norm = degree[src].rsqrt() * degree[dst].rsqrt()
        messages = x[src] * norm[:, None]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)
        return self.linear(out)


class SAOTModel(nn.Module):
    """Two-layer GCN backbone used by SAOT in the paper."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "layer1": GCNLayer(in_dim, hidden_dim),
            "layer2": GCNLayer(hidden_dim, out_dim),
        })
        self.sparse_encoder = nn.ModuleDict({
            "layer1": SparseGCNLayer(in_dim, hidden_dim),
            "layer2": SparseGCNLayer(hidden_dim, out_dim),
        })
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: Tensor, adjacency: Tensor) -> Tensor:
        h = torch.relu(self.encoder["layer1"](x, adjacency))
        h = self.dropout(h)
        return self.encoder["layer2"](h, adjacency)

    def encode_sparse(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = torch.relu(self.sparse_encoder["layer1"](x, edge_index))
        h = self.dropout(h)
        return self.sparse_encoder["layer2"](h, edge_index)

    def forward(self, x: Tensor, adjacency: Optional[Tensor] = None,
                edge_index: Optional[Tensor] = None, sparse: bool = False) -> Tensor:
        if sparse:
            if edge_index is None:
                raise ValueError("edge_index is required when sparse=True")
            return self.encode_sparse(x, edge_index)
        if adjacency is None:
            if edge_index is None:
                raise ValueError("provide adjacency or edge_index")
            adjacency = edge_index_to_adjacency(edge_index, x.shape[0], x.device)
        return self.encode(x, adjacency)

    def snapshot_teacher(self) -> Dict[str, Tensor]:
        """Return a detached, immutable-by-convention teacher state."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_teacher(self, state: Dict[str, Tensor]) -> None:
        self.load_state_dict(deepcopy(state), strict=True)

    def teacher(self, state: Dict[str, Tensor]) -> "SAOTModel":
        teacher = deepcopy(self)
        teacher.load_teacher(state)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher


def pairwise_representation_cost(z1: Tensor, z2: Tensor) -> Tensor:
    return torch.cdist(z1, z2).pow(2)
