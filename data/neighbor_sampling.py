"""Deterministic layer-wise neighbor sampling for replay subgraphs.

The sampler is intentionally small and dataset-agnostic.  It returns ordinary
PyG ``Data`` objects with local ``edge_index`` values and a global ``n_id``
mapping, which makes the result directly usable by GCN/GAT encoders.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

SamplingDirection = Literal["in", "out", "both"]


def _normalise_num_neighbors(num_neighbors: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(num_neighbors, bool):
        raise TypeError("num_neighbors must contain integers, not booleans.")
    if isinstance(num_neighbors, int):
        values = (num_neighbors,)
    else:
        try:
            values = tuple(num_neighbors)
        except TypeError as exc:
            raise TypeError("num_neighbors must be an integer or a sequence of integers.") from exc
    if not values:
        raise ValueError("num_neighbors must specify at least one GNN layer.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "each num_neighbors value must be an integer; use -1 for all neighbors."
            )
        if value < -1:
            raise ValueError(f"num_neighbors values must be >= -1, got {value}.")
    return values


def _normalise_seed_nodes(
    seed_nodes: Tensor | Iterable[int], *, num_nodes: int, device: torch.device
) -> list[int]:
    if isinstance(seed_nodes, Tensor):
        if seed_nodes.ndim != 1:
            raise ValueError(
                "seed_nodes tensor must be one-dimensional, "
                f"got shape {tuple(seed_nodes.shape)}."
            )
        if seed_nodes.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"seed_nodes must have an integer dtype, got {seed_nodes.dtype}.")
        values = seed_nodes.detach().cpu().tolist()
    else:
        try:
            values = list(seed_nodes)
        except TypeError as exc:
            raise TypeError("seed_nodes must be a 1-D tensor or an iterable of integers.") from exc

    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise TypeError("seed node IDs must be integers, not booleans.")
        try:
            node_id = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"seed node ID {value!r} is not an integer.") from exc
        if node_id != value:
            raise ValueError(f"seed node ID {value!r} is not an exact integer.")
        if node_id < 0 or node_id >= num_nodes:
            raise ValueError(
                f"seed node ID {node_id} is outside valid range [0, {num_nodes - 1}]."
            )
        if node_id not in seen:
            result.append(node_id)
            seen.add(node_id)
    return result


def _validate_graph(data: Data) -> tuple[int, Tensor, torch.device]:
    if not isinstance(data, Data):
        raise TypeError("data must be a torch_geometric.data.Data or Batch instance.")
    if data.edge_index is None:
        raise ValueError("data.edge_index is required for neighbor sampling.")
    edge_index = data.edge_index
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "data.edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("data.edge_index must contain integer node indices.")
    num_nodes = data.num_nodes
    if num_nodes is None:
        raise ValueError("Unable to infer data.num_nodes.")
    if num_nodes < 0:
        raise ValueError(f"data.num_nodes must be non-negative, got {num_nodes}.")
    if edge_index.numel():
        minimum = int(edge_index.min())
        maximum = int(edge_index.max())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError(
                "data.edge_index contains an out-of-range node index: "
                f"valid range is [0, {num_nodes - 1}], found [{minimum}, {maximum}]."
            )
    return num_nodes, edge_index.detach().cpu(), edge_index.device


def _build_adjacency(
    edge_index_cpu: Tensor, *, num_nodes: int, direction: SamplingDirection
) -> list[list[int]]:
    if direction not in ("in", "out", "both"):
        raise ValueError(f"direction must be 'in', 'out', or 'both', got {direction!r}.")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()
    for source, target in zip(sources, targets):
        if direction in ("in", "both"):
            adjacency[target].append(source)
        if direction in ("out", "both"):
            adjacency[source].append(target)

    # Keep input edge order for deterministic sampling while removing duplicate
    # candidates introduced by reciprocal edges or repeated COO entries.
    for node_id, candidates in enumerate(adjacency):
        seen: set[int] = set()
        adjacency[node_id] = [
            candidate
            for candidate in candidates
            if not (candidate in seen or seen.add(candidate))
        ]
    return adjacency


@dataclass(frozen=True)
class SamplingSummary:
    """Small immutable summary attached to a sampled subgraph."""

    seed_nodes: int
    sampled_nodes_per_hop: tuple[int, ...]
    num_edges: int


def _copy_sampled_data(
    data: Data,
    *,
    node_ids: Tensor,
    edge_ids: Tensor,
    local_edge_index: Tensor,
) -> Data:
    """Clone and slice node/edge attributes while preserving graph attributes."""

    sampled = data.clone()
    for key, value in data:
        if key == "edge_index" or key in {"n_id", "input_id", "seed_node_index"}:
            continue
        if not isinstance(value, Tensor):
            continue
        if data.is_edge_attr(key):
            cat_dim = data.__cat_dim__(key, value)
            if not isinstance(cat_dim, int):
                raise ValueError(f"Unsupported edge attribute dimension for {key!r}: {cat_dim!r}.")
            sampled[key] = value.index_select(cat_dim % value.ndim, edge_ids)
        elif data.is_node_attr(key):
            cat_dim = data.__cat_dim__(key, value)
            if not isinstance(cat_dim, int):
                raise ValueError(f"Unsupported node attribute dimension for {key!r}: {cat_dim!r}.")
            sampled[key] = value.index_select(cat_dim % value.ndim, node_ids)

    sampled.edge_index = local_edge_index
    sampled.num_nodes = int(node_ids.numel())
    return sampled


def sample_layerwise(
    data: Data,
    seed_nodes: Tensor | Iterable[int],
    num_neighbors: int | Sequence[int],
    *,
    seed: int = 42,
    direction: SamplingDirection = "in",
    replace: bool = False,
) -> Data:
    """Sample a multi-hop replay subgraph around seed nodes.

    ``num_neighbors=[15, 10]`` means sample at most 15 neighbors for each
    seed at hop 1 and at most 10 neighbors for each newly discovered node at
    hop 2.  ``-1`` keeps all available neighbors.  The returned ``Data`` has:

    - ``n_id``: global node IDs in local node order (seeds come first);
    - ``input_id``: global IDs of the requested seed nodes;
    - ``seed_node_index``: local indices of those seeds;
    - ``batch_size``: number of seed nodes;
    - ``node_hop``: hop number for every sampled local node;
    - ``num_sampled_nodes``: node counts for seeds and each sampled hop.

    By default, incoming neighbors are used, matching PyG's message-passing
    convention for ``edge_index=[source, target]``.  Use ``direction='both'``
    for undirected graphs represented with a single direction per edge.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    if not isinstance(replace, bool):
        raise TypeError(f"replace must be a boolean, got {replace!r}.")
    neighbors_per_hop = _normalise_num_neighbors(num_neighbors)
    num_nodes, edge_index_cpu, device = _validate_graph(data)
    seed_list = _normalise_seed_nodes(seed_nodes, num_nodes=num_nodes, device=device)
    adjacency = _build_adjacency(
        edge_index_cpu, num_nodes=num_nodes, direction=direction
    )
    rng = random.Random(seed)

    sampled_nodes: list[int] = list(seed_list)
    sampled_set = set(sampled_nodes)
    frontier = list(seed_list)
    node_hop: list[int] = [0] * len(seed_list)
    sampled_per_hop = [len(seed_list)]

    for hop, limit in enumerate(neighbors_per_hop, start=1):
        next_frontier: list[int] = []
        for node_id in frontier:
            candidates = [
                candidate
                for candidate in adjacency[node_id]
                if candidate not in sampled_set
            ]
            if limit == -1 or limit >= len(candidates):
                chosen = candidates
            elif replace:
                chosen = [rng.choice(candidates) for _ in range(limit)]
            else:
                chosen = rng.sample(candidates, limit)

            for candidate in chosen:
                if candidate in sampled_set:
                    continue
                sampled_set.add(candidate)
                sampled_nodes.append(candidate)
                next_frontier.append(candidate)
                node_hop.append(hop)
        frontier = next_frontier
        sampled_per_hop.append(len(next_frontier))
        if not frontier:
            # Later hops cannot discover anything if the current frontier is empty.
            sampled_per_hop.extend([0] * (len(neighbors_per_hop) - hop))
            break

    node_ids_cpu = torch.tensor(sampled_nodes, dtype=torch.long)
    node_mask_cpu = torch.zeros(num_nodes, dtype=torch.bool)
    if sampled_nodes:
        node_mask_cpu[node_ids_cpu] = True
    edge_mask_cpu = node_mask_cpu[edge_index_cpu[0]] & node_mask_cpu[edge_index_cpu[1]]
    edge_ids = edge_mask_cpu.nonzero(as_tuple=False).flatten().to(device)

    selected_edges_cpu = edge_index_cpu[:, edge_mask_cpu]
    local_index_cpu = torch.full((num_nodes,), -1, dtype=torch.long)
    local_index_cpu[node_ids_cpu] = torch.arange(len(sampled_nodes), dtype=torch.long)
    local_edge_index = local_index_cpu[selected_edges_cpu].to(device)
    node_ids = node_ids_cpu.to(device)
    sampled = _copy_sampled_data(
        data,
        node_ids=node_ids,
        edge_ids=edge_ids,
        local_edge_index=local_edge_index,
    )

    sampled.n_id = node_ids
    sampled.input_id = torch.tensor(seed_list, dtype=torch.long, device=device)
    sampled.seed_node_index = torch.arange(len(seed_list), dtype=torch.long, device=device)
    sampled.batch_size = len(seed_list)
    sampled.node_hop = torch.tensor(node_hop, dtype=torch.long, device=device)
    sampled.num_sampled_nodes = torch.tensor(sampled_per_hop, dtype=torch.long, device=device)
    sampled.num_sampled_edges = int(local_edge_index.size(1))
    sampled.sampling_summary = SamplingSummary(
        seed_nodes=len(seed_list),
        sampled_nodes_per_hop=tuple(sampled_per_hop),
        num_edges=int(local_edge_index.size(1)),
    )
    return sampled


class LayerWiseNeighborLoader:
    """Simple deterministic mini-batch loader backed by ``sample_layerwise``."""

    def __init__(
        self,
        data: Data,
        seed_nodes: Tensor | Iterable[int],
        num_neighbors: int | Sequence[int],
        *,
        batch_size: int = 64,
        shuffle: bool = False,
        seed: int = 42,
        direction: SamplingDirection = "in",
        replace: bool = False,
        drop_last: bool = False,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be a positive integer, got {batch_size!r}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise TypeError("shuffle and drop_last must be booleans.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {seed!r}.")
        # Validate once at construction and keep a cloned graph reference so
        # caller-side mutation cannot change the seed range unexpectedly.
        num_nodes, _, _ = _validate_graph(data)
        self.data = data
        self.seed_nodes = _normalise_seed_nodes(
            seed_nodes, num_nodes=num_nodes, device=data.edge_index.device
        )
        self.num_neighbors = _normalise_num_neighbors(num_neighbors)
        if direction not in ("in", "out", "both"):
            raise ValueError(f"direction must be 'in', 'out', or 'both', got {direction!r}.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.direction = direction
        self.replace = replace
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.seed_nodes) // self.batch_size
        return math.ceil(len(self.seed_nodes) / self.batch_size)

    def __iter__(self) -> Iterator[Data]:
        order = list(self.seed_nodes)
        if self.shuffle:
            random.Random(self.seed).shuffle(order)
        for batch_index, start in enumerate(range(0, len(order), self.batch_size)):
            batch_seeds = order[start : start + self.batch_size]
            if self.drop_last and len(batch_seeds) < self.batch_size:
                break
            yield sample_layerwise(
                self.data,
                batch_seeds,
                self.num_neighbors,
                seed=self.seed + batch_index,
                direction=self.direction,
                replace=self.replace,
            )


def build_replay_loader(
    data: Data,
    seed_nodes: Tensor | Iterable[int],
    num_neighbors: int | Sequence[int],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
    direction: SamplingDirection = "in",
    replace: bool = False,
    drop_last: bool = False,
) -> LayerWiseNeighborLoader:
    """Construct a replay loader from global seed nodes.

    This is the integration point for ``ReplayBuffer.sample(...)`` and later
    GCN/GAT training code.
    """

    return LayerWiseNeighborLoader(
        data,
        seed_nodes,
        num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        direction=direction,
        replace=replace,
        drop_last=drop_last,
    )


def _toy_graph() -> Data:
    source = torch.arange(11, dtype=torch.long)
    target = source + 1
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    x = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=torch.ones(edge_index.size(1)),
        num_nodes=12,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a layer-wise sampling demo.")
    parser.add_argument("--seed-node", type=int, default=0)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[2, 2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = _toy_graph()
    sampled = sample_layerwise(
        graph,
        [args.seed_node],
        args.neighbors,
        seed=args.seed,
        direction="both",
    )
    print("global_n_id=", sampled.n_id.tolist(), sep="")
    print("local_edge_index=", sampled.edge_index.tolist(), sep="")
    print("node_hop=", sampled.node_hop.tolist(), sep="")
    print("num_sampled_nodes=", sampled.num_sampled_nodes.tolist(), sep="")


if __name__ == "__main__":
    _main()
