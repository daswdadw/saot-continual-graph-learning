"""Graph augmentations used to construct the two SAOT input views.

The module is deliberately independent from the dataset pipeline.  It accepts a
PyTorch Geometric ``Data`` or ``Batch`` object and returns cloned objects, so an
augmentation never changes the caller's graph in place.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

MaskMode = Literal["element", "feature", "node"]


def _validate_probability(name: str, probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}.")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer node indices.")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")
    if edge_index.numel() == 0:
        return
    minimum = int(edge_index.min())
    maximum = int(edge_index.max())
    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "edge_index contains an out-of-range node index: "
            f"valid range is [0, {num_nodes - 1}], found [{minimum}, {maximum}]."
        )


def drop_edges(
    edge_index: Tensor,
    drop_probability: float,
    *,
    num_nodes: int,
    seed: int = 0,
    preserve_undirected: bool = True,
) -> tuple[Tensor, Tensor]:
    """Randomly remove edges and return ``(new_edge_index, keep_mask)``.

    When ``preserve_undirected`` is true, all stored directions and duplicates
    of the same unordered node pair share one random decision.  This prevents
    an undirected graph from becoming asymmetric after augmentation.

    Args:
        edge_index: COO edge indices with shape ``[2, num_edges]``.
        drop_probability: Probability of removing an edge (or undirected pair).
        num_nodes: Number of nodes used to validate and group the indices.
        seed: Local random seed. Global PyTorch RNG state is not changed.
        preserve_undirected: Keep/drop reciprocal edges together.
    """

    _validate_probability("drop_probability", drop_probability)
    _validate_edge_index(edge_index, num_nodes)

    num_edges = edge_index.size(1)
    if num_edges == 0 or drop_probability == 0.0:
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        return edge_index.clone(), keep_mask
    if drop_probability == 1.0:
        keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
        return edge_index[:, keep_mask].clone(), keep_mask

    generator = _make_generator(edge_index.device, seed)
    if preserve_undirected:
        source, target = edge_index
        low = torch.minimum(source, target)
        high = torch.maximum(source, target)
        pair_keys = low * max(num_nodes, 1) + high
        _, inverse = torch.unique(pair_keys, sorted=False, return_inverse=True)
        num_pairs = int(inverse.max()) + 1
        keep_pairs = (
            torch.rand(num_pairs, device=edge_index.device, generator=generator)
            >= drop_probability
        )
        keep_mask = keep_pairs[inverse]
    else:
        keep_mask = (
            torch.rand(num_edges, device=edge_index.device, generator=generator)
            >= drop_probability
        )

    return edge_index[:, keep_mask].clone(), keep_mask


def mask_node_features(
    features: Tensor,
    mask_probability: float,
    *,
    seed: int = 0,
    mode: MaskMode = "element",
    fill_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Mask node features and return ``(masked_features, mask)``.

    ``mode='element'`` independently masks entries, ``'feature'`` masks whole
    feature dimensions, and ``'node'`` masks complete node feature rows.  The
    returned Boolean mask always has the same shape as ``features`` and is true
    exactly where values were replaced.
    """

    _validate_probability("mask_probability", mask_probability)
    if features.ndim != 2:
        raise ValueError(
            "features must have shape [num_nodes, num_features], "
            f"got {tuple(features.shape)}."
        )
    if mode not in ("element", "feature", "node"):
        raise ValueError(
            f"mode must be 'element', 'feature', or 'node', got {mode!r}."
        )

    if mask_probability == 0.0 or features.numel() == 0:
        mask = torch.zeros_like(features, dtype=torch.bool)
        return features.clone(), mask
    if mask_probability == 1.0:
        mask = torch.ones_like(features, dtype=torch.bool)
    else:
        generator = _make_generator(features.device, seed)
        if mode == "element":
            random_shape = features.shape
        elif mode == "feature":
            random_shape = (1, features.size(1))
        else:
            random_shape = (features.size(0), 1)
        sampled_mask = (
            torch.rand(random_shape, device=features.device, generator=generator)
            < mask_probability
        )
        mask = sampled_mask.expand_as(features).clone()

    masked_features = features.clone()
    masked_features.masked_fill_(mask, fill_value)
    return masked_features, mask


def _select_edge_attributes(data: Data, view: Data, keep_mask: Tensor) -> None:
    """Apply an edge mask to every tensor PyG identifies as an edge attribute."""

    edge_ids = keep_mask.nonzero(as_tuple=False).flatten()
    for key, value in data:
        if key == "edge_index" or not isinstance(value, Tensor):
            continue
        if data.is_edge_attr(key):
            dimension = data.__cat_dim__(key, value)
            if not isinstance(dimension, int):
                raise ValueError(
                    f"Unsupported concatenation dimension {dimension!r} "
                    f"for edge attribute {key!r}."
                )
            view[key] = value.index_select(dimension % value.ndim, edge_ids)


@dataclass(frozen=True)
class GraphAugmentor:
    """Configuration and callable interface for SAOT graph augmentation."""

    edge_drop: float = 0.2
    feature_mask: float = 0.2
    feature_mask_mode: MaskMode = "element"
    preserve_undirected: bool = True
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        _validate_probability("edge_drop", self.edge_drop)
        _validate_probability("feature_mask", self.feature_mask)
        if self.feature_mask_mode not in ("element", "feature", "node"):
            raise ValueError(
                "feature_mask_mode must be 'element', 'feature', or 'node'."
            )

    def __call__(self, data: Data, *, seed: int) -> Data:
        """Return one augmented clone of ``data`` using a local seed."""

        if not isinstance(data, Data):
            raise TypeError(
                "data must be a torch_geometric.data.Data or Batch instance."
            )
        if data.x is None:
            raise ValueError("data.x is required for node feature masking.")
        if data.edge_index is None:
            raise ValueError("data.edge_index is required for edge dropping.")
        if data.x.device != data.edge_index.device:
            raise ValueError("data.x and data.edge_index must be on the same device.")

        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Unable to infer data.num_nodes.")
        if data.x.size(0) != num_nodes:
            raise ValueError(
                f"data.x has {data.x.size(0)} rows but data.num_nodes is {num_nodes}."
            )

        # Separate deterministic streams prevent the number of edges from
        # changing the feature-mask random sequence.
        augmented_edges, keep_mask = drop_edges(
            data.edge_index,
            self.edge_drop,
            num_nodes=num_nodes,
            seed=seed,
            preserve_undirected=self.preserve_undirected,
        )
        augmented_features, feature_mask = mask_node_features(
            data.x,
            self.feature_mask,
            seed=seed + 1,
            mode=self.feature_mask_mode,
            fill_value=self.fill_value,
        )

        view = data.clone()
        _select_edge_attributes(data, view, keep_mask)
        view.edge_index = augmented_edges
        view.x = augmented_features
        # Metadata is useful for tests and visualisation but does not affect
        # PyG message passing. Prefixes avoid colliding with dataset fields.
        view.augmentation_edge_keep_mask = keep_mask
        view.augmentation_feature_mask = feature_mask
        view.augmentation_seed = int(seed)
        return view

    def two_views(self, data: Data, *, seed: int) -> tuple[Data, Data]:
        """Create two independently sampled, reproducible augmented views."""

        return self(data, seed=seed), self(data, seed=seed + 10_000)


def create_augmented_views(
    data: Data,
    *,
    edge_drop: float = 0.2,
    feature_mask: float = 0.2,
    seed: int = 42,
    feature_mask_mode: MaskMode = "element",
    preserve_undirected: bool = True,
    fill_value: float = 0.0,
) -> tuple[Data, Data]:
    """Convenience function for constructing the two SAOT graph views."""

    augmentor = GraphAugmentor(
        edge_drop=edge_drop,
        feature_mask=feature_mask,
        feature_mask_mode=feature_mask_mode,
        preserve_undirected=preserve_undirected,
        fill_value=fill_value,
    )
    return augmentor.two_views(data, seed=seed)


def _toy_graph() -> Data:
    source = torch.arange(8, dtype=torch.long)
    target = (source + 1) % 8
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    return Data(x=features, edge_index=edge_index, num_nodes=8)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run SAOT graph augmentation demo.")
    parser.add_argument("--edge-drop", type=float, default=0.2)
    parser.add_argument("--feature-mask", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = _toy_graph()
    view_1, view_2 = create_augmented_views(
        graph,
        edge_drop=args.edge_drop,
        feature_mask=args.feature_mask,
        seed=args.seed,
    )
    print(
        "original_edges=", graph.num_edges,
        " view_1_edges=", view_1.num_edges,
        " view_2_edges=", view_2.num_edges,
        sep="",
    )
    print(
        "view_1_masked_features=",
        int(view_1.augmentation_feature_mask.sum()),
        " view_2_masked_features=",
        int(view_2.augmentation_feature_mask.sum()),
        sep="",
    )


if __name__ == "__main__":
    _main()
