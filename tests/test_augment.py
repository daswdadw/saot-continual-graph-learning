"""Tests for member 3 graph augmentations."""

import pytest
import torch
from torch_geometric.data import Data

from data.augment import (
    GraphAugmentor,
    create_augmented_views,
    drop_edges,
    mask_node_features,
)


def make_graph() -> Data:
    source = torch.arange(20, dtype=torch.long)
    target = (source + 1) % 20
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    edge_weight = torch.arange(edge_index.size(1), dtype=torch.float32)
    features = torch.arange(160, dtype=torch.float32).reshape(20, 8) + 1
    return Data(
        x=features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=20,
    )


def edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


def test_same_seed_reproduces_same_pair_without_mutating_input() -> None:
    graph = make_graph()
    original_x = graph.x.clone()
    original_edges = graph.edge_index.clone()

    first_pair = create_augmented_views(graph, seed=42)
    second_pair = create_augmented_views(graph, seed=42)

    for first, second in zip(first_pair, second_pair, strict=True):
        assert torch.equal(first.x, second.x)
        assert torch.equal(first.edge_index, second.edge_index)
        assert first.num_nodes == graph.num_nodes

    assert torch.equal(graph.x, original_x)
    assert torch.equal(graph.edge_index, original_edges)


def test_two_views_are_independently_sampled_and_legal() -> None:
    graph = make_graph()
    view_1, view_2 = create_augmented_views(
        graph, edge_drop=0.35, feature_mask=0.35, seed=7
    )

    assert not (
        torch.equal(view_1.x, view_2.x)
        and torch.equal(view_1.edge_index, view_2.edge_index)
    )
    for view in (view_1, view_2):
        assert view.edge_index.shape[0] == 2
        assert int(view.edge_index.min()) >= 0
        assert int(view.edge_index.max()) < graph.num_nodes
        assert view.x.shape == graph.x.shape
        assert view.edge_weight.numel() == view.num_edges
        assert torch.equal(
            view.edge_weight,
            graph.edge_weight[view.augmentation_edge_keep_mask],
        )


def test_undirected_edges_are_dropped_in_pairs() -> None:
    graph = make_graph()
    augmented, _ = drop_edges(
        graph.edge_index,
        0.5,
        num_nodes=graph.num_nodes,
        seed=12,
        preserve_undirected=True,
    )
    edges = edge_set(augmented)
    assert all((target, source) in edges for source, target in edges)


@pytest.mark.parametrize("mode", ["element", "feature", "node"])
def test_feature_mask_shape_and_fill_value(mode: str) -> None:
    features = torch.ones((10, 6))
    masked, mask = mask_node_features(
        features,
        0.5,
        seed=9,
        mode=mode,
        fill_value=-1.0,
    )
    assert mask.shape == features.shape
    assert torch.all(masked[mask] == -1.0)
    assert torch.all(masked[~mask] == 1.0)

    if mode == "feature":
        assert all(torch.equal(mask[0], row) for row in mask)
    elif mode == "node":
        assert all(row.all() or (~row).all() for row in mask)


def test_probability_boundaries() -> None:
    graph = make_graph()
    unchanged = GraphAugmentor(edge_drop=0.0, feature_mask=0.0)(graph, seed=1)
    assert torch.equal(unchanged.edge_index, graph.edge_index)
    assert torch.equal(unchanged.x, graph.x)

    removed = GraphAugmentor(edge_drop=1.0, feature_mask=1.0)(graph, seed=1)
    assert removed.edge_index.shape == (2, 0)
    assert torch.count_nonzero(removed.x) == 0


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_probabilities_raise(probability: float) -> None:
    graph = make_graph()
    with pytest.raises(ValueError):
        GraphAugmentor(edge_drop=probability)(graph, seed=1)
    with pytest.raises(ValueError):
        mask_node_features(graph.x, probability)
