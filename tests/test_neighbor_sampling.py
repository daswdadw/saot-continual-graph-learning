"""Tests for member 3 layer-wise neighbor sampling."""

import torch
import pytest
from torch_geometric.data import Data

from data.neighbor_sampling import (
    LayerWiseNeighborLoader,
    build_replay_loader,
    sample_layerwise,
)


def make_graph() -> Data:
    source = torch.arange(7, dtype=torch.long)
    target = source + 1
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    return Data(
        x=torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3),
        y=torch.arange(8),
        edge_index=edge_index,
        edge_weight=torch.arange(edge_index.size(1), dtype=torch.float32),
        num_nodes=8,
    )


def test_sample_has_global_mapping_and_valid_local_edges() -> None:
    graph = make_graph()
    sampled = sample_layerwise(graph, [0, 4], [2, 2], seed=7, direction="both")

    assert sampled.batch_size == 2
    assert sampled.input_id.tolist() == [0, 4]
    assert sampled.seed_node_index.tolist() == [0, 1]
    assert sampled.n_id[:2].tolist() == [0, 4]
    assert sampled.x.shape[0] == sampled.n_id.numel()
    assert sampled.y.shape[0] == sampled.n_id.numel()
    assert sampled.edge_weight.numel() == sampled.edge_index.size(1)
    assert sampled.edge_index.numel() == 0 or int(sampled.edge_index.min()) >= 0
    assert sampled.edge_index.numel() == 0 or int(sampled.edge_index.max()) < sampled.n_id.numel()
    assert sampled.num_sampled_nodes.tolist()[0] == 2
    assert torch.equal(sampled.node_hop[:2], torch.zeros(2, dtype=torch.long))


def test_sampling_is_layer_bounded_and_reproducible() -> None:
    graph = make_graph()
    first = sample_layerwise(graph, [0], [1, 1], seed=13, direction="both")
    second = sample_layerwise(graph, [0], [1, 1], seed=13, direction="both")
    assert torch.equal(first.n_id, second.n_id)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.node_hop, second.node_hop)
    assert first.num_sampled_nodes.tolist()[1] <= 1
    assert first.num_sampled_nodes.tolist()[2] <= 1


def test_all_neighbors_and_direction() -> None:
    graph = make_graph()
    sampled = sample_layerwise(graph, [3], [-1], direction="both")
    assert set(sampled.n_id.tolist()) == {2, 3, 4}

    # Use a directed chain here so incoming and outgoing neighbors are distinct.
    directed = Data(
        x=graph.x,
        y=graph.y,
        edge_index=graph.edge_index[:, :7],
        edge_weight=graph.edge_weight[:7],
        num_nodes=graph.num_nodes,
    )
    # With the edge convention source -> target, incoming neighbors of node 3
    # contain only node 2 in this directed graph.
    incoming = sample_layerwise(directed, [3], [-1], direction="in")
    assert set(incoming.n_id.tolist()) == {2, 3}

    outgoing = sample_layerwise(directed, [3], [-1], direction="out")
    assert set(outgoing.n_id.tolist()) == {3, 4}


def test_duplicate_seed_nodes_are_removed_but_order_is_kept() -> None:
    graph = make_graph()
    sampled = sample_layerwise(graph, [5, 1, 5, 1], [0], seed=1)
    assert sampled.input_id.tolist() == [5, 1]
    assert sampled.n_id[:2].tolist() == [5, 1]
    assert sampled.batch_size == 2


def test_loader_batches_seed_nodes_and_is_reusable() -> None:
    graph = make_graph()
    loader = build_replay_loader(
        graph,
        seed_nodes=range(5),
        num_neighbors=[1, 1],
        batch_size=2,
        shuffle=False,
        direction="both",
    )
    batches_first = list(loader)
    batches_second = list(loader)
    assert isinstance(loader, LayerWiseNeighborLoader)
    assert len(loader) == 3
    assert [batch.batch_size for batch in batches_first] == [2, 2, 1]
    assert [batch.input_id.tolist() for batch in batches_first] == [
        [0, 1],
        [2, 3],
        [4],
    ]
    assert [batch.n_id.tolist() for batch in batches_first] == [
        batch.n_id.tolist() for batch in batches_second
    ]


def test_loader_drop_last_and_shuffle_seed() -> None:
    graph = make_graph()
    first = build_replay_loader(
        graph, range(5), [1], batch_size=2, shuffle=True, seed=8, drop_last=True
    )
    second = build_replay_loader(
        graph, range(5), [1], batch_size=2, shuffle=True, seed=8, drop_last=True
    )
    first_ids = [batch.input_id.tolist() for batch in first]
    second_ids = [batch.input_id.tolist() for batch in second]
    assert len(first) == 2
    assert first_ids == second_ids


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_nodes": [-1], "num_neighbors": [1]},
        {"seed_nodes": [8], "num_neighbors": [1]},
        {"seed_nodes": [0], "num_neighbors": []},
        {"seed_nodes": [0], "num_neighbors": [-2]},
    ],
)
def test_invalid_sampling_inputs_raise(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        sample_layerwise(make_graph(), **kwargs)
