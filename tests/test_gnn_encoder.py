"""Tests for the member 3 GCN/GAT encoder."""

import pytest
import torch
from torch_geometric.data import Data

from data.neighbor_sampling import sample_layerwise
from models.gnn_encoder import GNNEncoder, build_gnn_encoder


def make_graph(num_nodes: int = 12, input_dim: int = 5) -> Data:
    source = torch.arange(num_nodes - 1)
    target = source + 1
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    return Data(
        x=torch.randn(num_nodes, input_dim),
        edge_index=edge_index,
        edge_weight=torch.ones(edge_index.size(1)),
        num_nodes=num_nodes,
    )


def test_two_layer_gcn_shape_and_gradients() -> None:
    graph = make_graph()
    model = GNNEncoder(5, hidden_dim=8, output_dim=6, encoder_type="gcn", num_layers=2)
    output = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight)
    assert output.shape == (graph.num_nodes, 6)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_three_layer_gat_and_intermediate_outputs() -> None:
    graph = make_graph()
    model = GNNEncoder(
        5,
        hidden_dim=8,
        output_dim=4,
        encoder_type="gat",
        num_layers=3,
        gat_heads=2,
    )
    outputs = model(graph.x, graph.edge_index, return_all_layers=True)
    assert len(outputs) == 3
    assert outputs[-1].shape == (graph.num_nodes, 4)
    assert outputs[0].shape == (graph.num_nodes, 8)


def test_encoder_consumes_sampled_local_subgraph() -> None:
    graph = make_graph()
    sampled = sample_layerwise(graph, [0, 5], [2, 2], direction="both", seed=4)
    model = GNNEncoder(5, hidden_dim=8, output_dim=8)
    output = model(sampled.x, sampled.edge_index, edge_weight=sampled.edge_weight)
    assert output.shape == (sampled.n_id.numel(), 8)


def test_config_builder_and_eval_dropout() -> None:
    model = build_gnn_encoder(
        {"encoder_type": "gcn", "hidden_dim": 7, "num_layers": 2, "dropout": 0.5},
        input_dim=5,
        output_dim=3,
    )
    assert model.output_dim == 3
    model.eval()
    graph = make_graph()
    first = model(graph.x, graph.edge_index)
    second = model(graph.x, graph.edge_index)
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"input_dim": 4, "encoder_type": "mlp"},
        {"input_dim": 4, "num_layers": 0},
        {"input_dim": 4, "dropout": 1.0},
    ],
)
def test_invalid_encoder_config_raises(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        GNNEncoder(**kwargs)
