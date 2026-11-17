import torch

from data.replay_buffer import induced_edge_index


def test_induced_edges_support_global_ids_larger_than_sampled_nodes():
    edges = torch.tensor([[0, 2, 7, 1], [2, 7, 1, 9]])
    nodes = torch.tensor([1, 2, 7])
    actual = induced_edge_index(edges, nodes)
    expected = torch.tensor([[1, 2], [2, 0]])
    assert torch.equal(actual, expected)


def test_induced_edges_empty_nodes_return_empty_edge_index():
    actual = induced_edge_index(torch.tensor([[0], [1]]), torch.empty(0, dtype=torch.long))
    assert actual.shape == (2, 0)
