"""Tests for the member 3 replay buffer."""

from pathlib import Path

import pytest
import torch

from data.replay_buffer import ReplayBuffer


def test_budget_deduplicates_and_never_overflows() -> None:
    buffer = ReplayBuffer(budget=5, seed=7)
    buffer.add(0, [0, 1, 1, 2])
    buffer.add(1, torch.tensor([10, 11, 12, 13]))

    assert len(buffer) == 5
    assert len(buffer) <= buffer.budget
    assert len(buffer.records) == len(set(buffer.records))
    assert set(buffer.task_ids) == {0, 1}


def test_zero_budget_is_a_safe_noop() -> None:
    buffer = ReplayBuffer(budget=0, seed=42)
    assert buffer.add(0, [1, 2, 3]) == 0
    assert len(buffer) == 0
    assert buffer.task_ids == ()
    assert buffer.sample(10).numel() == 0


def test_balanced_sampling_returns_task_ids_and_is_reproducible() -> None:
    buffer = ReplayBuffer(budget=12, seed=3)
    buffer.add(0, range(6))
    buffer.add(1, range(100, 106))

    nodes_1, tasks_1 = buffer.sample(6, seed=99, return_task_ids=True)
    nodes_2, tasks_2 = buffer.sample(6, seed=99, return_task_ids=True)

    assert torch.equal(nodes_1, nodes_2)
    assert tasks_1 == tasks_2
    assert len(tasks_1) == 6
    assert abs(tasks_1.count(0) - tasks_1.count(1)) <= 1
    assert all((task_id, int(node_id)) in buffer for task_id, node_id in zip(tasks_1, nodes_1))


def test_sampling_caps_without_replacement_and_can_replace() -> None:
    buffer = ReplayBuffer(budget=3, seed=1)
    buffer.add(0, [4, 5, 6])

    capped = buffer.sample(10, seed=2)
    assert len(capped) == 3
    assert len(set(capped.tolist())) == 3

    repeated = buffer.sample(10, replace=True, seed=2)
    assert len(repeated) == 10
    assert set(repeated.tolist()).issubset({4, 5, 6})


def test_task_filter_and_remove() -> None:
    buffer = ReplayBuffer(budget=10, seed=1)
    buffer.add_many({0: [1, 2], 1: [10, 11], 2: [20]})

    nodes, tasks = buffer.sample(10, task_ids=[1], return_task_ids=True, seed=4)
    assert set(nodes.tolist()) == {10, 11}
    assert tasks == [1, 1]
    assert buffer.remove_task(1) == 2
    assert buffer.nodes_for_task(1) == ()
    assert 1 not in buffer.task_ids


def test_save_and_load_preserves_records_and_rng(tmp_path: Path) -> None:
    path = tmp_path / "replay_buffer.pt"
    original = ReplayBuffer(budget=8, seed=17)
    original.add(0, range(5))
    original.add(1, range(100, 105))
    original.save(path)

    restored = ReplayBuffer.load(path)
    assert restored.budget == original.budget
    assert restored.task_sizes == original.task_sizes
    assert restored.records == original.records

    expected_nodes, expected_tasks = original.sample(4, return_task_ids=True)
    actual_nodes, actual_tasks = restored.sample(4, return_task_ids=True)
    assert torch.equal(expected_nodes, actual_nodes)
    assert expected_tasks == actual_tasks


def test_state_budget_mismatch_is_rejected() -> None:
    first = ReplayBuffer(budget=4)
    first.add(0, [1, 2])
    second = ReplayBuffer(budget=5)
    with pytest.raises(ValueError, match="does not match"):
        second.load_state_dict(first.state_dict())


@pytest.mark.parametrize("bad_budget", [-1, 1.5, True])
def test_invalid_budget_is_rejected(bad_budget: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReplayBuffer(bad_budget)  # type: ignore[arg-type]


def test_invalid_node_ids_are_rejected() -> None:
    buffer = ReplayBuffer(budget=4)
    with pytest.raises(ValueError):
        buffer.add(0, [-1])
    with pytest.raises(TypeError):
        buffer.add(0, [True])
    with pytest.raises(ValueError):
        buffer.add(0, torch.tensor([[1, 2]]))
