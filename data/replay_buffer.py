"""Fixed-budget replay buffer for continual graph learning.

The buffer stores node identifiers grouped by task.  It intentionally stores
identifiers rather than tensors or full subgraphs so that it remains small and
can later be paired with a dataset-specific neighbor sampler.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _validate_budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget must be a non-negative integer, got {budget!r}.")
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}.")
    return budget


def _validate_task_id(task_id: Hashable) -> None:
    try:
        hash(task_id)
    except TypeError as exc:
        raise TypeError(f"task_id must be hashable, got {task_id!r}.") from exc


def _normalise_node_ids(node_ids: Tensor | Iterable[int]) -> list[int]:
    if isinstance(node_ids, Tensor):
        if node_ids.ndim != 1:
            raise ValueError(
                "node_ids tensor must be one-dimensional, "
                f"got shape {tuple(node_ids.shape)}."
            )
        if node_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"node_ids tensor must have an integer dtype, got {node_ids.dtype}.")
        values = node_ids.detach().cpu().tolist()
    else:
        try:
            values = list(node_ids)
        except TypeError as exc:
            raise TypeError("node_ids must be a 1-D tensor or an iterable of integers.") from exc

    normalised: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("node IDs must be integers, not booleans.")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"node ID {value!r} is not an integer.") from exc
        if integer != value:
            raise ValueError(f"node ID {value!r} is not an exact integer.")
        if integer < 0:
            raise ValueError(f"node IDs must be non-negative, got {integer}.")
        normalised.append(integer)
    return normalised


class ReplayBuffer:
    """A deterministic, task-aware, fixed-budget node replay buffer.

    Args:
        budget: Maximum total number of ``(task_id, node_id)`` records.
        seed: Seed for eviction and sampling when no per-call seed is supplied.

    The same node identifier may appear in different tasks, but duplicate
    ``(task_id, node_id)`` records are stored only once.
    """

    STATE_VERSION = 1

    def __init__(self, budget: int, *, seed: int = 42) -> None:
        self.budget = _validate_budget(budget)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {seed!r}.")
        self.seed = seed
        self._rng = random.Random(seed)
        # dict insertion order records the order in which tasks first appeared.
        self._storage: dict[Hashable, list[int]] = {}

    def __len__(self) -> int:
        return sum(len(nodes) for nodes in self._storage.values())

    @property
    def num_tasks(self) -> int:
        """Number of tasks with at least one stored node."""

        return sum(bool(nodes) for nodes in self._storage.values())

    @property
    def task_ids(self) -> tuple[Hashable, ...]:
        """Task IDs currently represented in the buffer, in insertion order."""

        return tuple(task_id for task_id, nodes in self._storage.items() if nodes)

    @property
    def task_sizes(self) -> dict[Hashable, int]:
        """Number of stored nodes per task."""

        return {
            task_id: len(nodes)
            for task_id, nodes in self._storage.items()
            if nodes
        }

    @property
    def records(self) -> tuple[tuple[Hashable, int], ...]:
        """All stored ``(task_id, node_id)`` records in stable order."""

        return tuple(
            (task_id, node_id)
            for task_id, nodes in self._storage.items()
            for node_id in nodes
        )

    def __iter__(self):
        return iter(self.records)

    def __contains__(self, record: object) -> bool:
        if not isinstance(record, tuple) or len(record) != 2:
            return False
        task_id, node_id = record
        nodes = self._storage.get(task_id)
        return nodes is not None and node_id in nodes

    def clear(self) -> None:
        """Remove every record while keeping the buffer configuration."""

        self._storage.clear()

    def add(self, task_id: Hashable, node_ids: Tensor | Iterable[int]) -> int:
        """Add node IDs for a task and enforce the global budget.

        Duplicate node IDs within a task are ignored.  If adding records would
        exceed the budget, records are randomly evicted from overfull tasks
        using the buffer's seeded RNG.  The returned value is the number of
        records present for this task after deduplication and rebalancing.
        """

        _validate_task_id(task_id)
        incoming = _normalise_node_ids(node_ids)
        if self.budget == 0:
            return 0

        existing = self._storage.setdefault(task_id, [])
        existing_set = set(existing)
        existing.extend(node_id for node_id in incoming if node_id not in existing_set)
        self._rebalance()
        return len(self._storage.get(task_id, []))

    def add_many(self, task_to_node_ids: Mapping[Hashable, Tensor | Iterable[int]]) -> None:
        """Add several task batches in mapping iteration order."""

        for task_id, node_ids in task_to_node_ids.items():
            self.add(task_id, node_ids)

    def remove_task(self, task_id: Hashable) -> int:
        """Remove all records for one task and return the removed count."""

        removed = len(self._storage.pop(task_id, []))
        return removed

    def nodes_for_task(self, task_id: Hashable) -> tuple[int, ...]:
        """Return a read-only snapshot of nodes stored for ``task_id``."""

        return tuple(self._storage.get(task_id, ()))

    def _rebalance(self) -> None:
        if self.budget == 0:
            self._storage.clear()
            return

        active_tasks = [task_id for task_id, nodes in self._storage.items() if nodes]
        if not active_tasks:
            self._storage.clear()
            return

        # Give each active task an approximately equal target. Underfull tasks
        # release their unused quota to tasks that still have pending records.
        task_count = len(active_tasks)
        base, remainder = divmod(self.budget, task_count)
        targets = {
            task_id: base + (index < remainder)
            for index, task_id in enumerate(active_tasks)
        }
        kept: dict[Hashable, int] = {
            task_id: min(len(self._storage[task_id]), target)
            for task_id, target in targets.items()
        }
        spare = self.budget - sum(kept.values())
        while spare > 0:
            candidates = [
                task_id
                for task_id in active_tasks
                if kept[task_id] < len(self._storage[task_id])
            ]
            if not candidates:
                break
            # Prioritise the task with the most pending records; insertion order
            # resolves ties, making the policy deterministic for a fixed seed.
            task_id = max(
                candidates,
                key=lambda candidate: len(self._storage[candidate]) - kept[candidate],
            )
            kept[task_id] += 1
            spare -= 1

        for task_id in active_tasks:
            nodes = self._storage[task_id]
            keep_count = kept[task_id]
            if len(nodes) <= keep_count:
                continue
            remove_count = len(nodes) - keep_count
            remove_indices = set(self._rng.sample(range(len(nodes)), remove_count))
            self._storage[task_id] = [
                node_id
                for index, node_id in enumerate(nodes)
                if index not in remove_indices
            ]

        self._storage = {
            task_id: nodes
            for task_id, nodes in self._storage.items()
            if nodes
        }

    def _rng_for_call(self, seed: int | None) -> random.Random:
        if seed is None:
            return self._rng
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {seed!r}.")
        return random.Random(seed)

    def sample(
        self,
        num_samples: int,
        *,
        task_ids: Iterable[Hashable] | None = None,
        balanced: bool = True,
        replace: bool = False,
        seed: int | None = None,
        return_task_ids: bool = False,
    ) -> Tensor | tuple[Tensor, list[Hashable]]:
        """Sample node IDs, optionally returning their task IDs as well.

        Without replacement, requests larger than the available population are
        capped at the available size. With replacement, exactly ``num_samples``
        records are returned unless the selected task set is empty. Balanced
        sampling cycles through shuffled tasks, avoiding one task dominating a
        batch when multiple tasks are available.
        """

        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError(f"num_samples must be a non-negative integer, got {num_samples!r}.")
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")

        selected_tasks = None if task_ids is None else set(task_ids)
        available = {
            task_id: list(nodes)
            for task_id, nodes in self._storage.items()
            if nodes and (selected_tasks is None or task_id in selected_tasks)
        }
        rng = self._rng_for_call(seed)
        if num_samples == 0 or not available:
            empty = torch.empty(0, dtype=torch.long)
            return (empty, []) if return_task_ids else empty

        task_order = list(available)
        rng.shuffle(task_order)
        selected_nodes: list[int] = []
        selected_task_ids: list[Hashable] = []

        if replace:
            for index in range(num_samples):
                task_id = task_order[index % len(task_order)] if balanced else rng.choice(task_order)
                node_id = rng.choice(available[task_id])
                selected_nodes.append(node_id)
                selected_task_ids.append(task_id)
        else:
            target_count = min(num_samples, sum(map(len, available.values())))
            remaining = {task_id: list(nodes) for task_id, nodes in available.items()}
            while len(selected_nodes) < target_count:
                candidates = [task_id for task_id in task_order if remaining[task_id]]
                if not candidates:
                    break
                if balanced:
                    # Iterate through the shuffled task order in rounds.
                    task_id = candidates[len(selected_nodes) % len(candidates)]
                else:
                    task_id = rng.choice(candidates)
                node_index = rng.randrange(len(remaining[task_id]))
                selected_nodes.append(remaining[task_id].pop(node_index))
                selected_task_ids.append(task_id)

        # Shuffle the final batch so task order does not leak into training.
        order = list(range(len(selected_nodes)))
        rng.shuffle(order)
        selected_nodes = [selected_nodes[index] for index in order]
        selected_task_ids = [selected_task_ids[index] for index in order]
        node_tensor = torch.tensor(selected_nodes, dtype=torch.long)
        if return_task_ids:
            return node_tensor, selected_task_ids
        return node_tensor

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot, including the RNG state."""

        return {
            "version": self.STATE_VERSION,
            "budget": self.budget,
            "seed": self.seed,
            "tasks": [(task_id, list(nodes)) for task_id, nodes in self._storage.items()],
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot into this buffer after validating its budget."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping returned by state_dict().")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(f"Unsupported replay buffer state version: {state.get('version')!r}.")
        if state.get("budget") != self.budget:
            raise ValueError(
                f"State budget {state.get('budget')!r} does not match buffer budget {self.budget}."
            )
        tasks = state.get("tasks")
        if not isinstance(tasks, Sequence):
            raise TypeError("state['tasks'] must be a sequence of (task_id, node_ids) pairs.")

        restored: dict[Hashable, list[int]] = {}
        for item in tasks:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError(f"Invalid task entry in state: {item!r}.")
            task_id, node_ids = item
            _validate_task_id(task_id)
            if task_id in restored:
                raise ValueError(f"Duplicate task ID in state: {task_id!r}.")
            restored[task_id] = _normalise_node_ids(node_ids)

        self._storage = restored
        self._rebalance()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)

    def save(self, path: str | Path) -> None:
        """Save the buffer snapshot with ``torch.save``."""

        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ReplayBuffer":
        """Load a buffer saved by :meth:`save`."""

        state = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError("Saved replay buffer state is not a mapping.")
        buffer = cls(int(state["budget"]), seed=int(state.get("seed", 42)))
        buffer.load_state_dict(state)
        return buffer


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReplayBuffer smoke test.")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=12)
    args = parser.parse_args()

    buffer = ReplayBuffer(args.budget, seed=args.seed)
    buffer.add(0, range(10))
    buffer.add(1, range(10, 20))
    sample, task_ids = buffer.sample(
        args.sample_size,
        seed=args.seed,
        return_task_ids=True,
    )
    print("size=", len(buffer), " task_sizes=", buffer.task_sizes, sep="")
    print("sample_nodes=", sample.tolist(), " sample_tasks=", task_ids, sep="")


if __name__ == "__main__":
    _main()
