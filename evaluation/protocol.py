"""Shared sequential continual-learning protocol for every model.

Member 4's SAOT, member 5's baselines and member 6's ablations all expose the
same small surface, so one runner can evaluate any of them identically::

    method.prepare(graph, split, seed=42)
    for task in range(split.num_tasks):
        method.fit_task(task)
        logits = method.predict_logits()   # [num_nodes, num_classes]

``TaskSplit`` follows the JSON schema documented in
``docs/member_3_interfaces.md``: ``task_to_classes`` plus global, zero-based and
contiguous ``node_id`` lists.  Node ids are never re-indexed per task.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from models.gnn_encoder import GNNEncoder

SCENARIOS = ("class-il", "task-il")

__all__ = [
    "SCENARIOS",
    "SequentialMethod",
    "SequentialNodeClassifier",
    "TaskSplit",
    "accuracy",
    "derive_task_masks",
    "restrict_logits",
    "restricted_accuracy",
    "restricted_cross_entropy",
    "set_seed",
]


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch (CUDA included when available)."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TaskSplit:
    """Task-to-class and task-to-node mapping of one continual graph dataset.

    Args:
        num_nodes: Global number of nodes.
        num_classes: Global number of classes.
        task_to_classes: ``{task: [class, ...]}``; tasks are ``0..T-1``.
        task_to_train_ids: Global train node ids per task.
        task_to_val_ids: Global validation node ids per task (may be empty).
        task_to_test_ids: Global test node ids per task.
        task_to_node_ids: Optional superset per task (e.g. unlabeled nodes);
            defaults to the union of train/val/test ids.
        seed: Seed used to build the split.
        source: Human readable provenance (``member1``/``pyg``/``synthetic``).
        num_edges_removed: Inter-task edges dropped when building the split.
    """

    num_nodes: int
    num_classes: int
    task_to_classes: dict[int, list[int]]
    task_to_train_ids: dict[int, Tensor]
    task_to_val_ids: dict[int, Tensor]
    task_to_test_ids: dict[int, Tensor]
    task_to_node_ids: dict[int, Tensor] = field(default_factory=dict)
    seed: int = 42
    source: str = "unknown"
    num_edges_removed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.num_nodes, bool) or not isinstance(self.num_nodes, int):
            raise TypeError("num_nodes must be an integer.")
        if isinstance(self.num_classes, bool) or not isinstance(self.num_classes, int):
            raise TypeError("num_classes must be an integer.")
        if self.num_nodes <= 0 or self.num_classes <= 0:
            raise ValueError("num_nodes and num_classes must be positive.")
        if not self.task_to_classes:
            raise ValueError("task_to_classes must not be empty.")

        expected_tasks = list(range(len(self.task_to_classes)))
        if sorted(self.task_to_classes) != expected_tasks:
            raise ValueError(
                f"task ids must be contiguous 0..{len(self.task_to_classes) - 1}, "
                f"got {sorted(self.task_to_classes)}."
            )

        seen_classes: set[int] = set()
        for task_index, classes in self.task_to_classes.items():
            if not classes:
                raise ValueError(f"task {task_index} has no classes.")
            for class_id in classes:
                if not 0 <= int(class_id) < self.num_classes:
                    raise ValueError(
                        f"class {class_id} of task {task_index} is outside "
                        f"[0, {self.num_classes - 1}]."
                    )
                if int(class_id) in seen_classes:
                    raise ValueError(f"class {class_id} is assigned to two tasks.")
                seen_classes.add(int(class_id))
            self.task_to_classes[task_index] = [int(value) for value in classes]

        for name in ("task_to_train_ids", "task_to_val_ids", "task_to_test_ids"):
            mapping = getattr(self, name)
            if sorted(mapping) != expected_tasks:
                raise ValueError(f"{name} must cover every task exactly once.")
            setattr(
                self,
                name,
                {
                    task: self._validate_ids(ids, f"{name}[{task}]")
                    for task, ids in mapping.items()
                },
            )

        if not self.task_to_node_ids:
            self.task_to_node_ids = {
                task: torch.unique(
                    torch.cat(
                        [
                            self.task_to_train_ids[task],
                            self.task_to_val_ids[task],
                            self.task_to_test_ids[task],
                        ]
                    )
                ).sort().values
                for task in expected_tasks
            }
        else:
            if sorted(self.task_to_node_ids) != expected_tasks:
                raise ValueError("task_to_node_ids must cover every task exactly once.")
            self.task_to_node_ids = {
                task: self._validate_ids(ids, f"task_to_node_ids[{task}]")
                for task, ids in self.task_to_node_ids.items()
            }
            for task in expected_tasks:
                labelled = set(
                    self.task_to_train_ids[task].tolist()
                    + self.task_to_val_ids[task].tolist()
                    + self.task_to_test_ids[task].tolist()
                )
                if not labelled.issubset(set(self.task_to_node_ids[task].tolist())):
                    raise ValueError(
                        f"task_to_node_ids[{task}] must contain the train/val/test ids."
                    )

        if self.task_to_train_ids[0].numel() == 0:
            raise ValueError("task 0 has no training nodes.")

    def _validate_ids(self, ids: Tensor | Sequence[int], label: str) -> Tensor:
        tensor = ids if isinstance(ids, Tensor) else torch.as_tensor(list(ids), dtype=torch.long)
        tensor = tensor.to(dtype=torch.long).reshape(-1)
        if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= self.num_nodes):
            raise ValueError(
                f"{label} contains ids outside [0, {self.num_nodes - 1}]."
            )
        return tensor.cpu()

    @property
    def num_tasks(self) -> int:
        return len(self.task_to_classes)

    @property
    def all_classes(self) -> list[int]:
        return sorted({value for values in self.task_to_classes.values() for value in values})

    def classes_seen_until(self, task_index: int) -> list[int]:
        """Classes introduced by tasks ``0..task_index`` (Class-IL label space)."""

        self._check_task(task_index)
        classes: set[int] = set()
        for index in range(task_index + 1):
            classes.update(self.task_to_classes[index])
        return sorted(classes)

    def task_edge_index(self, edge_index: Tensor, task_index: int) -> Tensor:
        """Return the edges whose two endpoints both belong to ``task_index``."""

        self._check_task(task_index)
        node_ids = self.task_to_node_ids[task_index]
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[node_ids] = True
        mask = mask.to(edge_index.device)
        keep = mask[edge_index[0]] & mask[edge_index[1]]
        return edge_index[:, keep]

    def summary(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_classes": self.num_classes,
            "num_tasks": self.num_tasks,
            "classes_per_task": sorted({len(value) for value in self.task_to_classes.values()}),
            "seed": self.seed,
            "source": self.source,
            "num_edges_removed": self.num_edges_removed,
            "task_sizes": {
                str(task): int(self.task_to_node_ids[task].numel()) for task in range(self.num_tasks)
            },
        }

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_nodes": self.num_nodes,
            "num_classes": self.num_classes,
            "num_tasks": self.num_tasks,
            "seed": self.seed,
            "source": self.source,
            "num_edges_removed": self.num_edges_removed,
            "task_to_classes": {str(key): value for key, value in self.task_to_classes.items()},
            "task_to_node_ids": {
                str(key): value.tolist() for key, value in self.task_to_node_ids.items()
            },
            "task_to_train_ids": {
                str(key): value.tolist() for key, value in self.task_to_train_ids.items()
            },
            "task_to_val_ids": {
                str(key): value.tolist() for key, value in self.task_to_val_ids.items()
            },
            "task_to_test_ids": {
                str(key): value.tolist() for key, value in self.task_to_test_ids.items()
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskSplit":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        required = ("num_nodes", "num_classes", "task_to_classes")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"{source} is missing fields: {missing}.")

        seed = int(payload.get("seed", 42))
        task_to_classes = {
            int(key): [int(value) for value in values]
            for key, values in payload["task_to_classes"].items()
        }
        tasks = sorted(task_to_classes)
        task_to_node_ids = {
            int(key): torch.as_tensor(value, dtype=torch.long)
            for key, value in (payload.get("task_to_node_ids") or {}).items()
        }

        mask_fields = ("task_to_train_ids", "task_to_val_ids", "task_to_test_ids")
        present = [name for name in mask_fields if payload.get(name)]
        if present and len(present) != len(mask_fields):
            raise ValueError(
                f"{source} must provide all of {mask_fields} or none of them."
            )
        if present:
            train_ids = _require_ids(payload, "task_to_train_ids", tasks, source)
            val_ids = _require_ids(payload, "task_to_val_ids", tasks, source)
            test_ids = _require_ids(payload, "task_to_test_ids", tasks, source)
        else:
            if not task_to_node_ids:
                raise ValueError(
                    f"{source} provides neither node-id masks nor task_to_node_ids."
                )
            absent = [task for task in tasks if task not in task_to_node_ids]
            if absent:
                raise ValueError(f"{source} is missing task_to_node_ids for {absent}.")
            train_ids, val_ids, test_ids = derive_task_masks(
                {task: task_to_node_ids[task] for task in tasks}, seed=seed
            )

        return cls(
            num_nodes=int(payload["num_nodes"]),
            num_classes=int(payload["num_classes"]),
            task_to_classes=task_to_classes,
            task_to_train_ids=train_ids,
            task_to_val_ids=val_ids,
            task_to_test_ids=test_ids,
            task_to_node_ids=task_to_node_ids,
            seed=seed,
            source=str(payload.get("source", "json")),
            num_edges_removed=int(payload.get("num_edges_removed", 0)),
        )

    def _check_task(self, task_index: int) -> None:
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise TypeError(f"task_index must be an integer, got {task_index!r}.")
        if not 0 <= task_index < self.num_tasks:
            raise IndexError(
                f"task_index={task_index} is out of range for {self.num_tasks} tasks."
            )


def _require_ids(
    payload: dict[str, Any], name: str, tasks: Sequence[int], source: Path
) -> dict[int, Tensor]:
    if name not in payload:
        raise ValueError(f"{source} is missing {name}; provide node ids or a full split.")
    mapping = payload[name]
    missing = [task for task in tasks if str(task) not in {str(key) for key in mapping}]
    if missing:
        raise ValueError(f"{name} in {source} is missing task(s): {missing}.")
    return {
        int(key): torch.as_tensor(value, dtype=torch.long) for key, value in mapping.items()
    }


def derive_task_masks(
    task_to_node_ids: Mapping[int, Sequence[int] | Tensor],
    *,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
    """Split every task's node ids into deterministic train/val/test masks.

    Used when a split only provides ``task_to_node_ids`` (member 1's JSON) so
    that all models still see an identical, seed-reproducible partition.
    """

    if train_ratio <= 0.0:
        raise ValueError(f"train_ratio must be positive, got {train_ratio}.")
    if min(val_ratio, test_ratio) < 0.0:
        raise ValueError("val_ratio and test_ratio must be non-negative.")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.")

    train_ids: dict[int, Tensor] = {}
    val_ids: dict[int, Tensor] = {}
    test_ids: dict[int, Tensor] = {}
    for task_index, ids in sorted(task_to_node_ids.items()):
        nodes = ids if isinstance(ids, Tensor) else torch.as_tensor(list(ids), dtype=torch.long)
        nodes = nodes.to(dtype=torch.long).reshape(-1)
        generator = torch.Generator().manual_seed(int(seed) * 1009 + int(task_index))
        permutation = torch.randperm(nodes.numel(), generator=generator)
        nodes = nodes.index_select(0, permutation)

        count = nodes.numel()
        num_test = max(1, int(round(count * test_ratio))) if count >= 2 else 0
        num_val = max(1, int(round(count * val_ratio))) if count >= 5 else 0
        num_train = count - num_test - num_val
        if num_train < 1:
            num_train = max(1, count - 1)
            num_test = count - num_train
            num_val = 0

        train_ids[int(task_index)] = nodes[:num_train].sort().values
        val_ids[int(task_index)] = nodes[num_train : num_train + num_val].sort().values
        test_ids[int(task_index)] = nodes[num_train + num_val :].sort().values

    return train_ids, val_ids, test_ids


def restrict_logits(logits: Tensor, allowed_classes: Sequence[int] | Tensor) -> Tensor:
    """Keep only the columns of ``allowed_classes``, preserving their order."""

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}.")
    index = _class_index(allowed_classes, logits.size(1), logits.device)
    return logits.index_select(1, index)


def restricted_cross_entropy(
    logits: Tensor, targets: Tensor, allowed_classes: Sequence[int] | Tensor
) -> Tensor:
    """Cross entropy over ``allowed_classes`` only (Class-IL / Task-IL masking)."""

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}.")
    if targets.ndim != 1 or targets.size(0) != logits.size(0):
        raise ValueError(
            f"targets must have shape [{logits.size(0)}], got {tuple(targets.shape)}."
        )
    if logits.size(0) == 0:
        raise ValueError("cannot compute a loss over zero nodes.")
    index = _class_index(allowed_classes, logits.size(1), logits.device)
    local = torch.full(
        (logits.size(1),), -1, dtype=torch.long, device=logits.device
    )
    local[index] = torch.arange(index.numel(), device=logits.device)
    mapped = local[targets.to(logits.device)]
    if bool((mapped < 0).any()):
        raise ValueError("targets contain a class outside allowed_classes.")
    return F.cross_entropy(logits.index_select(1, index), mapped)


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Mean accuracy for full-label-space logits (column index == class id)."""

    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("accuracy expects logits [N, C] and targets [N].")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"batch mismatch: {logits.size(0)} logits vs {targets.size(0)} targets."
        )
    if logits.size(0) == 0:
        return float("nan")
    predictions = logits.argmax(dim=1)
    return float((predictions == targets.to(logits.device)).float().mean())


def restricted_accuracy(
    logits: Tensor, targets: Tensor, allowed_classes: Sequence[int] | Tensor
) -> float:
    """Accuracy when predictions are limited to ``allowed_classes``.

    ``argmax`` over a restricted column subset returns a *position*, so the
    position is mapped back to the original class id before comparing with the
    labels.  Skipping that mapping silently yields zero accuracy for Task-IL.
    """

    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("restricted_accuracy expects logits [N, C] and targets [N].")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"batch mismatch: {logits.size(0)} logits vs {targets.size(0)} targets."
        )
    if logits.size(0) == 0:
        return float("nan")
    index = _class_index(allowed_classes, logits.size(1), logits.device)
    predictions = index.index_select(0, logits.index_select(1, index).argmax(dim=1))
    return float((predictions == targets.to(logits.device)).float().mean())


def _class_index(
    allowed_classes: Sequence[int] | Tensor, num_classes: int, device: torch.device
) -> Tensor:
    index = (
        allowed_classes
        if isinstance(allowed_classes, Tensor)
        else torch.as_tensor(list(allowed_classes), dtype=torch.long)
    )
    index = index.to(dtype=torch.long, device=device).reshape(-1)
    if index.numel() == 0:
        raise ValueError("allowed_classes must not be empty.")
    if int(index.min()) < 0 or int(index.max()) >= num_classes:
        raise ValueError(
            f"allowed_classes must be within [0, {num_classes - 1}], "
            f"got [{int(index.min())}, {int(index.max())}]."
        )
    return index


@runtime_checkable
class SequentialMethod(Protocol):
    """Minimal surface the unified runner needs from every continual model."""

    name: str

    def prepare(self, graph: Data, split: TaskSplit, *, seed: int) -> None:
        ...

    def fit_task(self, task_index: int) -> dict[str, Any]:
        ...

    def predict_logits(self) -> Tensor:
        ...

    def checkpoint_payload(self) -> dict[str, Any]:
        ...


class SequentialNodeClassifier(nn.Module):
    """Two-layer GCN encoder plus one linear head over the global label space.

    The head always spans ``split.num_classes`` rows.  What a task may predict is
    controlled by ``scenario``:

    * ``class-il``: logits are restricted to every class seen so far;
    * ``task-il``: logits are restricted to the current task's classes.

    Subclasses customise training by overriding :meth:`_loss_terms` (extra
    regularisers such as replay) or :meth:`fit_task` (multi-phase objectives
    such as GAE's self-supervised pre-training).
    """

    name = "sequential"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        encoder_type: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs_per_task: int = 50,
        scenario: str = "class-il",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim!r}.")
        if encoder_type not in ("gcn", "gat"):
            raise ValueError(f"encoder_type must be 'gcn' or 'gat', got {encoder_type!r}.")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if float(learning_rate) <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if float(weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")
        if isinstance(epochs_per_task, bool) or not isinstance(epochs_per_task, int) or epochs_per_task < 1:
            raise ValueError(
                f"epochs_per_task must be a positive integer, got {epochs_per_task!r}."
            )
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}.")

        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs_per_task = epochs_per_task
        self.scenario = scenario
        self.device = torch.device(device)

        self.graph: Data | None = None
        self.split: TaskSplit | None = None
        self.encoder: GNNEncoder | None = None
        self.classifier: nn.Linear | None = None
        self.seed = 42
        self.tasks_trained = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ setup
    def prepare(self, graph: Data, split: TaskSplit, *, seed: int) -> None:
        """Attach the graph/split and build encoder, head and label space."""

        if not isinstance(graph, Data):
            raise TypeError(f"graph must be a torch_geometric Data, got {type(graph)!r}.")
        if graph.x is None or graph.edge_index is None or graph.y is None:
            raise ValueError("graph must provide x, edge_index and y.")
        if graph.x.ndim != 2:
            raise ValueError(f"graph.x must be 2-D, got shape {tuple(graph.x.shape)}.")
        if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
            raise ValueError(
                f"graph.edge_index must be [2, num_edges], got {tuple(graph.edge_index.shape)}."
            )
        num_nodes = int(graph.num_nodes)
        if num_nodes != split.num_nodes:
            raise ValueError(
                f"graph has {num_nodes} nodes but the split expects {split.num_nodes}."
            )
        if graph.y.ndim != 1 or graph.y.numel() != num_nodes:
            raise ValueError("graph.y must have one label per node.")
        if int(graph.y.max()) >= split.num_classes or int(graph.y.min()) < 0:
            raise ValueError(
                f"graph.y must be in [0, {split.num_classes - 1}]."
            )

        set_seed(seed)
        self.seed = int(seed)
        self.graph = self._to_device(graph, num_nodes)
        self.split = split
        self.encoder = GNNEncoder(
            input_dim=int(graph.x.size(1)),
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            encoder_type=self.encoder_type,  # type: ignore[arg-type]
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, split.num_classes).to(self.device)
        self.tasks_trained = 0
        self.history.clear()

    def _to_device(self, graph: Data, num_nodes: int) -> Data:
        data = graph.clone()
        data.x = graph.x.to(self.device, dtype=torch.float32)
        data.edge_index = graph.edge_index.to(self.device, dtype=torch.long)
        data.y = graph.y.to(self.device, dtype=torch.long)
        if getattr(graph, "edge_weight", None) is not None:
            data.edge_weight = graph.edge_weight.to(self.device, dtype=torch.float32)
        data.num_nodes = num_nodes
        return data

    # --------------------------------------------------------------- helpers
    def allowed_classes(self, task_index: int) -> list[int]:
        """Classes this scenario can predict after ``task_index``."""

        self._require_prepared()
        if self.scenario == "class-il":
            return self.split.classes_seen_until(task_index)
        return list(self.split.task_to_classes[task_index])

    def seen_classes(self, task_index: int) -> list[int]:
        self._require_prepared()
        return self.split.classes_seen_until(task_index)

    def _require_prepared(self) -> None:
        if self.graph is None or self.split is None or self.encoder is None:
            raise RuntimeError("call prepare(graph, split, seed=...) before training.")

    def _validate_task_index(self, task_index: int) -> None:
        self._require_prepared()
        self.split._check_task(task_index)

    def _train_ids(self, task_index: int) -> Tensor:
        return self.split.task_to_train_ids[task_index].to(self.device)

    def _forward_logits(self) -> Tensor:
        self._require_prepared()
        features = self.encoder(
            self.graph.x,
            self.graph.edge_index,
            edge_weight=getattr(self.graph, "edge_weight", None),
        )
        return self.classifier(features)

    def predict_logits(self) -> Tensor:
        """Full-graph logits ``[num_nodes, num_classes]`` (evaluation mode)."""

        self._require_prepared()
        self.eval()
        with torch.no_grad():
            logits = self._forward_logits()
        return logits.detach().cpu()

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.classifier.parameters())

    # -------------------------------------------------------------- training
    def _loss_terms(
        self, logits: Tensor, task_index: int, allowed: Sequence[int]
    ) -> tuple[Tensor, dict[str, float]]:
        """Loss on the current task; override to add regularisers."""

        ids = self._train_ids(task_index)
        task_loss = restricted_cross_entropy(
            logits.index_select(0, ids),
            self.graph.y.index_select(0, ids),
            allowed,
        )
        return task_loss, {"task_loss": float(task_loss.detach())}

    def fit_task(self, task_index: int) -> dict[str, Any]:
        """Train on one task for ``epochs_per_task`` epochs."""

        self._validate_task_index(task_index)
        allowed = self.allowed_classes(task_index)
        optimizer = torch.optim.Adam(
            self._trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train()
        losses: list[float] = []
        extras: dict[str, float] = {}
        for _ in range(self.epochs_per_task):
            optimizer.zero_grad(set_to_none=True)
            logits = self._forward_logits()
            loss, step_extras = self._loss_terms(logits, task_index, allowed)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            extras = step_extras

        train_accuracy = self._train_accuracy(task_index, allowed)
        info: dict[str, Any] = {
            "task": task_index,
            "epochs": self.epochs_per_task,
            "final_loss": losses[-1] if losses else float("nan"),
            "mean_loss": float(sum(losses) / len(losses)) if losses else float("nan"),
            "train_accuracy": train_accuracy,
            "allowed_classes": list(allowed),
            **extras,
        }
        self.tasks_trained = max(self.tasks_trained, task_index + 1)
        self.history.append(info)
        return info

    def _train_accuracy(self, task_index: int, allowed: Sequence[int]) -> float:
        ids = self._train_ids(task_index)
        self.eval()
        with torch.no_grad():
            logits = self._forward_logits().index_select(0, ids)
            targets = self.graph.y.index_select(0, ids)
        return restricted_accuracy(logits, targets, allowed)

    # ------------------------------------------------------------ artifacts
    def checkpoint_payload(self) -> dict[str, Any]:
        self._require_prepared()
        return {
            "model": self.name,
            "scenario": self.scenario,
            "hidden_dim": self.hidden_dim,
            "encoder_type": self.encoder_type,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs_per_task": self.epochs_per_task,
            "seed": self.seed,
            "tasks_trained": self.tasks_trained,
            "num_classes": self.split.num_classes,
            "input_dim": int(self.graph.x.size(1)),
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "history": self.history,
        }

    def extra_repr(self) -> str:
        return (
            f"scenario={self.scenario}, hidden_dim={self.hidden_dim}, "
            f"epochs_per_task={self.epochs_per_task}, lr={self.learning_rate}"
        )
