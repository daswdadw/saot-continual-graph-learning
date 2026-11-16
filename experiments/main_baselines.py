"""Unified baseline runner and evaluator (member 5).

Runs every registered continual model through the *same* protocol, the same data
split, the same metrics and the same output layout::

    python experiments/main_baselines.py --config configs/local.yaml \
        --dataset CoraFull-CL --seeds 42 43 44

Inputs
------
* ``--config`` YAML with the shared keys of ``configs/example.yaml``.  When the
  file is missing, ``configs/example.yaml`` is used and a warning is printed, so
  the acceptance command works on a fresh clone (``configs/local.yaml`` is
  git-ignored and therefore usually absent).
* ``--dataset`` name, e.g. ``CoraFull-CL``.  Data is resolved by a three-level
  fallback: member 1's ``data/task_splits/<slug>_seed<seed>.json`` (plus a
  processed graph), then ``torch_geometric.datasets`` auto-download, then a
  deterministic synthetic graph (``--synthetic`` forces this level).

Outputs (per run, matching the repository convention)
-----------------------------------------------------
::

    results/<model>/<dataset>/<scenario>/seed_<seed>/
        config.yaml                 # resolved settings + protocol + split info
        metrics.json                # AP/AF for both scenarios, per-task accuracy
        performance_matrix.csv      # T x T matrix of the run scenario
        performance_matrix_<other>.csv
        train.log
        checkpoint.pt
        performance_matrix.png      # heatmap (best effort)

    results/tables/main_table.csv   # mean +- std over seeds (percent)
    results/figures/member_5/main_ap_af.png
    logs/member_5/main_baselines/<model>_<dataset>_<scenario>_seed<seed>.log

Model contract for other members
--------------------------------
Any class registered in ``METHOD_REGISTRY`` must implement
:class:`evaluation.protocol.SequentialMethod` (``prepare`` / ``fit_task`` /
``predict_logits`` / ``checkpoint_payload``) and accept these keyword arguments:
``hidden_dim, encoder_type, num_layers, dropout, learning_rate, weight_decay,
epochs_per_task, scenario, device``.  A module-level ``build_method(**kwargs)``
factory is preferred.  That is all member 4 needs to plug SAOT in.

Memory
------
Full-graph training: CoraFull-CL (19,793 nodes, hidden 512) needs about 1-2 GB
of GPU memory per run; the synthetic fallback needs well under 1 GB.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch_geometric.data import Data

from evaluation.metrics import (
    PerformanceMatrix,
    aggregate_seeds,
    plot_main_ap_af,
    plot_matrix_heatmap,
    write_rows_csv,
)
from evaluation.protocol import (
    SCENARIOS,
    TaskSplit,
    derive_task_masks,
    restricted_accuracy,
)

__all__ = [
    "METHOD_REGISTRY",
    "PROTOCOL",
    "DatasetBundle",
    "build_synthetic_graph",
    "collect_runs",
    "evaluate_predictions",
    "load_dataset",
    "main",
    "run_single",
    "split_continual_tasks",
]

DEFAULT_CONFIG_PATH = "configs/example.yaml"

#: Unified settings shared by every baseline and by SAOT.
PROTOCOL: dict[str, Any] = {
    "classes_per_task": 2,
    "epochs_per_task": 50,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "hidden_dim": 512,
    "encoder_type": "gcn",
    "num_layers": 2,
    "dropout": 0.0,
    "replay_budget": 200,
    "seeds": [42, 43, 44],
    "scenarios": ["class-il", "task-il"],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
}

#: Paper values, used only to warn when a resolved run deviates from them.
PAPER_REFERENCE = {
    "hidden_dim": 512,
    "epochs_per_task": 50,
    "learning_rate": 1e-3,
    "replay_budget": 200,
}

METHOD_REGISTRY: dict[str, tuple[str, str]] = {
    "gcn_sequential": ("baselines.gcn_sequential", "GCNSequentialMethod"),
    "gae": ("baselines.gae_or_dgi", "GAEMethod"),
    "replay": ("baselines.replay_baseline", "ReplayMethod"),
    "saot": ("models.saot", "SAOTMethod"),
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None) -> tuple[dict[str, Any], Path]:
    """Load a YAML config, falling back to ``configs/example.yaml``."""

    requested = Path(path) if path else Path(DEFAULT_CONFIG_PATH)
    resolved = requested if requested.is_absolute() else PROJECT_ROOT / requested
    if not resolved.exists():
        fallback = PROJECT_ROOT / DEFAULT_CONFIG_PATH
        print(
            f"[warn] config {requested} not found; using {DEFAULT_CONFIG_PATH} instead. "
            "Copy configs/example.yaml to configs/local.yaml to customise a run."
        )
        resolved = fallback
    if not resolved.exists():
        raise FileNotFoundError(
            f"no config file found at {requested} or {DEFAULT_CONFIG_PATH}"
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping.")
    return payload, resolved


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int_list(value: Any) -> list[int]:
    return [int(item) for item in _as_list(value)]


def resolve_settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI > config file > :data:`PROTOCOL`."""

    def pick(cli_value: Any, config_key: str, protocol_key: str, default: Any = None) -> Any:
        if cli_value is not None:
            return cli_value
        if config_key in config and config[config_key] is not None:
            return config[config_key]
        if protocol_key in PROTOCOL:
            return PROTOCOL[protocol_key]
        return default

    scenarios = args.scenarios or _as_list(
        config.get("scenarios", config.get("scenario", PROTOCOL["scenarios"]))
    )
    scenarios = [str(item) for item in scenarios]
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario {scenario!r} must be one of {SCENARIOS}.")

    models = args.models or _as_list(config.get("models", list(METHOD_REGISTRY)))
    device = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "dataset": args.dataset or config.get("dataset", "CoraFull-CL"),
        "seeds": _as_int_list(args.seeds or config.get("seeds", PROTOCOL["seeds"])),
        "scenarios": scenarios,
        "models": [str(name) for name in models],
        "classes_per_task": int(pick(args.classes_per_task, "classes_per_task", "classes_per_task")),
        "epochs_per_task": int(pick(args.epochs_per_task, "epochs_per_task", "epochs_per_task")),
        "learning_rate": float(pick(args.learning_rate, "learning_rate", "learning_rate")),
        "weight_decay": float(pick(args.weight_decay, "weight_decay", "weight_decay")),
        "hidden_dim": int(pick(args.hidden_dim, "hidden_dim", "hidden_dim")),
        "encoder_type": str(pick(None, "encoder_type", "encoder_type")),
        "num_layers": int(pick(None, "num_layers", "num_layers")),
        "dropout": float(pick(None, "dropout", "dropout")),
        "replay_budget": int(pick(args.replay_budget, "replay_budget", "replay_budget")),
        "train_ratio": float(PROTOCOL["train_ratio"]),
        "val_ratio": float(PROTOCOL["val_ratio"]),
        "test_ratio": float(PROTOCOL["test_ratio"]),
        "device": str(device),
        "output_root": str(args.output_root or config.get("output_root", "results")),
        "log_dir": str(args.log_dir or config.get("log_dir", "logs/member_5/main_baselines")),
        "data_source": str(args.data_source or config.get("data_source", "synthetic" if args.synthetic else "auto")),
    }


def protocol_record(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of settings that must be identical for all models."""

    return {
        "dataset": settings["dataset"],
        "seeds": list(settings["seeds"]),
        "scenarios": list(settings["scenarios"]),
        "classes_per_task": settings["classes_per_task"],
        "epochs_per_task": settings["epochs_per_task"],
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "hidden_dim": settings["hidden_dim"],
        "encoder_type": settings["encoder_type"],
        "num_layers": settings["num_layers"],
        "dropout": settings["dropout"],
        "replay_budget": settings["replay_budget"],
        "split_ratios": {
            "train": settings["train_ratio"],
            "val": settings["val_ratio"],
            "test": settings["test_ratio"],
        },
    }


def protocol_deviations(record: Mapping[str, Any]) -> list[str]:
    """Differences from the paper protocol, printed into the run log."""

    notes = []
    for key, paper_value in PAPER_REFERENCE.items():
        value = record.get(key)
        if value is not None and not math.isclose(float(value), float(paper_value)):
            notes.append(f"{key}={value} (paper: {paper_value})")
    return notes


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    """A global graph plus its continual task split."""

    dataset: str
    graph: Data
    split: TaskSplit
    source: str
    num_edges: int
    num_edges_removed: int

    def info(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "num_nodes": int(self.graph.num_nodes),
            "num_edges": self.num_edges,
            "num_edges_removed": self.num_edges_removed,
            "input_dim": int(self.graph.x.size(1)),
            "num_classes": self.split.num_classes,
            "num_tasks": self.split.num_tasks,
            "seed": self.split.seed,
        }


def dataset_slug(dataset: str) -> str:
    """``CoraFull-CL`` -> ``corafull`` (member 1's JSON naming)."""

    base = str(dataset).split("-")[0]
    return "".join(character for character in base.lower() if character.isalnum())


def build_synthetic_graph(
    *,
    num_classes: int = 70,
    nodes_per_class: int = 8,
    feature_dim: int = 32,
    seed: int = 42,
) -> Data:
    """Deterministic class-block graph with no inter-class edges."""

    if num_classes < 1 or nodes_per_class < 2 or feature_dim < 1:
        raise ValueError("num_classes, feature_dim must be positive and nodes_per_class >= 2.")
    generator = torch.Generator().manual_seed(seed)
    num_nodes = num_classes * nodes_per_class
    labels = torch.arange(num_classes).repeat_interleave(nodes_per_class)
    centres = torch.randn(num_classes, feature_dim, generator=generator)
    features = centres.index_select(0, labels) + 0.6 * torch.randn(
        num_nodes, feature_dim, generator=generator
    )

    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for class_id in range(num_classes):
        start = class_id * nodes_per_class
        nodes = torch.arange(start, start + nodes_per_class)
        sources.append(nodes)
        targets.append(torch.roll(nodes, shifts=-1))
        sources.append(nodes)
        targets.append(
            nodes.index_select(
                0, torch.randint(0, nodes_per_class, (nodes_per_class,), generator=generator)
            )
        )
    forward = torch.stack([torch.cat(sources), torch.cat(targets)])
    edge_index = torch.cat([forward, forward.flip(0)], dim=1)
    return Data(x=features, y=labels, edge_index=edge_index, num_nodes=num_nodes)


def split_continual_tasks(
    graph: Data,
    *,
    dataset: str,
    seed: int,
    classes_per_task: int = 2,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    source: str = "graph",
) -> DatasetBundle:
    """Build a task split and drop inter-task edges.

    Classes are sorted ascending and grouped into consecutive pairs (the paper's
    protocol: two classes per task, fixed order, no inter-task edges).
    """

    if graph.y is None:
        raise ValueError("graph.y is required to build the task split.")
    labels = graph.y.view(-1).to(dtype=torch.long)
    classes = sorted(torch.unique(labels).tolist())
    if classes_per_task < 1 or len(classes) % classes_per_task != 0:
        raise ValueError(
            f"{len(classes)} classes cannot be split into tasks of {classes_per_task} classes."
        )

    num_tasks = len(classes) // classes_per_task
    task_to_classes = {
        task: classes[task * classes_per_task : (task + 1) * classes_per_task]
        for task in range(num_tasks)
    }
    class_to_task = {
        class_id: task for task, values in task_to_classes.items() for class_id in values
    }
    node_task = torch.tensor([class_to_task[int(label)] for label in labels], dtype=torch.long)

    edge_index = graph.edge_index.to(dtype=torch.long)
    if edge_index.size(1):
        keep = node_task.index_select(0, edge_index[0]) == node_task.index_select(0, edge_index[1])
        filtered = edge_index[:, keep]
    else:
        keep = torch.zeros(0, dtype=torch.bool)
        filtered = edge_index
    removed = int(edge_index.size(1) - filtered.size(1))

    task_to_node_ids = {
        task: (node_task == task).nonzero(as_tuple=True)[0] for task in range(num_tasks)
    }
    train_ids, val_ids, test_ids = derive_task_masks(
        task_to_node_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    new_graph = Data(x=graph.x, edge_index=filtered, y=labels, num_nodes=int(graph.num_nodes))
    if getattr(graph, "edge_weight", None) is not None and graph.edge_weight.numel() == edge_index.size(1):
        new_graph.edge_weight = graph.edge_weight[keep]

    split = TaskSplit(
        num_nodes=int(graph.num_nodes),
        num_classes=max(classes) + 1,
        task_to_classes=task_to_classes,
        task_to_train_ids=train_ids,
        task_to_val_ids=val_ids,
        task_to_test_ids=test_ids,
        task_to_node_ids=task_to_node_ids,
        seed=seed,
        source=source,
        num_edges_removed=removed,
    )
    return DatasetBundle(
        dataset=dataset,
        graph=new_graph,
        split=split,
        source=source,
        num_edges=int(filtered.size(1)),
        num_edges_removed=removed,
    )


def _coerce_graph(payload: Any) -> Data:
    if isinstance(payload, Data):
        return payload
    if isinstance(payload, Mapping):
        for key in ("graph", "data", "dataset"):
            if key in payload:
                return _coerce_graph(payload[key])
    raise TypeError(f"cannot interpret {type(payload)!r} as a torch_geometric Data.")


def _load_member1_bundle(
    dataset: str,
    seed: int,
    *,
    split_dir: Path,
    processed_dir: Path,
) -> DatasetBundle | None:
    slug = dataset_slug(dataset)
    candidates = sorted(split_dir.glob(f"{slug}_seed{seed}*.json"))
    candidates += sorted(split_dir.glob(f"{slug}_*{seed}*.json"))
    split_path = next((path for path in candidates if path.exists()), None)
    if split_path is None:
        return None

    graph = None
    for pattern in (f"{slug}.pt", f"{slug}_graph.pt", f"{slug}*.pt"):
        for candidate in sorted(processed_dir.glob(pattern)):
            try:
                graph = _coerce_graph(torch.load(candidate, map_location="cpu", weights_only=False))
            except Exception as error:  # pragma: no cover - defensive
                print(f"[warn] could not read {candidate}: {error}")
                continue
            break
        if graph is not None:
            break

    if graph is None:
        try:
            module = importlib.import_module("data.prepare_cgl")
        except Exception:
            module = None
        for name in ("load_graph", "load_dataset", "build_graph"):
            loader = getattr(module, name, None) if module is not None else None
            if loader is None:
                continue
            for kwargs in ({"dataset": dataset, "seed": seed}, {}):
                try:
                    graph = _coerce_graph(loader(**kwargs))
                    break
                except Exception:
                    graph = None
            if graph is not None:
                break

    if graph is None:
        print(
            f"[warn] found {split_path} but no processed graph for {dataset}; "
            "falling back to the next data source."
        )
        return None

    split = TaskSplit.from_json(split_path)
    if int(graph.num_nodes) != split.num_nodes:
        raise ValueError(
            f"graph has {int(graph.num_nodes)} nodes but {split_path} expects {split.num_nodes}."
        )
    if graph.y is None:
        raise ValueError("member 1's processed graph must provide node labels in graph.y.")

    edge_index = graph.edge_index.to(dtype=torch.long)
    node_task = torch.full((int(graph.num_nodes),), -1, dtype=torch.long)
    for task, nodes in split.task_to_node_ids.items():
        node_task[nodes] = task
    if edge_index.size(1) and node_task.min() >= 0:
        keep = node_task.index_select(0, edge_index[0]) == node_task.index_select(0, edge_index[1])
        removed = int(edge_index.size(1) - int(keep.sum()))
        graph = graph.clone()
        graph.edge_index = edge_index[:, keep]
    else:
        removed = 0

    return DatasetBundle(
        dataset=dataset,
        graph=graph,
        split=split,
        source="member1",
        num_edges=int(graph.edge_index.size(1)),
        num_edges_removed=removed,
    )


def _load_pyg_bundle(
    dataset: str,
    seed: int,
    *,
    settings: Mapping[str, Any],
    raw_root: Path,
    allow_download: bool,
) -> DatasetBundle | None:
    try:
        from torch_geometric.datasets import CoraFull
    except Exception as error:
        print(f"[warn] PyG datasets unavailable: {error}")
        return None

    slug = dataset_slug(dataset)
    if slug != "corafull":
        print(
            f"[warn] automatic download is only wired for CoraFull (requested {dataset}); "
            "provide member 1's split for other datasets."
        )
        return None

    root = raw_root / "CoraFull"
    if not allow_download and not root.exists():
        print(f"[warn] {root} is missing and --no-download is set.")
        return None
    try:
        raw = CoraFull(root=str(root))[0]
    except Exception as error:
        print(f"[warn] CoraFull download failed: {error}")
        return None

    data = Data(
        x=raw.x.to(dtype=torch.float32),
        edge_index=raw.edge_index.to(dtype=torch.long),
        y=raw.y.view(-1).to(dtype=torch.long),
        num_nodes=int(raw.num_nodes),
    )
    return split_continual_tasks(
        data,
        dataset=dataset,
        seed=seed,
        classes_per_task=int(settings["classes_per_task"]),
        train_ratio=float(settings["train_ratio"]),
        val_ratio=float(settings["val_ratio"]),
        test_ratio=float(settings["test_ratio"]),
        source="pyg",
    )


def _synthetic_bundle(dataset: str, seed: int, *, settings: Mapping[str, Any], args: argparse.Namespace) -> DatasetBundle:
    num_classes = int(args.synthetic_classes or settings.get("synthetic_classes") or 70)
    nodes_per_class = int(args.synthetic_nodes_per_class or 8)
    feature_dim = int(args.synthetic_feature_dim or 32)
    graph = build_synthetic_graph(
        num_classes=num_classes,
        nodes_per_class=nodes_per_class,
        feature_dim=feature_dim,
        seed=seed,
    )
    return split_continual_tasks(
        graph,
        dataset=f"{dataset}(synthetic)",
        seed=seed,
        classes_per_task=int(settings["classes_per_task"]),
        train_ratio=float(settings["train_ratio"]),
        val_ratio=float(settings["val_ratio"]),
        test_ratio=float(settings["test_ratio"]),
        source="synthetic",
    )


def load_dataset(
    dataset: str,
    seed: int,
    *,
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> DatasetBundle:
    """Resolve the dataset: member 1 -> PyG download -> synthetic."""

    requested = str(settings.get("data_source", "auto"))
    if requested == "synthetic":
        return _synthetic_bundle(dataset, seed, settings=settings, args=args)

    if requested in {"auto", "member1"}:
        bundle = _load_member1_bundle(
            dataset,
            seed,
            split_dir=PROJECT_ROOT / "data" / "task_splits",
            processed_dir=PROJECT_ROOT / "data" / "processed",
        )
        if bundle is not None:
            print(f"[data] {dataset} seed {seed}: member 1 split ({bundle.info()})")
            return bundle
        if requested == "member1":
            raise FileNotFoundError(
                "member 1's split/graph is not available; run without --data-source member1 "
                "to allow the PyG or synthetic fallback."
            )

    if requested in {"auto", "pyg"}:
        bundle = _load_pyg_bundle(
            dataset,
            seed,
            settings=settings,
            raw_root=PROJECT_ROOT / "data" / "raw",
            allow_download=not args.no_download,
        )
        if bundle is not None:
            cache_dir = PROJECT_ROOT / "data" / "processed" / "generated"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{dataset_slug(dataset)}_seed{seed}.json"
            bundle.split.to_json(cache_path)
            print(f"[data] {dataset} seed {seed}: PyG auto-download, split cached at {cache_path}")
            return bundle
        if requested == "pyg":
            raise RuntimeError("PyG data source requested but unavailable.")

    print(f"[warn] falling back to a synthetic graph for {dataset}; results are smoke-test only.")
    return _synthetic_bundle(dataset, seed, settings=settings, args=args)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    logits: torch.Tensor,
    graph: Data,
    split: TaskSplit,
    trained_until: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Accuracy on every evaluated task, under Class-IL and Task-IL rules."""

    logits = logits.detach().cpu()
    labels = graph.y.detach().cpu()
    seen = split.classes_seen_until(trained_until)
    class_il: dict[int, float] = {}
    task_il: dict[int, float] = {}
    for task in range(trained_until + 1):
        nodes = split.task_to_test_ids[task]
        if nodes.numel() == 0:
            continue
        selected = logits.index_select(0, nodes)
        targets = labels.index_select(0, nodes)
        class_il[task] = restricted_accuracy(selected, targets, seen)
        task_il[task] = restricted_accuracy(
            selected, targets, split.task_to_classes[task]
        )
    return class_il, task_il


class TeeLog:
    """Write the same progress lines to several files and stdout."""

    def __init__(self, paths: Sequence[Path], *, echo: bool = True) -> None:
        self._handles = []
        self._echo = echo
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles.append(path.open("w", encoding="utf-8"))

    def write(self, message: str) -> None:
        text = message.rstrip("\n")
        for handle in self._handles:
            handle.write(text + "\n")
            handle.flush()
        if self._echo:
            print(text)

    def close(self) -> None:
        for handle in self._handles:
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "TeeLog":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def run_single(
    *,
    method_name: str,
    method: Any,
    bundle: DatasetBundle,
    scenario: str,
    seed: int,
    settings: Mapping[str, Any],
    output_root: Path,
    log_root: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one model on one (dataset, scenario, seed) and write its artifacts."""

    run_dir = run_dir or (
        output_root / method_name / bundle.dataset / scenario / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [
        run_dir / "train.log",
        log_root / f"{method_name}_{bundle.dataset}_{scenario}_seed{seed}.log",
    ]
    matrices = {name: PerformanceMatrix(bundle.split.num_tasks) for name in SCENARIOS}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    with TeeLog(log_paths) as log:
        log.write(f"# model={method_name} dataset={bundle.dataset} scenario={scenario} seed={seed}")
        log.write(f"# data={json.dumps(bundle.info(), ensure_ascii=False)}")
        log.write(f"# protocol={json.dumps(protocol_record(settings), ensure_ascii=False)}")
        notes = protocol_deviations(protocol_record(settings))
        if notes:
            log.write(f"# deviates from paper protocol: {', '.join(notes)}")

        try:
            method.prepare(bundle.graph, bundle.split, seed=seed)
            for task in range(bundle.split.num_tasks):
                info = method.fit_task(task)
                logits = method.predict_logits()
                class_il, task_il = evaluate_predictions(
                    logits, bundle.graph, bundle.split, task
                )
                for index, value in class_il.items():
                    matrices["class-il"].update(task, index, value)
                for index, value in task_il.items():
                    matrices["task-il"].update(task, index, value)

                run_summary = matrices[scenario].summary()
                record = {
                    "task": task,
                    "train_loss": float(info.get("final_loss", float("nan"))),
                    "train_accuracy": float(info.get("train_accuracy", float("nan"))),
                    "ap": run_summary.ap,
                    "af": run_summary.af,
                    "class_il_ap": matrices["class-il"].summary().ap,
                    "task_il_ap": matrices["task-il"].summary().ap,
                }
                history.append(record)
                log.write(
                    f"[task {task + 1:02d}/{bundle.split.num_tasks}] "
                    f"loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.4f} "
                    f"AP={record['ap'] * 100:.2f} AF={record['af'] * 100:.2f} "
                    f"(class-il AP={record['class_il_ap'] * 100:.2f}, "
                    f"task-il AP={record['task_il_ap'] * 100:.2f})"
                )
                torch.save(method.checkpoint_payload(), run_dir / "checkpoint.pt")
        except Exception:
            log.write(traceback.format_exc())
            raise

        runtime = time.perf_counter() - started
        metrics = {
            "model": method_name,
            "dataset": bundle.dataset,
            "scenario": scenario,
            "seed": seed,
            "num_tasks": bundle.split.num_tasks,
            "runtime_sec": runtime,
            "data": bundle.info(),
            "protocol": protocol_record(settings),
            "history": history,
            "scenarios": {
                name: matrices[name].summary(prefix=False).to_dict() for name in SCENARIOS
            },
        }
        run_summary = matrices[scenario].summary(prefix=False)
        metrics.update(
            {
                "ap": run_summary.ap,
                "af": run_summary.af,
                "ap_percent": run_summary.ap * 100.0,
                "af_percent": run_summary.af * 100.0,
                "per_task_final_accuracy": metrics["scenarios"][scenario][
                    "per_task_final_accuracy"
                ],
            }
        )

        run_config = {
            **protocol_record(settings),
            "model": method_name,
            "scenario": scenario,
            "seed": seed,
            "device": settings["device"],
            "data": bundle.info(),
            "protocol_deviations": notes,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

        for name in SCENARIOS:
            path = (
                run_dir / "performance_matrix.csv"
                if name == scenario
                else run_dir / f"performance_matrix_{name.replace('-', '_')}.csv"
            )
            matrices[name].to_csv(path)

        plot_matrix_heatmap(
            matrices[scenario].to_numpy(),
            run_dir / "performance_matrix.png",
            title=f"{method_name} — {bundle.dataset} — {scenario} — seed {seed}",
        )

        log.write(
            f"# done ap={metrics['ap_percent']:.2f}% af={metrics['af_percent']:.2f}% "
            f"runtime={runtime:.1f}s artifacts={run_dir}"
        )

    return {
        "model": method_name,
        "dataset": bundle.dataset,
        "scenario": scenario,
        "seed": seed,
        "ap": metrics["ap"],
        "af": metrics["af"],
        "ap_percent": metrics["ap_percent"],
        "af_percent": metrics["af_percent"],
        "num_tasks": bundle.split.num_tasks,
        "runtime_sec": runtime,
        "source": bundle.source,
        "run_dir": str(run_dir),
        "status": "ok",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"{type(value)!r} is not JSON serialisable")


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
MAIN_TABLE_FIELDS = [
    "model",
    "dataset",
    "scenario",
    "num_seeds",
    "seeds",
    "ap_mean",
    "ap_std",
    "af_mean",
    "af_std",
]


def collect_runs(output_root: str | Path) -> list[dict[str, Any]]:
    """Rebuild per-seed rows from ``results/**/metrics.json``."""

    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/*/*/seed_*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as error:  # pragma: no cover - corrupt artifact
            print(f"[warn] skipping {metrics_path}: {error}")
            continue
        rows.append(
            {
                "model": payload.get("model", metrics_path.parts[-4]),
                "dataset": payload.get("dataset", metrics_path.parts[-3]),
                "scenario": payload.get("scenario", metrics_path.parts[-2]),
                "seed": int(payload.get("seed", 0)),
                "ap": payload.get("ap", float("nan")),
                "af": payload.get("af", float("nan")),
                "ap_percent": payload.get("ap_percent", float("nan")),
                "af_percent": payload.get("af_percent", float("nan")),
                "num_tasks": payload.get("num_tasks", 0),
                "run_dir": str(metrics_path.parent),
                "status": "ok",
            }
        )
    return rows


def write_main_table(rows: Sequence[Mapping[str, Any]], path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Aggregate seeds and write ``main_table.csv`` (percent mean/std)."""

    aggregated = aggregate_seeds(rows)
    target = write_rows_csv(path, aggregated, fieldnames=MAIN_TABLE_FIELDS)
    return target, aggregated


def make_figures(
    aggregated: Sequence[Mapping[str, Any]],
    *,
    figures_root: str | Path,
    dataset: str,
) -> Path | None:
    figures = Path(figures_root)
    figures.mkdir(parents=True, exist_ok=True)
    return plot_main_ap_af(
        aggregated,
        figures / "main_ap_af.png",
        title=f"{dataset}: AP / AF (mean +- std over seeds)",
    )


def verify_runs(run_dirs: Iterable[str | Path]) -> list[str]:
    """Recompute AP/AF from each ``performance_matrix.csv`` and compare."""

    problems: list[str] = []
    for run_dir in run_dirs:
        directory = Path(run_dir)
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            problems.append(f"{directory}: metrics.json missing")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for scenario in SCENARIOS:
            path = (
                directory / "performance_matrix.csv"
                if scenario == metrics.get("scenario")
                else directory / f"performance_matrix_{scenario.replace('-', '_')}.csv"
            )
            if not path.exists():
                problems.append(f"{directory}: {path.name} missing")
                continue
            recomputed = PerformanceMatrix.from_csv(path).summary(prefix=False).to_dict()
            stored = metrics.get("scenarios", {}).get(scenario)
            if stored is None:
                problems.append(f"{directory}: metrics.json lacks scenario {scenario}")
                continue
            for key in ("ap", "af"):
                left, right = recomputed[key], stored.get(key)
                if right is None or float("nan") in (left, right):
                    continue
                if abs(left - float(right)) > 1e-6:
                    problems.append(
                        f"{directory} [{scenario}]: {key} mismatch "
                        f"(csv={left:.6f}, json={float(right):.6f})"
                    )
    return problems


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def method_kwargs(method_name: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "hidden_dim": settings["hidden_dim"],
        "encoder_type": settings["encoder_type"],
        "num_layers": settings["num_layers"],
        "dropout": settings["dropout"],
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "epochs_per_task": settings["epochs_per_task"],
        "scenario": settings["scenario"],
        "device": settings["device"],
    }
    if method_name == "replay":
        kwargs["replay_budget"] = settings["replay_budget"]
    return kwargs


def resolve_models(names: Sequence[str]) -> dict[str, Any]:
    """Resolve model factories, skipping optional ones that are not available."""

    factories: dict[str, Any] = {}
    for name in names:
        entry = METHOD_REGISTRY.get(name)
        if entry is None:
            raise SystemExit(
                f"[error] unknown model {name!r}; choose from {sorted(METHOD_REGISTRY)}."
            )
        module_name, attribute = entry
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            print(
                f"[skip] {name}: cannot import {module_name} "
                f"({error.__class__.__name__}: {error})"
            )
            continue
        factory = getattr(module, "build_method", None) or getattr(module, attribute, None)
        if factory is None:
            print(f"[skip] {name}: {module_name} has neither build_method nor {attribute}.")
            continue
        factories[name] = factory
    return factories


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path")
    parser.add_argument("--dataset", default=None, help="dataset name, e.g. CoraFull-CL")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="random seeds")
    parser.add_argument("--models", nargs="+", default=None, help=f"subset of {sorted(METHOD_REGISTRY)}")
    parser.add_argument("--scenarios", nargs="+", default=None, choices=list(SCENARIOS))
    parser.add_argument("--classes-per-task", type=int, default=None)
    parser.add_argument("--epochs-per-task", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--replay-budget", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:1")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--data-source",
        default=None,
        choices=["auto", "member1", "pyg", "synthetic"],
        help="force one data source instead of the member1 -> pyg -> synthetic fallback",
    )
    parser.add_argument("--synthetic", action="store_true", help="force the synthetic graph")
    parser.add_argument("--synthetic-classes", type=int, default=None)
    parser.add_argument("--synthetic-nodes-per-class", type=int, default=None)
    parser.add_argument("--synthetic-feature-dim", type=int, default=None)
    parser.add_argument("--no-download", action="store_true", help="never download datasets")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic run for plumbing checks")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild main_table.csv and figures from existing results",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute AP/AF from performance_matrix.csv and compare with metrics.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.synthetic = True
        args.synthetic_classes = args.synthetic_classes or 6
        args.synthetic_nodes_per_class = args.synthetic_nodes_per_class or 10
        args.synthetic_feature_dim = args.synthetic_feature_dim or 16
        args.epochs_per_task = args.epochs_per_task or 2
        args.hidden_dim = args.hidden_dim or 16
        args.models = args.models or ["gcn_sequential", "gae", "replay"]

    config, config_path = load_config(args.config)
    settings = resolve_settings(config, args)

    output_root = PROJECT_ROOT / settings["output_root"] if not Path(settings["output_root"]).is_absolute() else Path(settings["output_root"])
    log_root = PROJECT_ROOT / settings["log_dir"] if not Path(settings["log_dir"]).is_absolute() else Path(settings["log_dir"])
    tables_dir = output_root / "tables"
    figures_root = output_root / "figures" / "member_5"

    print(f"[config] {config_path}")
    print(f"[settings] {json.dumps(protocol_record(settings), ensure_ascii=False)}")

    if args.aggregate_only:
        rows = collect_runs(output_root)
        if not rows:
            print(f"[error] no runs found under {output_root}")
            return 1
        table_path, aggregated = write_main_table(rows, tables_dir / "main_table.csv")
        figure = make_figures(aggregated, figures_root=figures_root, dataset=settings["dataset"])
        print(f"[done] {table_path}")
        if figure is not None:
            print(f"[done] {figure}")
        return 0

    factories = resolve_models(settings["models"])
    if not factories:
        print("[error] no runnable model; install torch/PyG or implement a baseline first.")
        return 1

    run_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for seed in settings["seeds"]:
        try:
            bundle = load_dataset(settings["dataset"], seed, settings=settings, args=args)
        except Exception as error:
            print(f"[error] dataset unavailable for seed {seed}: {error}")
            return 1

        for scenario in settings["scenarios"]:
            scenario_settings = dict(settings)
            scenario_settings["scenario"] = scenario
            for method_name, factory in factories.items():
                print(f"[run] {method_name} | {bundle.dataset} | {scenario} | seed {seed}")
                try:
                    method = factory(**method_kwargs(method_name, scenario_settings))
                    row = run_single(
                        method_name=method_name,
                        method=method,
                        bundle=bundle,
                        scenario=scenario,
                        seed=seed,
                        settings=scenario_settings,
                        output_root=output_root,
                        log_root=log_root,
                    )
                    run_rows.append(row)
                except Exception as error:
                    message = f"{method_name} | {scenario} | seed {seed}: {error}"
                    failures.append(message)
                    print(f"[fail] {message}")
                    failure_log = log_root / f"{method_name}_{bundle.dataset}_{scenario}_seed{seed}.log"
                    failure_log.parent.mkdir(parents=True, exist_ok=True)
                    with failure_log.open("a", encoding="utf-8") as handle:
                        handle.write(traceback.format_exc())

    if not run_rows:
        print("[error] every run failed; see the logs above.")
        return 1

    all_rows = run_rows
    table_path, aggregated = write_main_table(all_rows, tables_dir / "main_table.csv")
    figure = make_figures(aggregated, figures_root=figures_root, dataset=settings["dataset"])
    print(f"[done] {table_path}")
    if figure is not None:
        print(f"[done] {figure}")

    if args.verify:
        problems = verify_runs(row["run_dir"] for row in all_rows)
        if problems:
            print("[verify] mismatches found:")
            for problem in problems:
                print(f"  - {problem}")
            failures.extend(problems)
        else:
            print("[verify] AP/AF recomputed from CSV match metrics.json for all runs.")

    if failures:
        print(f"[warn] {len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
