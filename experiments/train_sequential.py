"""Train SAOT sequentially on graph tasks (paper Section 4.2).

The task file format is a torch file containing ``tasks``. Each task is a
dictionary with ``x`` and either dense ``adjacency`` or ``edge_index``. The
``--smoke-test`` mode creates tiny graph tasks so the complete algorithm can be
checked before the CoraFull-CL preparation pipeline is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.structural_distillation import (
    cross_task_plan_distillation_loss,
    saot_total_loss,
    structural_calibration_loss,
)
from losses.transport_alignment import (
    fused_gromov_wasserstein_plan,
    representation_transport_plan,
    transport_plan_alignment_loss,
)
from models.saot import SAOTModel, edge_index_to_adjacency
from data.replay_buffer import ReplayBuffer
from evaluation.metrics import average_forgetting, average_performance, save_performance_matrix
from evaluation.linear_probe import evaluate_frozen_encoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_graph(x: Tensor, adjacency: Tensor, edge_drop: float, feature_mask: float) -> Tuple[Tensor, Tensor]:
    """Paper augmentation: random edge dropping and feature masking."""
    keep = torch.rand_like(adjacency).lt(1.0 - edge_drop)
    keep = torch.triu(keep, diagonal=1)
    keep = keep | keep.T
    a = adjacency * keep.float()
    mask = torch.rand_like(x).lt(feature_mask)
    return x.masked_fill(mask, 0.0), a


def augment_sparse_graph(x: Tensor, edge_index: Tensor, edge_drop: float,
                         feature_mask: float) -> Tuple[Tensor, Tensor]:
    keep = torch.rand(edge_index.shape[1], device=edge_index.device).ge(edge_drop)
    kept = edge_index[:, keep]
    mask = torch.rand_like(x).lt(feature_mask)
    return x.masked_fill(mask, 0.0), kept


def synthetic_tasks(num_tasks: int, in_dim: int, nodes_per_task: int = 12) -> List[Dict[str, Tensor]]:
    tasks = []
    for task_id in range(num_tasks):
        x = torch.randn(nodes_per_task, in_dim)
        a = torch.zeros(nodes_per_task, nodes_per_task)
        for i in range(nodes_per_task - 1):
            a[i, i + 1] = a[i + 1, i] = 1.0
        extra = torch.rand(nodes_per_task, nodes_per_task).lt(0.12)
        a = torch.maximum(a, torch.triu(extra, diagonal=1).float())
        a = torch.maximum(a, a.T)
        y = torch.arange(nodes_per_task) % 2
        tasks.append({
            "x": x,
            "adjacency": a,
            "y": y,
            "train_mask": torch.arange(nodes_per_task) % 3 != 0,
            "validation_mask": torch.arange(nodes_per_task) % 3 == 0,
            "test_mask": torch.arange(nodes_per_task) % 3 == 0,
            "classes": torch.tensor([0, 1], dtype=torch.long),
        })
    return tasks


def load_tasks(path: str) -> List[Dict[str, Tensor]]:
    payload = torch.load(path, map_location="cpu")
    raw = payload["tasks"] if isinstance(payload, dict) and "tasks" in payload else payload
    tasks = []
    for item in raw:
        x = item["x"].float()
        if "edge_index" in item:
            edge_index = item["edge_index"].long()
            adjacency = edge_index_to_adjacency(edge_index, x.shape[0])
        elif "adjacency" in item:
            adjacency = item["adjacency"].float()
            edge_index = adjacency.nonzero(as_tuple=False).T.contiguous().long()
        else:
            raise ValueError("each task needs edge_index or adjacency")
        tasks.append({
            "task_id": item.get("task_id"),
            "x": x,
            "adjacency": adjacency,
            "edge_index": edge_index,
            "y": None if item.get("y") is None else item["y"].long(),
            "classes": None if item.get("classes") is None else item["classes"].long(),
            "train_mask": item.get("train_mask"),
            "validation_mask": item.get("validation_mask"),
            "test_mask": item.get("test_mask"),
            "eval_mask": item.get("eval_mask"),
        })
    return tasks


def _as_edge_index(adjacency: Tensor) -> Tensor:
    return adjacency.nonzero(as_tuple=False).T.contiguous().long()


def _disjoint_union(parts: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
    xs, edges, offset = [], [], 0
    for part in parts:
        x = part["x"]
        edge = part.get("edge_index", _as_edge_index(part["adjacency"]))
        xs.append(x)
        edges.append(edge + offset)
        offset += x.shape[0]
    return torch.cat(xs, 0), torch.cat(edges, 1) if edges else torch.empty((2, 0), dtype=torch.long)


def _sample_graph(x: Tensor, edge_index: Tensor, budget: int, seed: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample a bounded point cloud before quadratic OT, as in Section 5.7."""
    if x.shape[0] <= budget:
        nodes = torch.arange(x.shape[0], device=x.device)
    else:
        # Generate the deterministic index on CPU, then move it to the graph
        # device so GPU tensors never get indexed by a CPU mapping tensor.
        cpu_generator = torch.Generator().manual_seed(seed)
        nodes = torch.randperm(x.shape[0], generator=cpu_generator)[:budget].to(x.device)
        nodes, _ = torch.sort(nodes)
    mapping = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    mapping[nodes] = torch.arange(nodes.numel())
    keep = (mapping[edge_index[0]] >= 0) & (mapping[edge_index[1]] >= 0)
    return x[nodes], mapping[edge_index[:, keep]], nodes


def _task_adjacency(task: Dict[str, Tensor]) -> Tensor:
    if "adjacency" in task:
        return task["adjacency"]
    return edge_index_to_adjacency(task["edge_index"], task["x"].shape[0])


def _nearest_centroid_accuracy(z: Tensor, y: Tensor, class_centroids: Dict[int, Tensor]) -> float:
    if y is None or y.numel() == 0 or not class_centroids:
        return float("nan")
    labels = sorted(class_centroids)
    centroids = torch.stack([class_centroids[c] for c in labels])
    pred = torch.cdist(z, centroids).argmin(1)
    target = torch.tensor([labels.index(int(v)) if int(v) in labels else -1 for v in y], device=z.device)
    valid = target >= 0
    return float((pred[valid] == target[valid]).float().mean()) if valid.any() else float("nan")


def run(config: Dict, seed: int, smoke_test: bool = False, no_distillation: bool = False) -> Dict:
    set_seed(seed)
    in_dim = int(config.get("input_dim", 32))
    num_tasks = int(config.get("smoke_tasks", 3) if smoke_test else config.get("num_tasks", 35))
    tasks = synthetic_tasks(num_tasks, in_dim) if smoke_test else load_tasks(config["task_file"])
    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    in_dim = int(tasks[0]["x"].shape[1])
    for task in tasks:
        task.setdefault("edge_index", _as_edge_index(task["adjacency"]))
        for key in ("x", "edge_index", "y", "train_mask", "validation_mask", "test_mask", "classes"):
            if task.get(key) is not None:
                task[key] = task[key].to(device)
    model = SAOTModel(in_dim, int(config.get("hidden_dim", 512)), int(config.get("out_dim", 128)), float(config.get("dropout", 0.0))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    teacher_state = None
    replay = ReplayBuffer(int(config.get("replay_budget", 200)), seed)
    performance = np.full((len(tasks), len(tasks)), np.nan, dtype=float)
    centroids: Dict[int, Tensor] = {}
    rows: List[Dict] = []
    epochs = int(config.get("epochs", 2 if smoke_test else 100))
    alpha, beta = float(config.get("alpha", 1.0)), float(config.get("beta", 0.6))
    sigma, epsilon = float(config.get("sigma", 0.5)), float(config.get("ot_epsilon", 0.05))
    for task_id, task in enumerate(tasks):
        x, adjacency, edge_index = task["x"], _task_adjacency(task), task["edge_index"]
        task_x, task_edge, task_nodes = _sample_graph(x, edge_index, int(config.get("ot_node_budget", 256)), seed + task_id)
        model.train()
        for epoch in range(epochs):
            replay_parts = [{"x": item.x.to(device), "edge_index": item.edge_index.to(device)} for item in replay.items()]
            batch_x, batch_edge = _disjoint_union([{"x": task_x, "edge_index": task_edge}] + replay_parts)
            if batch_x.shape[0] > int(config.get("ot_node_budget", 256)):
                batch_x, batch_edge, _ = _sample_graph(batch_x, batch_edge, int(config.get("ot_node_budget", 256)), seed + task_id + epoch)
            batch_adj = edge_index_to_adjacency(batch_edge, batch_x.shape[0])
            x1, e1 = augment_sparse_graph(batch_x, batch_edge, float(config.get("edge_drop", 0.2)), float(config.get("feature_mask", 0.2)))
            x2, e2 = augment_sparse_graph(batch_x, batch_edge, float(config.get("edge_drop", 0.2)), float(config.get("feature_mask", 0.2)))
            a1 = edge_index_to_adjacency(e1, x1.shape[0]); a2 = edge_index_to_adjacency(e2, x2.shape[0])
            z1, z2 = model(x1, edge_index=e1, sparse=True), model(x2, edge_index=e2, sparse=True)
            pi_g, graph_cost = fused_gromov_wasserstein_plan(x1, x2, a1, a2, sigma=sigma, epsilon=epsilon)
            pi_z = representation_transport_plan(z1, z2, epsilon=epsilon)
            rep_cost = torch.cdist(z1, z2).pow(2)
            l_mat = transport_plan_alignment_loss(pi_g, pi_z)
            l_str = structural_calibration_loss(graph_cost.detach(), rep_cost, pi_g.detach())
            if teacher_state is None or no_distillation:
                l_skd = z1.sum() * 0.0
            else:
                teacher = model.teacher(teacher_state)
                with torch.no_grad():
                    teacher_z1, teacher_z2 = teacher(x1, edge_index=e1, sparse=True), teacher(x2, edge_index=e2, sparse=True)
                    teacher_plan = representation_transport_plan(teacher_z1, teacher_z2, epsilon=epsilon)
                l_skd = cross_task_plan_distillation_loss(teacher_plan, pi_z)
            loss = saot_total_loss(l_mat, l_str, l_skd, alpha, beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rows.append({"task": task_id, "epoch": epoch, "loss": float(loss.detach()), "l_mat": float(l_mat.detach()), "l_str": float(l_str.detach()), "l_skd": float(l_skd.detach())})
        teacher_state = model.snapshot_teacher()
        replay.add(task_id, x, edge_index, task.get("y"), task.get("eval_mask"))
        if task.get("y") is not None and all("train_mask" in item and "test_mask" in item for item in tasks[: task_id + 1]):
            probe_scores = evaluate_frozen_encoder(
                model,
                tasks,
                task_id,
                scenario=str(config.get("scenario", "class-il")),
                epochs=int(config.get("linear_probe_epochs", 100)),
                learning_rate=float(config.get("linear_probe_learning_rate", 0.01)),
                seed=seed + task_id,
            )
            for eval_task_id, score in probe_scores.items():
                performance[task_id, eval_task_id] = score
    output_root = Path(config.get("output_root", "results")) / "saot" / str(config.get("dataset", "CoraFull-CL")) / str(config.get("scenario", "class-il")) / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**config, "seed": seed, "no_distillation": no_distillation}, f, indent=2, default=str)
    with (output_root / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    torch.save({"model": model.state_dict(), "teacher": teacher_state, "seed": seed}, output_root / "checkpoint.pt")
    save_performance_matrix(performance, output_root / "performance_matrix.csv")
    metrics = {"average_performance": average_performance(performance), "average_forgetting": average_forgetting(performance)}
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "tasks": num_tasks, "final_loss": rows[-1]["loss"], "no_distillation": no_distillation, **metrics}, f, indent=2)
    return {"output": str(output_root), "final_loss": rows[-1]["loss"], "rows": rows, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-distillation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = run(config, args.seed, args.smoke_test, args.no_distillation)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
