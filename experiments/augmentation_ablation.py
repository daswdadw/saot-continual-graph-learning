"""Run member 3 augmentation and replay-buffer ablations.

The script is usable before the CoraFull-CL data pipeline is available: by
default it creates a deterministic synthetic graph and exercises the actual
augmentor, replay buffer, neighbor sampler and GNN encoder.  Once member 1's
loader is ready, ``build_experiment_graph`` is the only function that needs to
be replaced by a project dataset loader.

Example:
    python experiments/augmentation_ablation.py \
        --config configs/local.yaml --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data

# Support both ``python -m experiments.augmentation_ablation`` and the project
# convention ``python experiments/augmentation_ablation.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.augment import create_augmented_views
from data.neighbor_sampling import build_replay_loader
from data.replay_buffer import ReplayBuffer
from models.gnn_encoder import GNNEncoder


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch without requiring CUDA."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_experiment_graph(
    *, num_nodes: int = 256, input_dim: int = 32, seed: int = 42
) -> Data:
    """Build a deterministic ring-plus-shortcuts graph for a local smoke run."""

    if num_nodes < 2:
        raise ValueError(f"num_nodes must be at least 2, got {num_nodes}.")
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    generator = torch.Generator().manual_seed(seed)
    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    shortcut_target = (source + 7) % num_nodes
    edge_source = torch.cat([source, target, source, shortcut_target])
    edge_target = torch.cat([target, source, shortcut_target, source])
    edge_index = torch.stack([edge_source, edge_target])
    x = torch.randn(num_nodes, input_dim, generator=generator)
    y = (torch.arange(num_nodes) % 4).long()
    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=torch.ones(edge_index.size(1)),
        num_nodes=num_nodes,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    title: str,
    *,
    left_key: str,
    left_label: str,
) -> None:
    """Write a small PNG if matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [float(row[x_key]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x_values, [float(row[left_key]) for row in rows], marker="o")
    axes[0].set_xlabel(x_key)
    axes[0].set_ylabel(left_label)
    axes[0].set_title(left_label)
    axes[1].plot(x_values, [float(row["elapsed_ms"]) for row in rows], marker="o")
    axes[1].set_xlabel(x_key)
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_title("Runtime")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _train_proxy_step(
    graph: Data,
    *,
    hidden_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    """Run one local encoder step and return ``(loss, elapsed_ms)``.

    This is deliberately a proxy step for the code-only ablation.  The final
    AP/AF experiment will use the group's sequential training and evaluation
    pipeline once member 1 and member 4 expose those interfaces.
    """

    torch.manual_seed(seed)
    model = GNNEncoder(
        input_dim=graph.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        encoder_type="gcn",
        num_layers=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    local_graph = graph.to(device)
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    embedding = model(
        local_graph.x,
        local_graph.edge_index,
        edge_weight=local_graph.edge_weight,
    )
    loss = embedding.square().mean()
    loss.backward()
    optimizer.step()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(loss.detach().cpu()), elapsed_ms


def run_augmentation_ablation(
    config: dict[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
    edge_drop_values: list[float] | None = None,
    buffer_budgets: list[int] | None = None,
    num_nodes: int = 256,
    input_dim: int = 32,
    device: str | torch.device | None = None,
) -> dict[str, Path]:
    """Run both code-level ablations and return generated file paths."""

    seed_everything(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    edge_drop_values = edge_drop_values or [0.0, 0.2, 0.5]
    buffer_budgets = buffer_budgets or [0, 200, 800]
    feature_mask = float(config.get("feature_mask", 0.2))
    hidden_dim = int(config.get("hidden_dim", 64))
    replay_seed = int(config.get("seed", seed))
    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    graph = build_experiment_graph(
        num_nodes=num_nodes,
        input_dim=input_dim,
        seed=seed,
    )
    augmentation_rows: list[dict[str, Any]] = []
    for edge_drop in edge_drop_values:
        start = time.perf_counter()
        view_1, view_2 = create_augmented_views(
            graph,
            edge_drop=float(edge_drop),
            feature_mask=feature_mask,
            seed=seed,
        )
        loss, train_ms = _train_proxy_step(
            view_1,
            hidden_dim=hidden_dim,
            seed=seed,
            device=chosen_device,
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        augmentation_rows.append(
            {
                "seed": seed,
                "edge_drop": float(edge_drop),
                "feature_mask": feature_mask,
                "num_edges": int(view_1.num_edges),
                "view_2_num_edges": int(view_2.num_edges),
                "masked_features": int(view_1.augmentation_feature_mask.sum()),
                "proxy_loss": loss,
                "train_elapsed_ms": train_ms,
                "elapsed_ms": total_ms,
            }
        )

    task_count = max(1, min(8, int(config.get("num_tasks", 4))))
    task_nodes = torch.arange(num_nodes, dtype=torch.long).chunk(task_count)
    buffer_rows: list[dict[str, Any]] = []
    for budget in buffer_budgets:
        start = time.perf_counter()
        buffer = ReplayBuffer(int(budget), seed=replay_seed)
        for task_id, nodes in enumerate(task_nodes):
            buffer.add(task_id, nodes)
        seed_sample = buffer.sample(min(32, len(buffer)), seed=seed)
        loader = build_replay_loader(
            graph,
            seed_sample,
            num_neighbors=[10, 10],
            batch_size=16,
            direction="both",
            seed=seed,
        )
        batches = 0
        sampled_nodes = 0
        model = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            encoder_type="gcn",
            num_layers=2,
        ).to(chosen_device)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(chosen_device)
                model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
                batches += 1
                sampled_nodes += int(batch.n_id.numel())
        buffer_rows.append(
            {
                "seed": seed,
                "replay_budget": int(budget),
                "buffer_size": len(buffer),
                "num_tasks": buffer.num_tasks,
                "sample_size": int(seed_sample.numel()),
                "loader_batches": batches,
                "sampled_subgraph_nodes": sampled_nodes,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            }
        )

    config_path = output_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                **config,
                "seed": seed,
                "device": str(chosen_device),
                "num_nodes": num_nodes,
                "input_dim": input_dim,
                "edge_drop_values": edge_drop_values,
                "buffer_budgets": buffer_budgets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    augmentation_csv = output_path / "augmentation_results.csv"
    buffer_csv = output_path / "buffer_results.csv"
    _write_csv(augmentation_csv, augmentation_rows)
    _write_csv(buffer_csv, buffer_rows)
    augmentation_plot = output_path / "augmentation_results.png"
    buffer_plot = output_path / "buffer_results.png"
    _write_plot(
        augmentation_plot,
        augmentation_rows,
        "edge_drop",
        "Member 3 augmentation ablation",
        left_key="num_edges",
        left_label="Edges kept",
    )
    _write_plot(
        buffer_plot,
        buffer_rows,
        "replay_budget",
        "Member 3 replay buffer ablation",
        left_key="buffer_size",
        left_label="Buffer size",
    )
    return {
        "config": config_path,
        "augmentation_csv": augmentation_csv,
        "buffer_csv": buffer_csv,
        "augmentation_plot": augmentation_plot,
        "buffer_plot": buffer_plot,
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping, got {type(config).__name__}.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    config = _load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    output_dir = args.output_dir or str(
        Path(config.get("output_root", "results")) / "member_3" / "augmentation"
    )
    paths = run_augmentation_ablation(
        config,
        seed=seed,
        output_dir=output_dir,
        num_nodes=args.num_nodes,
        input_dim=args.input_dim,
        device=args.device,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
