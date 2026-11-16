"""Smoke tests for the member 5 baselines and the unified runner.

Skipped automatically when torch or PyTorch Geometric are unavailable, so the
NumPy-only metric tests keep running on machines without the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
try:
    pytest.importorskip("torch_geometric")
except Exception as error:  # a broken torch install raises OSError, not ImportError
    pytest.skip(f"torch/PyG unavailable: {error}", allow_module_level=True)

from baselines.gae_or_dgi import GAEMethod  # noqa: E402
from baselines.gcn_sequential import GCNSequentialMethod  # noqa: E402
from baselines.replay_baseline import ReplayMethod  # noqa: E402
from evaluation.metrics import PerformanceMatrix  # noqa: E402
from evaluation.protocol import TaskSplit, restricted_accuracy  # noqa: E402
from experiments.main_baselines import (  # noqa: E402
    build_synthetic_graph,
    run_single,
    split_continual_tasks,
)

EPOCHS = 2
HIDDEN_DIM = 8


def _settings(scenario: str) -> dict[str, object]:
    return {
        "dataset": "toy",
        "seeds": [42],
        "scenarios": ["class-il", "task-il"],
        "classes_per_task": 2,
        "epochs_per_task": EPOCHS,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "hidden_dim": HIDDEN_DIM,
        "encoder_type": "gcn",
        "num_layers": 2,
        "dropout": 0.0,
        "replay_budget": 4,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "device": "cpu",
        "output_root": "results",
        "log_dir": "logs",
        "data_source": "synthetic",
        "scenario": scenario,
    }


def _bundle():
    graph = build_synthetic_graph(
        num_classes=6, nodes_per_class=8, feature_dim=8, seed=42
    )
    return split_continual_tasks(
        graph, dataset="toy", seed=42, classes_per_task=2, source="synthetic"
    )


def _kwargs(scenario: str, method_name: str, *, replay_budget: int | None = 4) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "hidden_dim": HIDDEN_DIM,
        "encoder_type": "gcn",
        "num_layers": 2,
        "dropout": 0.0,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "epochs_per_task": EPOCHS,
        "scenario": scenario,
        "device": "cpu",
    }
    if method_name == "replay" and replay_budget is not None:
        kwargs["replay_budget"] = replay_budget
    return kwargs


@pytest.mark.parametrize("scenario", ["class-il", "task-il"])
def test_baselines_write_uniform_artifacts(tmp_path: Path, scenario: str) -> None:
    bundle = _bundle()
    factories = {
        "gcn_sequential": GCNSequentialMethod,
        "gae": GAEMethod,
        "replay": ReplayMethod,
    }

    for method_name, factory in factories.items():
        row = run_single(
            method_name=method_name,
            method=factory(**_kwargs(scenario, method_name)),
            bundle=bundle,
            scenario=scenario,
            seed=42,
            settings=_settings(scenario),
            output_root=tmp_path,
            log_root=tmp_path / "logs",
        )
        run_dir = Path(row["run_dir"])

        for name in (
            "config.yaml",
            "metrics.json",
            "performance_matrix.csv",
            "train.log",
            "checkpoint.pt",
        ):
            assert (run_dir / name).exists(), f"{method_name} missing {name}"

        matrix = PerformanceMatrix.from_csv(run_dir / "performance_matrix.csv")
        assert matrix.shape == (bundle.split.num_tasks, bundle.split.num_tasks)
        assert not any(
            value is None
            for value in matrix.summary(prefix=False).to_dict()["per_task_final_accuracy"]
        )

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert 0.0 <= metrics["ap"] <= 1.0
        assert set(metrics["scenarios"]) == {"class-il", "task-il"}
        assert metrics["scenario"] == scenario
        # The second matrix of the same run must also be written.
        other = "task-il" if scenario == "class-il" else "class-il"
        assert (run_dir / f"performance_matrix_{other.replace('-', '_')}.csv").exists()


def test_replay_with_zero_budget_matches_plain_sequential(tmp_path: Path) -> None:
    bundle = _bundle()
    scenario = "class-il"

    plain = run_single(
        method_name="gcn_sequential",
        method=GCNSequentialMethod(**_kwargs(scenario, "gcn_sequential")),
        bundle=bundle,
        scenario=scenario,
        seed=42,
        settings=_settings(scenario),
        output_root=tmp_path / "plain",
        log_root=tmp_path / "logs",
    )
    replay = run_single(
        method_name="replay",
        method=ReplayMethod(**_kwargs(scenario, "replay", replay_budget=0)),
        bundle=bundle,
        scenario=scenario,
        seed=42,
        settings=_settings(scenario),
        output_root=tmp_path / "replay",
        log_root=tmp_path / "logs",
    )

    assert replay["ap"] == pytest.approx(plain["ap"], abs=1e-9)
    assert replay["af"] == pytest.approx(plain["af"], abs=1e-9)


def test_restricted_accuracy_maps_positions_back_to_class_ids() -> None:
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.0], [0.0, 0.1, 0.2, 4.0]])
    targets = torch.tensor([2, 3])

    assert restricted_accuracy(logits, targets, [2, 3]) == pytest.approx(1.0)
    assert restricted_accuracy(logits, targets, [0, 1]) == pytest.approx(0.0)


def test_task_split_rejects_invalid_class_assignment() -> None:
    graph = build_synthetic_graph(num_classes=4, nodes_per_class=6, feature_dim=6, seed=0)
    bundle = split_continual_tasks(graph, dataset="toy", seed=0, classes_per_task=2)
    split = bundle.split
    assert split.num_tasks == 2
    assert split.classes_seen_until(0) == [0, 1]
    assert split.classes_seen_until(1) == [0, 1, 2, 3]

    with pytest.raises(ValueError):
        # class 1 must not belong to two tasks
        TaskSplit(
            num_nodes=split.num_nodes,
            num_classes=split.num_classes,
            task_to_classes={0: [0, 1], 1: [1, 3]},
            task_to_train_ids=split.task_to_train_ids,
            task_to_val_ids=split.task_to_val_ids,
            task_to_test_ids=split.task_to_test_ids,
        )
    with pytest.raises(ValueError):
        # 99 is outside [0, num_classes - 1]
        TaskSplit(
            num_nodes=split.num_nodes,
            num_classes=split.num_classes,
            task_to_classes={0: [0, 1], 1: [2, 99]},
            task_to_train_ids=split.task_to_train_ids,
            task_to_val_ids=split.task_to_val_ids,
            task_to_test_ids=split.task_to_test_ids,
        )
