"""Validate SAOT graph-space and representation-space transport on toy graphs.

Run from the repository root:

    python -m experiments.toy_ot_alignment --output-dir results/toy_ot

The experiment creates an eight-node graph, permutes its nodes to obtain a
second view, perturbs one edge, and studies how ``sigma`` changes the FGW plan.
All matrices, logs, CSV summaries, and heatmaps are written to the output
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.ot_solver import (
    fused_gromov_wasserstein,
    pairwise_cosine_cost,
    representation_transport_plan,
)


def _adjacency(node_count: int, edges: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    for left, right in edges:
        matrix[left, right] = 1.0
        matrix[right, left] = 1.0
    return matrix


def make_toy_graphs(seed: int = 42) -> dict[str, np.ndarray]:
    """Return two related eight-node graphs with a known correspondence."""

    rng = np.random.default_rng(seed)
    n = 8
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # dense community A
        (4, 5), (5, 6), (6, 7), (7, 4), (5, 7),  # dense community B
        (2, 4),  # bridge
    ]
    adjacency_1 = _adjacency(n, edges)
    # Two community channels plus an identity-like attribute channel per node.
    # The latter makes the known permutation observable while topology still
    # matters when sigma is small.
    features_1 = np.zeros((n, 2 + n), dtype=np.float64)
    features_1[:4, 0] = 1.0
    features_1[4:, 1] = 1.0
    features_1[:, 2:] = 0.8 * np.eye(n)
    # graph-2 index j corresponds to graph-1 index permutation[j].
    permutation = np.array([5, 1, 7, 3, 0, 6, 2, 4], dtype=int)
    adjacency_2 = adjacency_1[np.ix_(permutation, permutation)].copy()
    # One view-specific topology perturbation.
    adjacency_2[0, 2] = adjacency_2[2, 0] = 0.0
    adjacency_2[0, 7] = adjacency_2[7, 0] = 1.0
    features_2 = features_1[permutation] + rng.normal(0.0, 0.02, features_1.shape)

    # Embeddings preserve community and node identity but add mild view noise.
    projection = rng.normal(0.0, 1.0, (2 + n, 8))
    embeddings_1 = features_1 @ projection
    embeddings_2 = features_2 @ projection + rng.normal(0.0, 0.01, (n, 8))
    expected_target = np.argsort(permutation)
    return {
        "adjacency_1": adjacency_1,
        "adjacency_2": adjacency_2,
        "features_1": features_1,
        "features_2": features_2,
        "embeddings_1": embeddings_1,
        "embeddings_2": embeddings_2,
        "permutation": permutation,
        "expected_target": expected_target,
    }


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.save(path.with_suffix(".npy"), matrix)
    np.savetxt(path.with_suffix(".csv"), matrix, delimiter=",", fmt="%.10f")


def _heat_color(value: float, maximum: float) -> tuple[int, int, int]:
    ratio = 0.0 if maximum <= 0 else min(max(value / maximum, 0.0), 1.0)
    # Light cream -> coral -> deep navy, readable in print and on screen.
    if ratio < 0.5:
        t = ratio / 0.5
        return tuple(int(a + (b - a) * t) for a, b in zip((249, 245, 235), (232, 124, 96)))
    t = (ratio - 0.5) / 0.5
    return tuple(int(a + (b - a) * t) for a, b in zip((232, 124, 96), (31, 52, 79)))


def save_heatmap(matrix: np.ndarray, path: Path, title: str, expected: np.ndarray) -> None:
    cell = 62
    left, top, right, bottom = 92, 92, 28, 70
    width = left + matrix.shape[1] * cell + right
    height = top + matrix.shape[0] * cell + bottom
    image = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17)
    title_font = ImageFont.load_default(size=24)
    draw.text((left, 24), title, fill=(20, 34, 51), font=title_font)
    maximum = float(matrix.max())
    for i in range(matrix.shape[0]):
        draw.text((35, top + i * cell + 20), f"G1-{i}", fill=(20, 34, 51), font=font)
        for j in range(matrix.shape[1]):
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=_heat_color(matrix[i, j], maximum))
            text = f"{matrix[i, j]:.3f}"
            color = (255, 255, 255) if matrix[i, j] > maximum * 0.62 else (20, 34, 51)
            draw.text((x0 + 9, y0 + 20), text, fill=color, font=font)
            if j == int(expected[i]):
                draw.rectangle((x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2), outline=(255, 215, 80), width=4)
    for j in range(matrix.shape[1]):
        draw.text((left + j * cell + 15, top - 28), f"G2-{j}", fill=(20, 34, 51), font=font)
    draw.text((left, height - 40), "Gold outline = known node correspondence", fill=(80, 88, 96), font=font)
    image.save(path)


def plan_accuracy(plan: np.ndarray, expected_target: np.ndarray) -> float:
    return float(np.mean(np.argmax(plan, axis=1) == expected_target))


def run(output_dir: Path, seed: int, sigmas: list[float]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_toy_graphs(seed)
    for name, matrix in data.items():
        if name not in {"permutation", "expected_target"}:
            _save_matrix(output_dir / name, matrix)
    np.savetxt(output_dir / "permutation.csv", data["permutation"], delimiter=",", fmt="%d")
    np.savetxt(output_dir / "expected_target.csv", data["expected_target"], delimiter=",", fmt="%d")

    feature_cost = pairwise_cosine_cost(data["features_1"], data["features_2"])
    _save_matrix(output_dir / "feature_cost", feature_cost)
    rows: list[dict[str, object]] = []
    experiment_log: dict[str, object] = {"seed": seed, "sigmas": sigmas, "fgw": {}}

    for sigma in sigmas:
        result = fused_gromov_wasserstein(
            feature_cost,
            data["adjacency_1"],
            data["adjacency_2"],
            sigma=sigma,
            epsilon=0.02,
            max_iter=300,
            tolerance=1e-9,
            structure_loss="absolute",
            relaxation=0.5,
        )
        label = f"sigma_{sigma:.2f}".replace(".", "p")
        _save_matrix(output_dir / f"pi_g_{label}", result.plan)
        save_heatmap(
            result.plan,
            output_dir / f"transport_plan_{label}.png",
            f"Graph-space FGW plan, sigma={sigma:.2f}",
            data["expected_target"],
        )
        accuracy = plan_accuracy(result.plan, data["expected_target"])
        diagonal_mass = float(result.plan[np.arange(8), data["expected_target"]].sum())
        row = {
            "sigma": sigma,
            "objective": result.objective,
            "feature_term": result.feature_term,
            "structure_term": result.structure_term,
            "top1_correspondence_accuracy": accuracy,
            "expected_correspondence_mass": diagonal_mass,
            "iterations": result.iterations,
            "converged": result.converged,
            "marginal_error": result.marginal_error,
        }
        rows.append(row)
        experiment_log["fgw"][str(sigma)] = {**row, "objective_history": result.objective_history}

    z_result = representation_transport_plan(
        data["embeddings_1"], data["embeddings_2"], epsilon=0.02, tolerance=1e-10
    )
    _save_matrix(output_dir / "pi_z", z_result.plan)
    save_heatmap(
        z_result.plan,
        output_dir / "transport_plan_pi_z.png",
        "Representation-space OT plan (pi_Z)",
        data["expected_target"],
    )
    experiment_log["representation_ot"] = {
        "cost": z_result.cost,
        "top1_correspondence_accuracy": plan_accuracy(z_result.plan, data["expected_target"]),
        "expected_correspondence_mass": float(
            z_result.plan[np.arange(8), data["expected_target"]].sum()
        ),
        "iterations": z_result.iterations,
        "converged": z_result.converged,
        "marginal_error": z_result.marginal_error,
        "objective_history": z_result.objective_history,
    }

    with (output_dir / "sigma_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "experiment_log.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_log, handle, indent=2)
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Toy OT alignment results\n\n"
            "The gold outline in each heatmap marks the known permutation between the two "
            "eight-node graph views. `sigma_ablation.csv` reports how the graph-space plan "
            "changes as feature weight `sigma` moves from structure-only to feature-only. "
            "Every displayed matrix is also saved as CSV and NPY.\n"
        )
    return experiment_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_ot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    log = run(args.output_dir, args.seed, args.sigmas)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
