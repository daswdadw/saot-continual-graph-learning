# AutoDL Experiment Runbook

Use this runbook after the blocking integration items in `AUDIT_REPORT.md` are resolved. The order is staged so an interface failure consumes minutes rather than a full GPU run.

## 0. Hardware choice

| Stage | Recommended device | Reason |
|---|---|---|
| Smoke test and CoraFull-CL | RTX 4090 24 GB or better | Sampled transport budgets fit comfortably for development |
| Arxiv-CL | RTX 4090 for development; L40S 48 GB or A100 40/80 GB for final runs | More tasks and replay require a larger memory margin |
| Reddit-CL | L40S 48 GB or A100 80 GB | Dense neighborhoods and the paper's GAT encoder are expensive |
| Products-CL | A100 80 GB or equivalent | 2.45 million nodes require strict sampling |

The paper reports NVIDIA L40S 48 GB. For a defensible CoraFull/Arxiv subset, one RTX 4090 is sufficient for development; use L40S/A100 for final scale runs.

## 1. Prepare AutoDL

Select a PyTorch image with Python 3.10 or 3.11 and a matching CUDA version:

```bash
git clone https://github.com/daswdadw/saot-continual-graph-learning.git
cd saot-continual-graph-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch Geometric wheels matching the installed PyTorch/CUDA pair according to the official PyG wheel index. Verify the GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')
PY
```

Do not start a GPU experiment if CUDA is unavailable.

## 2. Independent OT validation

```bash
python -m unittest discover -s tests -p 'test_ot_solver.py' -v
python -m experiments.toy_ot_alignment --output-dir results/toy_ot --seed 42
```

Check plan marginals, `sigma_ablation.csv`, convergence flags, and heatmaps under `results/toy_ot/`.

## 3. Prepare CoraFull-CL

```bash
python data/prepare_cgl.py --dataset CoraFull-CL --seed 42
```

Save `data/task_splits/corafull_seed42.json` and its SHA-256 hash. Do not change class order, split ratios, or task order after the first baseline.

The JSON is metadata, not the tensor file consumed by the integrated trainer. Create and checksum:

```text
data/processed/corafull_tasks_seed42.pt
data/processed/corafull_tasks_seed42.sha256
```

Run the conversion from the repository root:

```bash
python data/build_task_tensor.py \
  --dataset CoraFull-CL \
  --split data/task_splits/corafull_seed42.json \
  --output data/processed/corafull_tasks_seed42.pt
```

Each task record must contain local `x`, local `edge_index`, `y`, and train/validation/test masks. Edges must be induced within the task so inter-task edges are not reintroduced.

## 4. Tiny integrated smoke test

```bash
python experiments/train_sequential.py --config configs/example.yaml --seed 42 --smoke-test
python experiments/member_4_ablation.py --config configs/example.yaml --seed 42 --smoke-test
```

The smoke test only checks integration. Do not compare its AP/AF with the paper.

## 5. CoraFull development run

Create a local config based on `configs/example.yaml`:

```yaml
dataset: CoraFull-CL
scenario: class-il
task_file: data/processed/corafull_tasks_seed42.pt
device: cuda
hidden_dim: 512
out_dim: 128
learning_rate: 0.001
alpha: 1.0
beta: 0.6
sigma: 0.5
ot_epsilon: 0.05
epochs: 100
replay_budget: 200
ot_node_budget: 256
eval_node_budget: 2000
num_tasks: 35
```

Run one seed:

```bash
python experiments/train_sequential.py --config configs/corafull_class_il.yaml --seed 42
```

Confirm the config snapshot, loss log, checkpoint, and performance matrix. Check that transport matrices are bounded by `ot_node_budget` rather than the full graph size.

## 6. Paper-style evaluation protocol

At each task checkpoint `i`, freeze the encoder and train a linear classifier on permitted training nodes. Evaluate every task `j` and write `M_p[i,j]`:

```text
AP = mean(M_p[T-1, :])
AF = mean(M_p[T-1, i] - M_p[i, i]) for i = 0 ... T-2
```

Run Class-IL first, then Task-IL with task identity supplied during evaluation. Keep separate result directories under `results/saot/CoraFull-CL/`.

The updated audit copy uses `evaluation/linear_probe.py` for this protocol. Verify the probe on AutoDL before using AP/AF in a report.

## 7. Seeds and ablations

After seed 42 is stable, run seeds `42 43 44 45 46` and report mean and standard deviation. Required Class-IL ablations are SAOT, cosine distillation, MSE distillation, no OT alignment, and no cross-task distillation.

```bash
python experiments/member_4_ablation.py --config configs/corafull_class_il.yaml --seed 42
python experiments/ablation.py --config configs/ablation.yaml --seeds 42 43 44 45 46
```

Keep task file, encoder, augmentations, replay budget, transport budget, optimizer, update count, and seeds identical across variants.

## 8. Arxiv scale run

```bash
python data/prepare_cgl.py --dataset Arxiv-CL --seed 42
```

Use replay budget 800 and a fixed `ot_node_budget`. Start with a one-task or few-task memory benchmark, then run all 20 tasks only after peak memory and transport time are recorded.

## 9. AutoDL artifacts to upload

- exact YAML configuration;
- task split JSON and SHA-256 hash;
- task tensor manifest and hash;
- `performance_matrix.csv`;
- `metrics.json` with AP, AF, seed, runtime, and peak memory;
- loss and training logs;
- ablation summary CSVs;
- generated figures.

Do not commit raw datasets, checkpoints, `.npy` caches, or large logs to GitHub.
