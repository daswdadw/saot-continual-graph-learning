# SAOT Paper-Aligned Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository a paper-aligned, runnable implementation of SAOT's replay-based continual graph representation learning pipeline.

**Architecture:** Keep the paper's two-stage objective intact: FGW plan construction in graph space, Wasserstein plan construction in representation space, plan alignment/structural calibration, and frozen-teacher plan distillation. Add a bounded replay buffer, sparse edge-index GCN path, task-level evaluation, and paper-style result artifacts around the existing losses.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, optional SciPy/POT for comparison.

**Spec:** `docs/paper/ICML2026_66335_SAOT.pdf`, Sections 4.1--5.6.

## Global Constraints

- Use `L_total = L_mat + alpha * L_str + beta * L_skd`.
- Use graph cost `c_A(A1_ik, A2_jl) = |A1_ik - A2_jl|` and uniform empirical node masses unless task data supplies masses.
- Freeze the encoder state after each task and use it as the next task's teacher.
- Keep graph alignment bounded by a configurable node budget because the paper reports quadratic OT cost.
- Save config, losses, checkpoint, and performance matrix under `results/<model>/<dataset>/<scenario>/seed_<seed>/`.

---

### Task 1: Sparse encoder and replay buffer

**Files:** Modify `models/saot.py`; create `data/replay_buffer.py`; test in `tests/test_member4_pipeline.py`.

- Add edge-index message passing without materializing an `N x N` adjacency matrix.
- Store at most `budget` nodes per task, preserving features, labels, and induced edges.
- Test sparse output shape and strict replay budget.

### Task 2: Paper metrics and task data adapter

**Files:** Create `evaluation/metrics.py`; modify `experiments/train_sequential.py`; test AP/AF and task loading.

- Implement performance matrix, AP, and AF exactly as Equations in Section 5.3.
- Accept task dictionaries with `x`, `edge_index` or `adjacency`, optional `y`, and optional `eval_mask`.
- Evaluate each seen task after each task and save `performance_matrix.csv`.

### Task 3: Paper-aligned sequential loop and ablations

**Files:** Modify `experiments/train_sequential.py` and `experiments/member_4_ablation.py`.

- Combine the current task with sampled historical replay tasks before two stochastic augmentations.
- Keep teacher plans detached and record `L_mat`, `L_str`, `L_skd`, and total loss.
- Expose `w/o Cross-task Distillation`, plan-level cosine, and plan-level MSE variants.

### Task 4: Verification and integration

**Files:** Modify `README.md`, `configs/example.yaml`, `.gitignore` if needed.

- Run static compilation and dependency-aware tests.
- Run the synthetic end-to-end smoke test when PyTorch is available.
- Commit and push the implementation when network access permits.
