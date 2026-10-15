# SAOT Repository Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a local Git repository for the SAOT ICML 2026 course project with the paper, six-person file-level ownership rules, reproducible configuration, and empty implementation directories.

**Architecture:** The repository is organized by responsibility: data preparation, models/losses, baselines/evaluation, experiments/results, and documentation. The six-person work packages map directly to files and directories documented in `docs/SAOT_六人分工与实际成果.md`.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, POT/GeomLoss as needed, YAML configuration, Git.

**Spec:** `docs/SAOT_六人分工与实际成果.md`

## Global Constraints

- Use CoraFull-CL as the first end-to-end dataset; add Arxiv-CL after the pipeline is stable.
- Every member must commit code or data, one independent experiment, raw logs, a report section, and slides.
- All runs must save `config.yaml`, `metrics.json`, `performance_matrix.csv`, `train.log`, and checkpoints under `results/<model>/<dataset>/<scenario>/seed_<seed>/`.
- Do not commit downloaded datasets, generated checkpoints, or large experiment logs; use the documented ignored directories.

---

### Task 1: Create repository scaffold

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/example.yaml`
- Create: `docs/SAOT_六人分工与实际成果.md`
- Create: `docs/paper/ICML2026_66335_SAOT.pdf`
- Create: `docs/superpowers/plans/2026-10-15-saot-repository.md`
- Create: `.gitkeep` files under `data/`, `models/`, `losses/`, `baselines/`, `evaluation/`, `experiments/`, and `results/`

**Interfaces:**
- `README.md` points contributors to the six-person ownership document.
- The ownership document defines exact files and acceptance commands for each member.
- `configs/example.yaml` is the shared configuration template.

- [ ] Create directories and placeholder files.
- [ ] Copy the verified SAOT PDF and the expanded ownership document.
- [ ] Write README, ignore rules, dependency template, and example configuration.
- [ ] Initialize Git and verify `git status`.
- [ ] Commit the scaffold as `chore: initialize SAOT project repository`.

### Task 2: Verify repository handoff

**Files:**
- Verify: `README.md`
- Verify: `docs/SAOT_六人分工与实际成果.md`
- Verify: `docs/paper/ICML2026_66335_SAOT.pdf`

**Interfaces:**
- A new contributor can identify their assigned files without reading chat history.
- The paper opens as a valid PDF and the repository has no generated artifacts tracked.

- [ ] Run the documented configuration and tree checks.
- [ ] Confirm the PDF magic bytes begin with `%PDF-`.
- [ ] Confirm the six ownership sections each list files, outputs, and acceptance criteria.
- [ ] Report the repository path and initial commit hash.
