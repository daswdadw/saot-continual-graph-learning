# SAOT Repository Audit Report

Audit date: 2026-11-17  
Remote repository: `https://github.com/daswdadw/saot-continual-graph-learning`  
Remote snapshot: `2878c8f` (`main`)  
Local snapshot: `2194d7b` (`main`, four commits ahead of the recorded remote ref)

## Audit copy

This directory is an independent audit copy. It was created from the remote `main` branch and then overlaid with the local member-4 implementation. The original directories remain unchanged:

- Remote-only checkout: `saot-continual-graph-learning-remote`
- Local checkout: `saot-continual-graph-learning`
- Combined audit checkout: `saot-continual-graph-learning-complete-audit`

The combined copy contains the remote members' data preparation, NumPy FGW solver, toy OT experiment, reports, and experiment scripts, together with the local SAOT model, transport losses, replay buffer, sequential trainer, metrics, and member-4 tests.

## Verification evidence

| Check | Result | Meaning |
|---|---|---|
| Python `compileall` over the combined copy | Pass | Python syntax is valid in the current environment |
| NumPy OT unit tests | 3/3 pass | Sinkhorn marginals, permutation recovery, and sigma sensitivity work on toy inputs |
| Toy OT experiment | Runs | FGW plans and artifacts are generated |
| Full unit-test discovery | Blocked | PyTorch is not installed in the current environment, so member-4 tests cannot import |
| Remote experiment imports | Initially failed | The remote branch referenced missing `utils.*` modules |
| Combined audit imports | Static imports resolved | Minimal runtime utility modules were added to this audit copy |

The toy experiment reached the maximum Sinkhorn iteration count for `pi_Z` with marginal error about `5e-6`. The correspondence was correct, but the tolerance and iteration settings should be revisited before using it as a numerical benchmark.

## Blocking findings before a paper-faithful run

### 1. Missing runtime modules

The remote branch referenced missing `utils.config_io`, `utils.paths`, `utils.results_io`, `utils.training_client`, `utils.variants`, `utils.adaptive_beta`, and `utils.plotting`. The audit copy contains minimal implementations for those contracts; they have not yet been pushed to GitHub.

### 2. Task data is not connected to the trainer

`data/prepare_cgl.py` writes JSON task metadata with node IDs and train/validation/test splits. `experiments/train_sequential.py` expects a PyTorch file containing task tensors named `x` and either `adjacency` or `edge_index`. A conversion step and checksum are still required.

### 3. The evaluator required a paper-protocol upgrade

The audit copy now includes `evaluation/linear_probe.py` and the sequential trainer calls it at every task checkpoint. The encoder is frozen while a linear classifier is trained on the task training masks. The original nearest-centroid path remains in the file only as legacy helper code and is not used by the updated run loop. The linear probe still requires PyTorch verification on AutoDL.

### 4. Only part of the benchmark setup is implemented

The checked-in data preparation supports CoraFull-CL and Arxiv-CL. The paper also reports Reddit-CL and Products-CL. The checked-in model has a two-layer GCN path; the paper uses a three-layer, four-head GAT path for Reddit-CL.

### 5. Paper settings are not enforced by one runner

The paper uses hidden dimension 512 for CoraFull, Arxiv, and Products, replay budget 200 for CoraFull and 800 for the other large datasets, a learning-rate grid from `1e-4` to `1e-2`, and five random seeds. The default configuration uses hidden dimension 256 and does not provide a complete five-seed linear-probe result aggregator.

### 6. Replay edge-index boundary bug

The original replay implementation could index a compact mapping with an edge endpoint outside the sampled-node range. The audit copy contains a regression test and a search-based induced-edge implementation that handles global IDs safely.

## Interpretation

The repository contains substantial implementation work, and the NumPy OT component is independently testable. It is not yet evidence of a complete SAOT reproduction. The first AutoDL run should be a staged integration run on a small synthetic task file, followed by CoraFull-CL with the exact task protocol and linear-probe evaluation. Arxiv-CL is a second-stage scale test. Reddit-CL and Products-CL require additional data/model work before a four-dataset claim.
