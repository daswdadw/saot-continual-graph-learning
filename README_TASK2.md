# Task 2: OT / GW / FGW theory and solver

This directory contains the independent member-2 deliverables for the SAOT
reproduction:

- `models/ot_solver.py`: NumPy implementation of log-domain Sinkhorn OT and
  entropic FGW with the paper's feature-weight convention;
- `experiments/toy_ot_alignment.py`: reproducible eight-node alignment study;
- `results/toy_ot/`: raw matrices, iteration logs, sigma ablation, and heatmaps;
- `reports/task2_ot_fgw_theory.md`: report Chapter 3;
- `slides/SAOT_Task2_OT_FGW.pptx`: four-slide English presentation section;
- `tests/test_ot_solver.py`: marginal, permutation, and sigma-sensitivity tests.

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m experiments.toy_ot_alignment --output-dir results/toy_ot
```

The experiment uses only NumPy and Pillow and does not depend on the full SAOT
training code.
