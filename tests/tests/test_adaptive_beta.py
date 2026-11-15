import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.adaptive_beta import (
    AdaptiveBetaConfig,
    betas_for_task_sequence,
    compute_adaptive_beta,
    prototype_cosine_shift,
    schedule_betas_for_tasks,
)


def test_shift_increases_when_prototypes_differ():
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=(100, 16))
    y1 = np.zeros(100, dtype=int)
    f2 = rng.normal(size=(100, 16)) + 3.0
    y2 = np.ones(100, dtype=int)
    s = prototype_cosine_shift(f1, y1, f2, y2)
    assert 0.0 <= s <= 1.0
    assert s > 0.5


def test_beta_monotonic_with_shift():
    cfg = AdaptiveBetaConfig(beta_min=0.2, beta_max=1.0)
    b_low = compute_adaptive_beta(0.1, cfg)
    b_high = compute_adaptive_beta(0.9, cfg)
    assert b_high > b_low


def test_schedule_length():
    cfg = AdaptiveBetaConfig()
    shifts = [0.2, 0.5, 0.8]
    betas = betas_for_task_sequence(shifts, cfg)
    assert len(betas) == len(shifts) + 1
    assert betas[0] == cfg.beta_min
    assert len(schedule_betas_for_tasks(shifts, cfg)) == 3
