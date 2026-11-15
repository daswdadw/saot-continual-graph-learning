import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.adaptive_beta import shifts_from_prototype_dir


def test_shifts_from_prototype_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rng = np.random.default_rng(1)
        for t in range(3):
            f = rng.normal(size=(50, 8)) + t * 2
            y = np.full(50, t % 2)
            np.savez(d / f"task_{t:02d}.npz", features=f, labels=y)
        shifts = shifts_from_prototype_dir(d)
        assert len(shifts) == 2
        assert all(0 <= s <= 1 for s in shifts)
