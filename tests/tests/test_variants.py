import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.variants import ABLATION_VARIANTS, variant_to_overrides


def test_each_variant_differs_from_full_in_one_axis():
    full = variant_to_overrides("SAOT (ours)")
    for v in ABLATION_VARIANTS:
        if v == "SAOT (ours)":
            continue
        cur = variant_to_overrides(v)
        assert cur != full


def test_no_ot_variant():
    cur = variant_to_overrides("w/o Optimal Transport")
    assert cur["use_ot"] is False
    assert cur["ot_replacement"] == "graph_barlow_twins"
