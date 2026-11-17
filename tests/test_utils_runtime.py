import json
from pathlib import Path

import numpy as np

from utils.adaptive_beta import (
    AdaptiveBetaConfig,
    betas_for_task_sequence,
    compute_adaptive_beta,
    prototype_cosine_shift,
    schedule_betas_for_tasks,
)
from utils.config_io import merge_config
from utils.paths import ablation_run_dir, slugify, standard_run_dir
from utils.results_io import aggregate_runs
from utils.variants import ABLATION_VARIANTS, variant_to_overrides


def test_adaptive_beta_contract_is_monotonic_and_bounded():
    cfg = AdaptiveBetaConfig(beta_min=0.2, beta_max=1.0)
    assert compute_adaptive_beta(0.1, cfg) < compute_adaptive_beta(0.9, cfg)
    values = betas_for_task_sequence([0.2, 0.5, 0.8], cfg)
    assert len(values) == 4
    assert values[0] == cfg.beta_min
    assert len(schedule_betas_for_tasks([0.2, 0.5, 0.8], cfg)) == 3


def test_prototype_shift_detects_disjoint_centroids():
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=(100, 16))
    f2 = rng.normal(size=(100, 16)) + 3.0
    assert prototype_cosine_shift(f1, np.zeros(100), f2, np.ones(100)) > 0.5


def test_runtime_helpers_provide_stable_paths_and_aggregation():
    assert slugify("CoraFull-CL") == "corafull_cl"
    assert standard_run_dir(Path("results"), "saot", "CoraFull-CL", "class-il", 42).as_posix().endswith("saot/corafull_cl/class-il/seed_42")
    assert ablation_run_dir(Path("results"), "SAOT (ours)", 42).name == "seed_42"
    merged = merge_config({"a": 1, "nested": {"x": 1}}, {"nested": {"y": 2}})
    assert merged == {"a": 1, "nested": {"x": 1, "y": 2}}
    summary = aggregate_runs(
        [{"variant": "SAOT", "AP": 0.8, "AF": -0.1}, {"variant": "SAOT", "AP": 0.9, "AF": -0.2}],
        ["variant"],
    )
    assert float(summary.loc[0, "AP_mean"]) == 0.85


def test_variant_contract_keeps_single_component_changes():
    full = variant_to_overrides("SAOT (ours)")
    assert "SAOT (ours)" in ABLATION_VARIANTS
    assert variant_to_overrides("w/o Optimal Transport")["use_ot"] is False
    assert variant_to_overrides("w/o Optimal Transport")["ot_replacement"] == "graph_barlow_twins"
    assert variant_to_overrides("w/o Cross-task Distillation")["use_cross_task_distillation"] is False
    assert full["use_ot"] is True
