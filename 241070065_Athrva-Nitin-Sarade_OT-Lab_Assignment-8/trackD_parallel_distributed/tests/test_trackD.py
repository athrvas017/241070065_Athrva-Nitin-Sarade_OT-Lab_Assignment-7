"""
Unit Tests for Track D: Submodular, Parallel, and Distributed
"""

import numpy as np
import pytest
from trackD_parallel_distributed.run import (
    FacilityLocationCoverage,
    FedAvgSimulation,
    simulate_pipeline_parallelism
)

def test_submodularity_diminishing_returns():
    fl = FacilityLocationCoverage(n_items=10, seed=42)
    assert fl.verify_diminishing_returns(n_trials=25), "Diminishing returns condition failed on facility location"

def test_greedy_submodular_approximation_bound():
    fl = FacilityLocationCoverage(n_items=10, seed=42)
    k = 3
    val_bf, _ = fl.brute_force_select(k=k)
    val_greedy, _ = fl.greedy_select(k=k)

    ratio = val_greedy / val_bf
    theoretical_bound = 1.0 - 1.0 / np.e
    assert ratio >= theoretical_bound - 1e-6, f"Greedy ratio {ratio} lower than (1 - 1/e) = {theoretical_bound}"

def test_pipeline_bubble_model():
    model = simulate_pipeline_parallelism(n_stages=4, n_microbatches=12)
    expected_bubble = (4 - 1) / (12 + 4 - 1)  # 3 / 15 = 0.20
    assert np.isclose(model["bubble_fraction"], expected_bubble), f"Pipeline bubble {model['bubble_fraction']} != {expected_bubble}"
    assert model["ideal_speedup"] > 1.0, "Pipeline speedup must exceed 1"
