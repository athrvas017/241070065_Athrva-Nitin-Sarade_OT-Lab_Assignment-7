"""
Unit Tests for Track I: Quantum Algorithms and QML
"""

import pytest
from trackI_quantum_algos_qml.run import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    toy_shor_factoring_15
)

def test_bernstein_vazirani_exact_recovery():
    for test_string in ["101", "110", "011"]:
        res = run_bernstein_vazirani(secret_string=test_string)
        assert res["exact_match"], f"Bernstein-Vazirani failed for string {test_string}"
        assert res["success_prob"] > 0.99

def test_deutsch_jozsa_classification():
    res = run_deutsch_jozsa()
    assert res["balanced_classified"], "Failed to classify balanced oracle"
    assert res["constant_classified"], "Failed to classify constant oracle"

def test_shors_order_finding_factors_15():
    res = toy_shor_factoring_15()
    assert res["success"]
    assert sorted(res["derived_factors"]) == [3, 5]
