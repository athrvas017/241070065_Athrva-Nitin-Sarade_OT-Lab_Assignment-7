"""
Unit Tests for Track J: Software Engineering Optimization
"""

import pytest
from trackJ_software_opt.run import (
    TEST_SUITE,
    ALL_REACHABLE_LINES,
    select_tests_greedy,
    select_tests_ilp,
    profile_and_optimize
)

def test_coverage_100_percent_preserved():
    tests_greedy, time_greedy, cov_greedy = select_tests_greedy(TEST_SUITE)
    tests_ilp, time_ilp, cov_ilp = select_tests_ilp(TEST_SUITE)

    assert cov_greedy == ALL_REACHABLE_LINES, "Greedy failed to preserve all reachable statement lines"
    assert cov_ilp == ALL_REACHABLE_LINES, "ILP failed to preserve all reachable statement lines"

def test_ilp_time_optimality():
    _, time_greedy, _ = select_tests_greedy(TEST_SUITE)
    _, time_ilp, _ = select_tests_ilp(TEST_SUITE)

    assert time_ilp <= time_greedy + 1e-5, f"ILP optimal time {time_ilp} exceeded greedy time {time_greedy}"

def test_profiling_optimization_gain():
    res = profile_and_optimize()
    assert res["speedup"] > 1.0, f"Profiling optimization failed to demonstrate speedup: {res['speedup']}x"
