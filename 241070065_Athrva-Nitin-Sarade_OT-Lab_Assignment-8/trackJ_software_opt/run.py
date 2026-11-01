"""
Track J: Software Engineering Optimization — Test Suite Minimization & Profiling
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from scipy.optimize import milp, LinearConstraint, Bounds

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Target Codebase & Synthetic Test Suite Coverage Profile
# -------------------------------------------------------------

# Suppose we have a codebase module with 25 critical statement lines [1..25]
# and a test suite of 10 test cases with diverse execution times and coverage sets.

TEST_SUITE = [
    {"name": "test_auth_login", "time_ms": 45.0, "lines": {1, 2, 3, 4, 5}},
    {"name": "test_auth_token_refresh", "time_ms": 35.0, "lines": {3, 4, 6, 7}},
    {"name": "test_db_query_simple", "time_ms": 80.0, "lines": {8, 9, 10, 11}},
    {"name": "test_db_transaction_commit", "time_ms": 110.0, "lines": {9, 10, 11, 12, 13}},
    {"name": "test_api_crud_user", "time_ms": 65.0, "lines": {1, 2, 8, 9, 14, 15}},
    {"name": "test_api_pagination", "time_ms": 40.0, "lines": {14, 15, 16, 17}},
    {"name": "test_crypto_hash_verify", "time_ms": 95.0, "lines": {18, 19, 20}},
    {"name": "test_crypto_hmac_sign", "time_ms": 85.0, "lines": {19, 20, 21, 22}},
    {"name": "test_cache_lru_evict", "time_ms": 30.0, "lines": {23, 24, 25}},
    {"name": "test_system_healthcheck", "time_ms": 25.0, "lines": {1, 8, 14, 23}}
]

ALL_REACHABLE_LINES = set.union(*[t["lines"] for t in TEST_SUITE])

# -------------------------------------------------------------
# 2. Test Suite Minimization Formulations
# -------------------------------------------------------------

def select_tests_greedy(test_suite: List[Dict[str, any]]) -> Tuple[List[str], float, Set[int]]:
    """
    Greedy set cover: repeatedly pick the test maximizing (uncovered lines covered / test execution time).
    Guarantees 100% preservation of all reachable lines.
    """
    uncovered = set(ALL_REACHABLE_LINES)
    selected_tests = []
    total_time = 0.0

    while uncovered:
        best_idx = -1
        best_ratio = -1.0
        best_new_lines = set()

        for idx, t in enumerate(test_suite):
            if t["name"] not in selected_tests:
                new_lines = t["lines"] & uncovered
                if len(new_lines) > 0:
                    ratio = len(new_lines) / t["time_ms"]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = idx
                        best_new_lines = new_lines

        if best_idx == -1:
            break

        chosen = test_suite[best_idx]
        selected_tests.append(chosen["name"])
        total_time += chosen["time_ms"]
        uncovered -= best_new_lines

    covered = set.union(*[t["lines"] for t in test_suite if t["name"] in selected_tests])
    return selected_tests, total_time, covered

def select_tests_ilp(test_suite: List[Dict[str, any]]) -> Tuple[List[str], float, Set[int]]:
    """
    Exact Minimum Set Cover via ILP (scipy.optimize.milp):
    min sum_{t=1}^n Time(t) * x_t
    s.t. sum_{t covers line l} x_t >= 1  for all l in ALL_REACHABLE_LINES
         x_t in {0, 1}
    """
    n_tests = len(test_suite)
    lines_list = sorted(list(ALL_REACHABLE_LINES))
    n_lines = len(lines_list)

    # Cost vector: execution time per test
    c = np.array([t["time_ms"] for t in test_suite])

    # Constraint matrix A: (n_lines, n_tests)
    A = np.zeros((n_lines, n_tests))
    for line_idx, line in enumerate(lines_list):
        for test_idx, t in enumerate(test_suite):
            if line in t["lines"]:
                A[line_idx, test_idx] = 1.0

    # Lower bound is 1 (each line covered at least once), upper bound is infinity
    constraints = LinearConstraint(A, 1.0, np.inf)
    integrality = np.ones(n_tests)
    bounds = Bounds(0, 1)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError("ILP failed on test suite minimization")

    sol = np.round(res.x).astype(int)
    selected_tests = [test_suite[i]["name"] for i in range(n_tests) if sol[i] == 1]
    total_time = float(res.fun)
    covered = set.union(*[test_suite[i]["lines"] for i in range(n_tests) if sol[i] == 1])

    return selected_tests, total_time, covered

# -------------------------------------------------------------
# 3. cProfile Bottleneck Profiling & Optimization Demo
# -------------------------------------------------------------

def slow_string_concatenation(n: int = 4000) -> str:
    """Bottleneck: quadratic string concatenation."""
    s = ""
    for i in range(n):
        s += str(i) + ","
    return s

def fast_string_concatenation(n: int = 4000) -> str:
    """Optimized: pre-allocated list join in O(n) time."""
    parts = [str(i) for i in range(n)]
    return ",".join(parts) + ","

def profile_and_optimize() -> Dict[str, any]:
    # Profile slow implementation
    pr_slow = cProfile.Profile()
    pr_slow.enable()
    t0 = time.perf_counter()
    _ = slow_string_concatenation(5000)
    t_slow = (time.perf_counter() - t0) * 1000.0
    pr_slow.disable()

    # Profile fast implementation
    pr_fast = cProfile.Profile()
    pr_fast.enable()
    t0 = time.perf_counter()
    _ = fast_string_concatenation(5000)
    t_fast = (time.perf_counter() - t0) * 1000.0
    pr_fast.disable()

    speedup = t_slow / max(t_fast, 1e-4)
    return {
        "slow_time_ms": round(t_slow, 2),
        "fast_time_ms": round(t_fast, 2),
        "speedup": round(speedup, 2)
    }

# -------------------------------------------------------------
# 4. Main Execution & Comparison
# -------------------------------------------------------------

def run_software_opt_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track J: Software Engineering Optimization ===")

    # 1. Baseline Full Suite
    total_baseline_time = sum(t["time_ms"] for t in TEST_SUITE)
    total_lines = len(ALL_REACHABLE_LINES)
    print(f"Full Test Suite (Baseline): {len(TEST_SUITE)} tests, {total_baseline_time:.1f} ms, covers {total_lines}/{total_lines} lines (100%)")

    # 2. Greedy Minimization
    tests_greedy, time_greedy, cov_greedy = select_tests_greedy(TEST_SUITE)
    cov_greedy_pct = (len(cov_greedy) / total_lines) * 100
    time_saved_greedy = ((total_baseline_time - time_greedy) / total_baseline_time) * 100
    print(f"Greedy Minimization: {len(tests_greedy)} tests ({tests_greedy}), {time_greedy:.1f} ms ({time_saved_greedy:.1f}% time saved, {cov_greedy_pct:.1f}% coverage preserved)")

    # 3. ILP Exact Minimization
    tests_ilp, time_ilp, cov_ilp = select_tests_ilp(TEST_SUITE)
    cov_ilp_pct = (len(cov_ilp) / total_lines) * 100
    time_saved_ilp = ((total_baseline_time - time_ilp) / total_baseline_time) * 100
    print(f"ILP Exact Minimization: {len(tests_ilp)} tests ({tests_ilp}), {time_ilp:.1f} ms ({time_saved_ilp:.1f}% time saved, {cov_ilp_pct:.1f}% coverage preserved)")

    # Assert 100% statement coverage preservation
    assert cov_greedy == ALL_REACHABLE_LINES, "Greedy failed to preserve 100% statement coverage!"
    assert cov_ilp == ALL_REACHABLE_LINES, "ILP failed to preserve 100% statement coverage!"
    assert time_ilp <= time_greedy + 1e-6, "ILP must find optimal or equal time compared to greedy!"

    comparison_table = [
        {"Method": "Baseline (All Tests)", "Test Count": len(TEST_SUITE), "Runtime (ms)": total_baseline_time, "Time Reduction (%)": 0.0, "Coverage Preserved (%)": 100.0},
        {"Method": "Greedy Set Cover", "Test Count": len(tests_greedy), "Runtime (ms)": time_greedy, "Time Reduction (%)": round(time_saved_greedy, 1), "Coverage Preserved (%)": cov_greedy_pct},
        {"Method": "ILP Exact Set Cover", "Test Count": len(tests_ilp), "Runtime (ms)": time_ilp, "Time Reduction (%)": round(time_saved_ilp, 1), "Coverage Preserved (%)": cov_ilp_pct}
    ]
    results["minimization_comparison"] = comparison_table

    # 4. cProfile Optimization
    prof_res = profile_and_optimize()
    print(f"cProfile Algorithmic Optimization: Slow={prof_res['slow_time_ms']} ms, Fast={prof_res['fast_time_ms']} ms, Speedup={prof_res['speedup']}x")
    results["profiling"] = prof_res

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot 1: Test Suite Minimization (Time vs Test Count)
    labels = ["All Tests", "Greedy", "ILP Exact"]
    times = [total_baseline_time, time_greedy, time_ilp]
    counts = [len(TEST_SUITE), len(tests_greedy), len(tests_ilp)]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, times, width, label="Execution Time (ms)", color="indianred")
    axes[0].bar(x + width/2, [c * 50 for c in counts], width, label="Test Count (scaled x50)", color="cornflowerblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Test Suite Minimization (100% Coverage Preserved)")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: Cumulative Statement Coverage Growth
    # Order tests by selection in greedy
    sorted_order = tests_greedy
    cum_lines = set()
    cov_progress = [0]
    time_progress = [0]
    for t_name in sorted_order:
        t_obj = [t for t in TEST_SUITE if t["name"] == t_name][0]
        cum_lines |= t_obj["lines"]
        cov_progress.append(len(cum_lines))
        time_progress.append(time_progress[-1] + t_obj["time_ms"])

    axes[1].step(time_progress, cov_progress, where='post', color="forestgreen", lw=2.5, label="Cumulative Coverage")
    axes[1].axhline(total_lines, color="crimson", linestyle="--", label=f"Total Lines ({total_lines})")
    axes[1].set_xlabel("Elapsed Test Time (ms)")
    axes[1].set_ylabel("Statement Lines Covered")
    axes[1].set_title("Marginal Coverage Gain per Test")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackJ_software_opt.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["software_opt_plot"] = plot_path

    print("=== Track J Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_software_opt_experiments()
    print(f"Track J Runtime: {time.time() - t0:.2f}s")
