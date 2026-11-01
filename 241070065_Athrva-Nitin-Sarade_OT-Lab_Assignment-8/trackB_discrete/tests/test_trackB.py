"""
Unit Tests for Track B: Discrete Optimization
"""

import numpy as np
import pytest
from trackB_discrete.run import (
    knapsack_brute_force,
    knapsack_dp,
    knapsack_ilp,
    KnapsackBranchAndBound,
    generate_random_graph,
    maxcut_to_qubo,
    qubo_to_ising,
    maxcut_brute_force,
    maxcut_objective
)

def test_knapsack_exact_methods_agree():
    values = np.array([20, 35, 12, 40, 50])
    weights = np.array([5, 10, 3, 12, 15])
    capacity = 25

    v_bf, _ = knapsack_brute_force(values, weights, capacity)
    v_dp, _ = knapsack_dp(values, weights, capacity)
    v_ilp, _ = knapsack_ilp(values, weights, capacity)
    bb = KnapsackBranchAndBound(values, weights, capacity)
    v_bb, _ = bb.solve()

    assert v_bf == v_dp == v_ilp == v_bb, f"Discrepancy among exact solvers: BF={v_bf}, DP={v_dp}, ILP={v_ilp}, BB={v_bb}"

def test_qubo_ising_consistency():
    adj, _ = generate_random_graph(n_nodes=5, edge_prob=0.5, seed=42)
    Q = maxcut_to_qubo(adj)
    J, h, offset = qubo_to_ising(Q)

    # For any binary vector x, compute x^T Q x
    for i in range(1 << 5):
        x = np.array([(i >> j) & 1 for j in range(5)])
        qubo_val = float(x.T @ Q @ x)

        # Map to spins: s_i = 1 - 2*x_i
        s = 1 - 2 * x
        # Ising energy: s^T J s + h^T s + offset
        ising_val = float(s.T @ J @ s + h.T @ s + offset)

        assert np.isclose(qubo_val, ising_val, atol=1e-8), f"QUBO energy {qubo_val} != Ising energy {ising_val}"

def test_maxcut_brute_force_optimal():
    # Triangle graph (3 nodes, 3 edges): optimal cut is 2 edges
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    cut_val, partition = maxcut_brute_force(adj)
    assert cut_val == 2.0, f"Max-Cut on triangle should be 2, got {cut_val}"
    assert maxcut_objective(np.array(partition), adj) == 2.0
