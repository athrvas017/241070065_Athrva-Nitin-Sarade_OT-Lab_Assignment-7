"""
Track B: Discrete Optimization
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Optional

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Knapsack Problem Formulations & Algorithms
# -------------------------------------------------------------

def knapsack_brute_force(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, List[int]]:
    n = len(values)
    best_val = 0
    best_comb = []
    for i in range(1 << n):
        total_w = sum(weights[j] for j in range(n) if (i >> j) & 1)
        if total_w <= capacity:
            total_v = sum(values[j] for j in range(n) if (i >> j) & 1)
            if total_v > best_val:
                best_val = total_v
                best_comb = [(i >> j) & 1 for j in range(n)]
    return best_val, best_comb

def knapsack_greedy(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, List[int]]:
    """Greedy based on value-to-weight ratio."""
    n = len(values)
    ratios = values / weights
    order = np.argsort(-ratios)
    total_w = 0
    total_v = 0
    chosen = [0] * n
    for idx in order:
        if total_w + weights[idx] <= capacity:
            chosen[idx] = 1
            total_w += weights[idx]
            total_v += values[idx]
    return total_v, chosen

def knapsack_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, List[int]]:
    """Dynamic programming table O(n * W)."""
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    for i in range(1, n + 1):
        v = values[i - 1]
        w = weights[i - 1]
        for c in range(capacity + 1):
            if c >= w:
                dp[i, c] = max(dp[i - 1, c], dp[i - 1, c - w] + v)
            else:
                dp[i, c] = dp[i - 1, c]
    
    # Backtrack chosen items
    chosen = [0] * n
    curr_c = capacity
    for i in range(n, 0, -1):
        if dp[i, curr_c] != dp[i - 1, curr_c]:
            chosen[i - 1] = 1
            curr_c -= weights[i - 1]
    return int(dp[n, capacity]), chosen

def knapsack_ilp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, List[int]]:
    """Exact ILP using scipy.optimize.milp."""
    n = len(values)
    c = -values.astype(float)  # scipy minimizes c^T x
    A = weights.reshape(1, n).astype(float)
    constraints = LinearConstraint(A, 0, capacity)
    integrality = np.ones(n)  # 1 = integer variable
    bounds = Bounds(0, 1)     # binary 0 or 1
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError("ILP failed to solve knapsack")
    chosen = np.round(res.x).astype(int).tolist()
    best_val = int(np.round(-res.fun))
    return best_val, chosen

class KnapsackBranchAndBound:
    """Hand-written Branch-and-Bound with fractional LP relaxation upper bounds."""
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: int):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.n = len(values)
        self.best_val = 0
        self.best_sol = [0] * self.n

    def _fractional_bound(self, idx: int, curr_w: int, curr_v: int) -> float:
        if curr_w > self.capacity:
            return 0.0
        bound = curr_v
        rem_cap = self.capacity - curr_w
        for i in range(idx, self.n):
            if self.weights[i] <= rem_cap:
                rem_cap -= self.weights[i]
                bound += self.values[i]
            else:
                bound += self.values[i] * (rem_cap / self.weights[i])
                break
        return bound

    def solve(self) -> Tuple[int, List[int]]:
        # Sort items by value/weight ratio
        ratios = self.values / self.weights
        order = np.argsort(-ratios)
        v_sorted = self.values[order]
        w_sorted = self.weights[order]

        def search(idx: int, curr_w: int, curr_v: int, current_sol: List[int]):
            if curr_w > self.capacity:
                return
            if curr_v > self.best_val:
                self.best_val = curr_v
                # map back to original indices
                sol_orig = [0] * self.n
                for s_i, orig_i in enumerate(order[:len(current_sol)]):
                    sol_orig[orig_i] = current_sol[s_i]
                self.best_sol = sol_orig

            if idx >= self.n:
                return

            bound = self._fractional_bound(idx, curr_w, curr_v)
            if bound <= self.best_val:
                return  # Prune branch

            # Branch 1: Include item idx
            if curr_w + w_sorted[idx] <= self.capacity:
                search(idx + 1, curr_w + w_sorted[idx], curr_v + v_sorted[idx], current_sol + [1])
            # Branch 2: Exclude item idx
            search(idx + 1, curr_w, curr_v, current_sol + [0])

        search(0, 0, 0, [])
        return self.best_val, self.best_sol

# -------------------------------------------------------------
# 2. Max-Cut, QUBO, and Ising Spin Formulation
# -------------------------------------------------------------

def generate_random_graph(n_nodes: int = 10, edge_prob: float = 0.4, seed: int = SEED) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n_nodes, n_nodes))
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < edge_prob:
                weight = 1.0  # unweighted or unit weight
                adj[i, j] = weight
                adj[j, i] = weight
                edges.append((i, j, weight))
    # Ensure graph is connected or has edges
    if len(edges) == 0:
        edges.append((0, 1, 1.0))
        adj[0, 1] = adj[1, 0] = 1.0
    return adj, edges

def maxcut_objective(x: np.ndarray, adj: np.ndarray) -> float:
    """Cut weight: sum_{(i,j) in E} w_ij * (x_i + x_j - 2 x_i x_j)."""
    n = len(x)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                cut += adj[i, j] * (x[i] + x[j] - 2 * x[i] * x[j])
    return cut

def maxcut_to_qubo(adj: np.ndarray, cardinality_k: Optional[int] = None, penalty_lambda: float = 5.0) -> np.ndarray:
    """
    Construct QUBO matrix Q such that min x^T Q x corresponds to:
    min (-MaxCut(x) + lambda * (sum x_i - k)^2).
    """
    n = adj.shape[0]
    Q = np.zeros((n, n))
    deg = np.sum(adj, axis=1)

    # Max-Cut minimization: - Cut(x)
    # Cut(x) = sum_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)
    # x_i^2 = x_i for binary, so diag(Q)_i = - deg_i, Q_ij = 2 * w_ij
    for i in range(n):
        Q[i, i] -= deg[i]
        for j in range(n):
            if i != j:
                Q[i, j] += adj[i, j]

    # Optional cardinality constraint: penalty_lambda * (sum x_i - k)^2
    if cardinality_k is not None:
        for i in range(n):
            Q[i, i] += penalty_lambda * (1.0 - 2.0 * cardinality_k)
            for j in range(n):
                if i != j:
                    Q[i, j] += penalty_lambda
    return Q

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert binary x in {0, 1}^n to Ising spin s in {-1, +1}^n via x_i = (1 - s_i) / 2.
    Returns: (J coupling matrix, h local fields, offset constant).
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    for i in range(n):
        for j in range(n):
            q_ij = Q[i, j]
            # x_i x_j = (1 - s_i)(1 - s_j)/4 = (1 - s_i - s_j + s_i s_j)/4
            offset += q_ij / 4.0
            h[i] -= q_ij / 4.0
            h[j] -= q_ij / 4.0
            J[i, j] += q_ij / 4.0
    return J, h, offset

def maxcut_brute_force(adj: np.ndarray) -> Tuple[float, List[int]]:
    n = adj.shape[0]
    best_cut = -1.0
    best_x = []
    for i in range(1 << n):
        x = np.array([(i >> j) & 1 for j in range(n)])
        c = maxcut_objective(x, adj)
        if c > best_cut:
            best_cut = c
            best_x = x.tolist()
    return best_cut, best_x

def maxcut_simulated_annealing(Q: np.ndarray, iters: int = 1500, t_init: float = 10.0, alpha: float = 0.995) -> Tuple[float, List[int]]:
    """Solves min x^T Q x using Simulated Annealing."""
    n = Q.shape[0]
    x = np.random.randint(0, 2, size=n)
    current_energy = float(x.T @ Q @ x)
    best_x = x.copy()
    best_energy = current_energy

    T = t_init
    for step in range(iters):
        # Pick random flip
        flip_idx = np.random.randint(0, n)
        x_new = x.copy()
        x_new[flip_idx] = 1 - x_new[flip_idx]
        new_energy = float(x_new.T @ Q @ x_new)
        delta_E = new_energy - current_energy

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / max(T, 1e-6)):
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        T *= alpha
    return best_energy, best_x.tolist()

# -------------------------------------------------------------
# 3. Execution & Validation
# -------------------------------------------------------------

def run_discrete_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track B: Discrete Optimization ===")

    # 1. Knapsack Evaluation
    # Seeded instance of 12 items
    rng = np.random.RandomState(SEED)
    values = rng.randint(10, 100, size=12)
    weights = rng.randint(5, 30, size=12)
    capacity = int(0.5 * np.sum(weights))

    print(f"Knapsack: 12 items, Capacity = {capacity}")
    
    t0 = time.time()
    val_bf, sol_bf = knapsack_brute_force(values, weights, capacity)
    t_bf = time.time() - t0

    t0 = time.time()
    val_greedy, sol_greedy = knapsack_greedy(values, weights, capacity)
    t_greedy = time.time() - t0

    t0 = time.time()
    val_dp, sol_dp = knapsack_dp(values, weights, capacity)
    t_dp = time.time() - t0

    t0 = time.time()
    val_ilp, sol_ilp = knapsack_ilp(values, weights, capacity)
    t_ilp = time.time() - t0

    t0 = time.time()
    bb = KnapsackBranchAndBound(values, weights, capacity)
    val_bb, sol_bb = bb.solve()
    t_bb = time.time() - t0

    knapsack_comparison = [
        {"Method": "Brute Force", "Value": val_bf, "Weight": int(sum(weights[i]*sol_bf[i] for i in range(12))), "Time_ms": t_bf*1000},
        {"Method": "Greedy (Value/Weight)", "Value": val_greedy, "Weight": int(sum(weights[i]*sol_greedy[i] for i in range(12))), "Time_ms": t_greedy*1000},
        {"Method": "Dynamic Programming", "Value": val_dp, "Weight": int(sum(weights[i]*sol_dp[i] for i in range(12))), "Time_ms": t_dp*1000},
        {"Method": "Branch and Bound", "Value": val_bb, "Weight": int(sum(weights[i]*sol_bb[i] for i in range(12))), "Time_ms": t_bb*1000},
        {"Method": "ILP (scipy.milp)", "Value": val_ilp, "Weight": int(sum(weights[i]*sol_ilp[i] for i in range(12))), "Time_ms": t_ilp*1000},
    ]
    results["knapsack_comparison"] = knapsack_comparison
    print(f"Knapsack Results: BF={val_bf}, DP={val_dp}, B&B={val_bb}, ILP={val_ilp}, Greedy={val_greedy}")

    # Assert exact methods match
    assert val_bf == val_dp == val_bb == val_ilp, "Knapsack exact solvers must match!"
    greedy_gap = (val_bf - val_greedy) / val_bf
    print(f"Greedy Approximation Gap: {greedy_gap * 100:.2f}%")
    results["greedy_gap_percent"] = greedy_gap * 100

    # 2. Max-Cut on 10-node random graph
    n_nodes = 10
    adj, edges = generate_random_graph(n_nodes=n_nodes, edge_prob=0.45, seed=SEED)
    print(f"Max-Cut Graph: {n_nodes} nodes, {len(edges)} edges")

    cut_bf, sol_cut_bf = maxcut_brute_force(adj)
    print(f"Brute Force Optimal Cut: {cut_bf} edges cut. Partition: {sol_cut_bf}")

    # QUBO & Simulated Annealing
    Q = maxcut_to_qubo(adj)
    J, h, offset = qubo_to_ising(Q)

    # Run Simulated Annealing 5 times
    sa_cuts = []
    for r in range(5):
        _, sol_sa = maxcut_simulated_annealing(Q, iters=2000)
        c_sa = maxcut_objective(np.array(sol_sa), adj)
        sa_cuts.append(c_sa)
    
    print(f"Simulated Annealing Cuts (5 runs): {sa_cuts}, Mean={np.mean(sa_cuts):.2f}")
    results["maxcut_optimal_edges"] = cut_bf
    results["maxcut_optimal_partition"] = sol_cut_bf
    results["maxcut_sa_mean"] = float(np.mean(sa_cuts))
    results["maxcut_graph_edges"] = len(edges)

    # Save Max-Cut data for Track C to import
    trackB_export = {
        "n_nodes": n_nodes,
        "adj": adj.tolist(),
        "edges": edges,
        "optimal_cut": cut_bf,
        "optimal_partition": sol_cut_bf,
        "Q": Q.tolist()
    }
    with open(os.path.join(RESULTS_DIR, "trackB_maxcut_baseline.json"), "w") as f:
        json.dump(trackB_export, f, indent=2)

    # Plot Knapsack runtimes & Max-Cut graph
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = [row["Method"] for row in knapsack_comparison]
    runtimes = [row["Time_ms"] for row in knapsack_comparison]
    bars = axes[0].bar(methods, runtimes, color=['gray', 'orange', 'blue', 'green', 'purple'])
    axes[0].set_ylabel("Execution Time (ms, log scale)")
    axes[0].set_yscale('log')
    axes[0].set_title("Knapsack Solvers: Runtime Scaling")
    axes[0].tick_params(axis='x', rotation=30)

    # Max-Cut partition visual
    pos = {i: (np.cos(2 * np.pi * i / n_nodes), np.sin(2 * np.pi * i / n_nodes)) for i in range(n_nodes)}
    for (u, v, w) in edges:
        is_cut = (sol_cut_bf[u] != sol_cut_bf[v])
        color = 'red' if is_cut else 'gray'
        lw = 2.0 if is_cut else 0.8
        ls = '-' if is_cut else ':'
        axes[1].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color, lw=lw, ls=ls)
    
    for i in range(n_nodes):
        color = 'lightblue' if sol_cut_bf[i] == 0 else 'salmon'
        axes[1].scatter(pos[i][0], pos[i][1], s=350, color=color, edgecolors='black', zorder=5)
        axes[1].text(pos[i][0], pos[i][1], str(i), ha='center', va='center', fontweight='bold')

    axes[1].set_title(f"Max-Cut Optimal Partition (Cut={int(cut_bf)} / {len(edges)} edges)")
    axes[1].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackB_discrete.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["discrete_plot"] = plot_path

    print("=== Track B Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_discrete_experiments()
    print(f"Track B Runtime: {time.time() - t0:.2f}s")
