"""
Track D: Submodular, Parallel, and Distributed Optimization
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Submodular Facility Location / Coverage Function
# -------------------------------------------------------------

class FacilityLocationCoverage:
    """
    Submodular function: F(S) = sum_{i in V} max_{j in S} sim(i, j).
    Monotone and submodular.
    """
    def __init__(self, n_items: int = 15, seed: int = SEED):
        rng = np.random.RandomState(seed)
        # 2D coordinates for synthetic users and facilities
        self.coords = rng.rand(n_items, 2)
        # Pairwise Gaussian similarity matrix
        dists = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        self.sim_matrix = np.exp(-dists / 0.5)
        self.n = n_items

    def evaluate(self, S: Set[int]) -> float:
        if not S:
            return 0.0
        idx = list(S)
        # Max similarity from each point to any point in S
        max_sims = np.max(self.sim_matrix[:, idx], axis=1)
        return float(np.sum(max_sims))

    def verify_diminishing_returns(self, n_trials: int = 20) -> bool:
        """Verifies F(S U {u}) - F(S) >= F(T U {u}) - F(T) for S subseteq T."""
        rng = np.random.RandomState(SEED)
        all_elements = set(range(self.n))

        for _ in range(n_trials):
            # Sample random subset T of size 4 to 8
            k_t = rng.randint(4, 9)
            T = set(rng.choice(self.n, size=k_t, replace=False))
            # Sample subset S subseteq T
            k_s = rng.randint(1, k_t)
            S = set(rng.choice(list(T), size=k_s, replace=False))
            # Element u not in T
            candidates = list(all_elements - T)
            if not candidates:
                continue
            u = rng.choice(candidates)

            gain_S = self.evaluate(S | {u}) - self.evaluate(S)
            gain_T = self.evaluate(T | {u}) - self.evaluate(T)

            # Diminishing returns requires gain_S >= gain_T (within float tolerance)
            if gain_S < gain_T - 1e-10:
                return False
        return True

    def greedy_select(self, k: int) -> Tuple[float, List[int]]:
        S: Set[int] = set()
        chosen: List[int] = []
        for _ in range(k):
            best_u = -1
            best_gain = -1.0
            curr_val = self.evaluate(S)
            for u in range(self.n):
                if u not in S:
                    gain = self.evaluate(S | {u}) - curr_val
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
            if best_u != -1:
                S.add(best_u)
                chosen.append(best_u)
        return self.evaluate(S), chosen

    def brute_force_select(self, k: int) -> Tuple[float, List[int]]:
        from itertools import combinations
        best_val = -1.0
        best_set = []
        for comb in combinations(range(self.n), k):
            val = self.evaluate(set(comb))
            if val > best_val:
                best_val = val
                best_set = list(comb)
        return best_val, best_set

# -------------------------------------------------------------
# 2. Parallel Processing: Multiprocessing on CPU Tasks
# -------------------------------------------------------------

def cpu_work_task(args: Tuple[int, float]) -> float:
    """Simulates uneven CPU-bound tasks."""
    task_id, workload_factor = args
    # Synthetic CPU computation
    count = int(1.5e5 * workload_factor)
    acc = 0.0
    for i in range(count):
        acc += np.sin(i * 0.01) ** 2
    return acc

def benchmark_parallel_workers(workloads: List[float], n_cores_list: List[int]) -> Dict[str, any]:
    tasks = [(i, w) for i, w in enumerate(workloads)]
    results = {}

    for cores in n_cores_list:
        # Static vs Dynamic chunking
        t0 = time.time()
        with mp.Pool(processes=cores) as pool:
            # Dynamic work assignment: chunksize=1
            _ = pool.map(cpu_work_task, tasks, chunksize=1)
        t_dynamic = time.time() - t0

        t0 = time.time()
        with mp.Pool(processes=cores) as pool:
            # Static partition: large chunksize
            static_chunk = max(1, len(tasks) // cores)
            _ = pool.map(cpu_work_task, tasks, chunksize=static_chunk)
        t_static = time.time() - t0

        results[cores] = {"dynamic_time": t_dynamic, "static_time": t_static}

    return results

# -------------------------------------------------------------
# 3. Distributed Optimization: Federated Averaging (FedAvg)
# -------------------------------------------------------------

class FedAvgSimulation:
    def __init__(self, n_clients: int = 5, n_features: int = 4, seed: int = SEED):
        rng = np.random.RandomState(seed)
        self.n_clients = n_clients
        self.n_features = n_features
        # Create non-IID datasets for each client
        self.client_data = []
        for i in range(n_clients):
            # Client-specific bias/skew
            mean_shift = rng.uniform(-1.0, 1.0, size=n_features)
            X = rng.randn(100, n_features) + mean_shift
            w_true = rng.randn(n_features)
            # Binary labels
            logits = X @ w_true
            y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)
            self.client_data.append((X, y))

    def _client_update(self, w: np.ndarray, client_idx: int, local_steps: int, lr: float = 0.05) -> np.ndarray:
        X, y = self.client_data[client_idx]
        w_local = w.copy()
        N = len(y)
        for _ in range(local_steps):
            # Logistic regression gradient
            p = 1.0 / (1.0 + np.exp(-X @ w_local))
            grad = (X.T @ (p - y)) / N + 1e-4 * w_local
            w_local -= lr * grad
        return w_local

    def evaluate_global_loss(self, w: np.ndarray) -> float:
        total_loss = 0.0
        total_samples = 0
        for X, y in self.client_data:
            N = len(y)
            p = 1.0 / (1.0 + np.exp(-X @ w))
            loss = -np.mean(y * np.log(p + 1e-10) + (1.0 - y) * np.log(1.0 - p + 1e-10))
            total_loss += loss * N
            total_samples += N
        return float(total_loss / total_samples)

    def run_fedavg(self, local_steps: int, rounds: int = 20) -> List[float]:
        w_global = np.zeros(self.n_features)
        loss_hist = [self.evaluate_global_loss(w_global)]

        for r in range(rounds):
            local_weights = []
            for c in range(self.n_clients):
                w_c = self._client_update(w_global, c, local_steps=local_steps)
                local_weights.append(w_c)
            # Central Aggregation (Equal weighting since clients have equal sample sizes)
            w_global = np.mean(local_weights, axis=0)
            loss_hist.append(self.evaluate_global_loss(w_global))

        return loss_hist

# -------------------------------------------------------------
# 4. Multi-Task Optimization: MGDA vs GradNorm vs Equal Weighting
# -------------------------------------------------------------

def run_multitask_optimization(iters: int = 60) -> Dict[str, List[float]]:
    """
    Toy 2-task optimization:
    Task 1: f1(theta) = 0.5 * (theta_0 - 2)^2 + (theta_1 - 2)^2
    Task 2: f2(theta) = 2.0 * (theta_0 + 2)^2 + 0.5 * (theta_1 + 2)^2
    Conflicting objectives with Pareto front.
    """
    def grads(theta):
        g1 = np.array([theta[0] - 2.0, 2.0 * (theta[1] - 2.0)])
        g2 = np.array([4.0 * (theta[0] + 2.0), theta[1] + 2.0])
        return g1, g2

    def losses(theta):
        l1 = 0.5 * (theta[0] - 2.0)**2 + (theta[1] - 2.0)**2
        l2 = 2.0 * (theta[0] + 2.0)**2 + 0.5 * (theta[1] + 2.0)**2
        return l1, l2

    theta0 = np.array([0.0, 0.0])

    # 1. Equal Weighting: w1 = 0.5, w2 = 0.5
    theta_eq = theta0.copy()
    hist_eq = []
    for _ in range(iters):
        g1, g2 = grads(theta_eq)
        g_comb = 0.5 * g1 + 0.5 * g2
        theta_eq -= 0.05 * g_comb
        l1, l2 = losses(theta_eq)
        hist_eq.append(l1 + l2)

    # 2. MGDA (Multiple Gradient Descent Algorithm): Frank-Wolfe min over convex hull
    theta_mgda = theta0.copy()
    hist_mgda = []
    for _ in range(iters):
        g1, g2 = grads(theta_mgda)
        # Min norm point in convex hull of g1 and g2: min_{alpha in [0, 1]} ||alpha g1 + (1-alpha) g2||^2
        diff = g1 - g2
        denom = float(diff @ diff)
        if denom < 1e-12:
            alpha = 0.5
        else:
            alpha = np.clip(float(g2 @ (g2 - g1)) / denom, 0.0, 1.0)
        g_mgda = alpha * g1 + (1.0 - alpha) * g2
        theta_mgda -= 0.05 * g_mgda
        l1, l2 = losses(theta_mgda)
        hist_mgda.append(l1 + l2)

    # 3. GradNorm heuristic (balance gradient magnitudes)
    theta_gn = theta0.copy()
    hist_gn = []
    for _ in range(iters):
        g1, g2 = grads(theta_gn)
        n1 = np.linalg.norm(g1) + 1e-8
        n2 = np.linalg.norm(g2) + 1e-8
        avg_n = (n1 + n2) / 2.0
        # Weight inversely to gradient magnitude relative to average
        w1 = avg_n / n1
        w2 = avg_n / n2
        norm_factor = w1 + w2
        w1 /= norm_factor
        w2 /= norm_factor
        g_gn = w1 * g1 + w2 * g2
        theta_gn -= 0.05 * g_gn
        l1, l2 = losses(theta_gn)
        hist_gn.append(l1 + l2)

    return {"Equal Weighting": hist_eq, "MGDA": hist_mgda, "GradNorm": hist_gn}

# -------------------------------------------------------------
# 5. Pipeline Parallelism Timing Model (Simulated)
# -------------------------------------------------------------

def simulate_pipeline_parallelism(n_stages: int = 4, n_microbatches: int = 8) -> Dict[str, float]:
    """
    Simulated 1F1B (One-Forward-One-Backward) pipeline parallelism timing model.
    Bubble ratio = (n_stages - 1) / (n_microbatches + n_stages - 1).
    """
    bubble_fraction = (n_stages - 1) / (n_microbatches + n_stages - 1)
    ideal_speedup = n_stages * (1.0 - bubble_fraction)
    return {
        "n_stages": n_stages,
        "n_microbatches": n_microbatches,
        "bubble_fraction": bubble_fraction,
        "ideal_speedup": ideal_speedup
    }

# -------------------------------------------------------------
# 6. Main Execution & Plots
# -------------------------------------------------------------

def run_trackD_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track D: Submodular, Parallel, and Distributed ===")

    # 1. Submodular Coverage
    fl = FacilityLocationCoverage(n_items=12, seed=SEED)
    is_submod = fl.verify_diminishing_returns(n_trials=30)
    print(f"Submodularity (Diminishing Returns) Numerically Verified: {is_submod}")
    results["submodularity_verified"] = is_submod

    k_budget = 4
    val_bf, set_bf = fl.brute_force_select(k=k_budget)
    val_greedy, set_greedy = fl.greedy_select(k=k_budget)
    ratio = val_greedy / val_bf
    theoretical_guarantee = 1.0 - 1.0 / np.e

    print(f"Facility Location (k={k_budget}): BruteForce={val_bf:.4f}, Greedy={val_greedy:.4f}")
    print(f"Empirical Ratio: {ratio:.4f} (Theoretical Bound >= {theoretical_guarantee:.4f})")
    assert ratio >= theoretical_guarantee, "Greedy submodular guarantee violated!"
    results["greedy_ratio"] = ratio
    results["theoretical_bound"] = theoretical_guarantee

    # 2. Parallel Processing
    # Synthetic uneven workload distribution
    rng = np.random.RandomState(SEED)
    uneven_workloads = rng.exponential(scale=1.0, size=24) + 0.2
    available_cores = min(4, mp.cpu_count())
    cores_to_test = [1, 2] if available_cores < 4 else [1, 2, 4]
    
    print(f"Benchmarking CPU Multiprocessing across cores: {cores_to_test}")
    parallel_stats = benchmark_parallel_workers(uneven_workloads.tolist(), cores_to_test)
    results["parallel_stats"] = parallel_stats

    t1 = parallel_stats[1]["dynamic_time"]
    for c in cores_to_test:
        t_c = parallel_stats[c]["dynamic_time"]
        speedup = t1 / t_c
        efficiency = speedup / c
        print(f"Cores: {c} | Time: {t_c:.3f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency*100:.1f}%")

    # 3. FedAvg non-IID
    fed_sim = FedAvgSimulation(n_clients=5, n_features=4, seed=SEED)
    fed_hist_1 = fed_sim.run_fedavg(local_steps=1, rounds=20)
    fed_hist_5 = fed_sim.run_fedavg(local_steps=5, rounds=20)
    fed_hist_20 = fed_sim.run_fedavg(local_steps=20, rounds=20)
    results["fedavg_loss_1"] = fed_hist_1[-1]
    results["fedavg_loss_5"] = fed_hist_5[-1]
    results["fedavg_loss_20"] = fed_hist_20[-1]
    print(f"FedAvg Final Loss: E=1: {fed_hist_1[-1]:.4f}, E=5: {fed_hist_5[-1]:.4f}, E=20: {fed_hist_20[-1]:.4f}")

    # 4. Multi-Task Optimization
    mt_res = run_multitask_optimization(iters=50)
    results["multitask_final_loss"] = {k: v[-1] for k, v in mt_res.items()}
    print(f"Multi-Task Final Combined Loss: {results['multitask_final_loss']}")

    # 5. Pipeline Parallelism Simulation
    pipe_model = simulate_pipeline_parallelism(n_stages=4, n_microbatches=8)
    results["pipeline_simulation"] = pipe_model
    print(f"Pipeline Parallelism Bubble Fraction: {pipe_model['bubble_fraction']*100:.1f}%")

    # 6. Combined Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: Parallel Speedup & Amdahl's Law
    c_arr = np.array(cores_to_test)
    measured_speedups = [t1 / parallel_stats[c]["dynamic_time"] for c in c_arr]
    # Amdahl curve assuming parallel fraction p = 0.85
    p_frac = 0.85
    amdahl_curve = 1.0 / ((1.0 - p_frac) + p_frac / c_arr)
    axes[0].plot(c_arr, measured_speedups, 'o-', color='navy', label="Measured (Dynamic)", lw=2)
    axes[0].plot(c_arr, amdahl_curve, '--', color='crimson', label=f"Amdahl's Law (p={p_frac})")
    axes[0].plot(c_arr, c_arr, ':', color='gray', label="Ideal Linear")
    axes[0].set_title("CPU Multiprocessing Speedup")
    axes[0].set_xlabel("Number of Cores")
    axes[0].set_ylabel("Speedup")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: FedAvg Local Steps Convergence
    axes[1].plot(fed_hist_1, label="FedAvg (Local Steps E=1)", lw=2)
    axes[1].plot(fed_hist_5, label="FedAvg (Local Steps E=5)", lw=2)
    axes[1].plot(fed_hist_20, label="FedAvg (Local Steps E=20)", lw=2)
    axes[1].set_title("Federated Averaging (non-IID Drift)")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Global Objective Loss")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    # Plot 3: Multi-Task Optimization
    for name, loss_vals in mt_res.items():
        axes[2].plot(loss_vals, label=name, lw=2)
    axes[2].set_title("Multi-Task Optimization: MGDA vs GradNorm")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Sum of Task Losses")
    axes[2].legend()
    axes[2].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackD_submodular_parallel.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["trackD_plot"] = plot_path

    print("=== Track D Complete! ===")
    return results

if __name__ == "__main__":
    mp.freeze_support()
    t0 = time.time()
    res = run_trackD_experiments()
    print(f"Track D Runtime: {time.time() - t0:.2f}s")
