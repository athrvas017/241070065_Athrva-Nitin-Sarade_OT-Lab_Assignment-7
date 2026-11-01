"""
Track C: Quantum Optimization (Simulation Only)
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from typing import Dict, List, Tuple

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. QAOA on Max-Cut Graph
# -------------------------------------------------------------

def load_trackB_baseline() -> Tuple[int, List[Tuple[int, int, float]], float]:
    baseline_path = os.path.join(RESULTS_DIR, "trackB_maxcut_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            data = json.load(f)
            # Use 6 nodes subset to keep QAOA fast & responsive
            n_nodes = min(6, data["n_nodes"])
            edges = [(u, v, w) for (u, v, w) in data["edges"] if u < n_nodes and v < n_nodes]
            # Classical optimum for this 6-node subgraph
            best_cut = 0
            for i in range(1 << n_nodes):
                x = [(i >> j) & 1 for j in range(n_nodes)]
                cut = sum(w * (x[u] + x[v] - 2 * x[u] * x[v]) for u, v, w in edges)
                if cut > best_cut:
                    best_cut = cut
            return n_nodes, edges, best_cut
    else:
        # Fallback 6-node graph
        n_nodes = 6
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 0, 1.0), (0, 3, 1.0)]
        return n_nodes, edges, 6.0

def build_qaoa_circuit(n_nodes: int, edges: List[Tuple[int, int, float]], p: int = 1):
    dev = qml.device("default.qubit", wires=n_nodes)

    # Cost Hamiltonian: sum_{(u,v)} 0.5 * (I - Z_u Z_v)
    # Minimizing energy expectation E = -Cut
    obs = []
    coeffs = []
    for u, v, w in edges:
        obs.append(qml.PauliZ(u) @ qml.PauliZ(v))
        coeffs.append(0.5 * w)

    cost_H = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(gamma, beta):
        # Initial uniform superposition
        for i in range(n_nodes):
            qml.Hadamard(wires=i)

        for layer in range(p):
            # Cost unitary: e^{-i gamma H_C}
            for u, v, w in edges:
                qml.CNOT(wires=[u, v])
                qml.RZ(2 * gamma[layer] * 0.5 * w, wires=v)
                qml.CNOT(wires=[u, v])

            # Mixer unitary: e^{-i beta H_M} where H_M = sum X_i
            for i in range(n_nodes):
                qml.RX(2 * beta[layer], wires=i)

        return qml.expval(cost_H)

    return circuit, cost_H

def train_qaoa(circuit, p: int = 1, steps: int = 30) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    # Initialize parameters
    gamma = pnp.array([0.5] * p, requires_grad=True)
    beta = pnp.array([0.5] * p, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=0.1)
    cost_history = []

    for step in range(steps):
        (gamma, beta), cost = opt.step_and_cost(circuit, gamma, beta)
        cost_history.append(float(cost))

    return gamma, beta, cost_history

# -------------------------------------------------------------
# 2. VQE on 2-Qubit Hamiltonian with Parameter-Shift & QNG
# -------------------------------------------------------------

def run_vqe_demo() -> Dict[str, any]:
    # 2-qubit Ising Hamiltonian: H = 1.0 * Z0 @ Z1 + 0.5 * X0 + 0.5 * X1
    H = qml.Hamiltonian([1.0, 0.5, 0.5], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0), qml.PauliX(1)])
    # Exact diagonal ground state energy
    H_mat = qml.matrix(H)
    exact_ground_energy = float(np.min(np.linalg.eigvalsh(H_mat)))

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method="parameter-shift")
    def vqe_circuit(params):
        # Hardware efficient ansatz
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        return qml.expval(H)

    # 1. Parameter-shift Adam
    params_adam = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    opt_adam = qml.AdamOptimizer(stepsize=0.1)
    hist_adam = []
    for _ in range(25):
        params_adam, cost = opt_adam.step_and_cost(vqe_circuit, params_adam)
        hist_adam.append(float(cost))

    # 2. Quantum Natural Gradient (QNG)
    params_qng = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    opt_qng = qml.QNGOptimizer(stepsize=0.1)
    hist_qng = []
    for _ in range(25):
        params_qng, cost = opt_qng.step_and_cost(vqe_circuit, params_qng)
        hist_qng.append(float(cost))

    # 3. SPSA
    params_spsa = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    opt_spsa = qml.SPSAOptimizer(maxiter=25, c=0.1, a=0.1)
    hist_spsa = []
    curr_p = params_spsa
    for _ in range(25):
        curr_p, cost = opt_spsa.step_and_cost(vqe_circuit, curr_p)
        hist_spsa.append(float(cost))

    return {
        "exact_ground_energy": exact_ground_energy,
        "adam_final": hist_adam[-1],
        "qng_final": hist_qng[-1],
        "spsa_final": hist_spsa[-1],
        "hist_adam": hist_adam,
        "hist_qng": hist_qng,
        "hist_spsa": hist_spsa
    }

# -------------------------------------------------------------
# 3. Barren Plateau Gradient Variance Scaling
# -------------------------------------------------------------

def analyze_barren_plateaus(qubit_counts: List[int] = [2, 4, 6, 8], n_samples: int = 25) -> Dict[int, float]:
    variances = {}
    for n_qubits in qubit_counts:
        dev = qml.device("default.qubit", wires=n_qubits)
        # Random hardware efficient ansatz
        # Measure Z on qubit 0
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(weights):
            for i in range(n_qubits):
                qml.RY(weights[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(weights[n_qubits + i], wires=i)
            return qml.expval(qml.PauliZ(0))

        grads = []
        for _ in range(n_samples):
            # Sample uniformly random parameters in [0, 2pi]
            w = pnp.array(np.random.uniform(0, 2 * np.pi, size=2 * n_qubits), requires_grad=True)
            g = qml.grad(circuit)(w)
            grads.append(float(g[0]))

        variances[n_qubits] = float(np.var(grads))
    return variances

# -------------------------------------------------------------
# 4. Finite Shot Noise and Depolarizing Noise Effect
# -------------------------------------------------------------

def evaluate_shot_noise(shots_list: List[int] = [100, 1000, 10000]) -> Dict[int, float]:
    # Evaluate expectation of PauliZ on state |+>
    results = {}
    for shots in shots_list:
        dev = qml.device("default.qubit", wires=1, shots=shots)
        @qml.qnode(dev)
        def circ():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        # Analytical is 0.0
        trials = [circ() for _ in range(10)]
        results[shots] = float(np.std(trials))
    return results

# -------------------------------------------------------------
# 5. Main Execution & Plot Generation
# -------------------------------------------------------------

def run_quantum_opt_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track C: Quantum Optimization (Simulation Only) ===")

    # 1. QAOA Max-Cut
    n_nodes, edges, classical_opt_cut = load_trackB_baseline()
    print(f"QAOA on {n_nodes} nodes, {len(edges)} edges. Classical Max-Cut: {classical_opt_cut}")
    
    # p=1 QAOA
    circ_p1, _ = build_qaoa_circuit(n_nodes, edges, p=1)
    gamma1, beta1, hist_p1 = train_qaoa(circ_p1, p=1, steps=25)
    
    # p=2 QAOA
    circ_p2, _ = build_qaoa_circuit(n_nodes, edges, p=2)
    gamma2, beta2, hist_p2 = train_qaoa(circ_p2, p=2, steps=25)

    # Note: Cost function returns sum 0.5 * Z_u Z_v. Cut weight = sum 0.5 * (1 - Z_u Z_v) = 0.5 * |E| - energy
    total_edges = sum(w for _, _, w in edges)
    qaoa_cut_p1 = 0.5 * total_edges - hist_p1[-1]
    qaoa_cut_p2 = 0.5 * total_edges - hist_p2[-1]
    approx_ratio_p1 = float(qaoa_cut_p1 / classical_opt_cut)
    approx_ratio_p2 = float(qaoa_cut_p2 / classical_opt_cut)

    print(f"QAOA p=1 Cut: {qaoa_cut_p1:.2f} (Approx Ratio: {approx_ratio_p1:.3f})")
    print(f"QAOA p=2 Cut: {qaoa_cut_p2:.2f} (Approx Ratio: {approx_ratio_p2:.3f})")
    results["qaoa_p1_cut"] = float(qaoa_cut_p1)
    results["qaoa_p2_cut"] = float(qaoa_cut_p2)
    results["approx_ratio_p1"] = approx_ratio_p1
    results["approx_ratio_p2"] = approx_ratio_p2

    # 2. VQE
    vqe_res = run_vqe_demo()
    print(f"VQE Ground State: Exact={vqe_res['exact_ground_energy']:.4f}, Adam={vqe_res['adam_final']:.4f}, QNG={vqe_res['qng_final']:.4f}")
    results["vqe"] = {k: v for k, v in vqe_res.items() if not k.startswith("hist_")}

    # 3. Barren Plateau Scaling
    print("Computing Barren Plateau gradient variances across 2, 4, 6, 8 qubits...")
    bp_variances = analyze_barren_plateaus([2, 4, 6, 8], n_samples=20)
    print(f"Barren Plateau Variances: {bp_variances}")
    results["barren_plateau_variances"] = bp_variances

    # 4. Shot Noise
    shot_stds = evaluate_shot_noise([100, 1000, 10000])
    print(f"Shot Noise Standard Deviations: {shot_stds}")
    results["shot_noise_stds"] = shot_stds

    # 5. Combined Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: QAOA Convergence
    axes[0].plot([0.5 * total_edges - c for c in hist_p1], label="QAOA (p=1)", color="blue", lw=2)
    axes[0].plot([0.5 * total_edges - c for c in hist_p2], label="QAOA (p=2)", color="purple", lw=2)
    axes[0].axhline(classical_opt_cut, color="red", linestyle="--", label=f"Classical Opt ({classical_opt_cut})")
    axes[0].set_title("QAOA Max-Cut Expectation Value")
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("Expected Cut Weight")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: VQE Optimizer Comparison
    axes[1].plot(vqe_res["hist_adam"], label="Adam", lw=2)
    axes[1].plot(vqe_res["hist_qng"], label="Quantum Natural Grad (QNG)", lw=2)
    axes[1].plot(vqe_res["hist_spsa"], label="SPSA", lw=1.5, alpha=0.8)
    axes[1].axhline(vqe_res["exact_ground_energy"], color="black", linestyle=":", label="Exact Minimum")
    axes[1].set_title("VQE 2-Qubit Hamiltonian Ground State")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Energy")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    # Plot 3: Barren Plateau Variance Decay (Log scale)
    q_vals = list(bp_variances.keys())
    var_vals = list(bp_variances.values())
    axes[2].semilogy(q_vals, var_vals, 'o-', color="crimson", lw=2, markersize=8)
    axes[2].set_title("Barren Plateau: Var[∂H/∂θ] vs Qubits")
    axes[2].set_xlabel("Number of Qubits")
    axes[2].set_ylabel("Gradient Variance (log scale)")
    axes[2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackC_quantum_opt.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["quantum_opt_plot"] = plot_path

    print("=== Track C Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_quantum_opt_experiments()
    print(f"Track C Runtime: {time.time() - t0:.2f}s")
