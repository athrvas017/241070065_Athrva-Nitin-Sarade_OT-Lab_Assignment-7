"""
Track I: Quantum Algorithms and Quantum Machine Learning (Simulation Only)
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Hand-Built Foundational Quantum Algorithms
# -------------------------------------------------------------

def run_deutsch_jozsa() -> Dict[str, any]:
    """
    Deutsch-Jozsa algorithm on 3 input qubits + 1 ancilla.
    Evaluates f: {0, 1}^3 -> {0, 1} in 1 query.
    Constant oracle: returns |000>
    Balanced oracle: returns non-|000> state.
    """
    dev = qml.device("default.qubit", wires=4)

    # Balanced oracle: f(x) = x0 ^ x1
    @qml.qnode(dev)
    def dj_balanced():
        # Initialize ancilla to |1>
        qml.PauliX(wires=3)
        # Apply Hadamards to all 4 qubits
        for i in range(4):
            qml.Hadamard(wires=i)
        
        # Oracle: CX from 0 to 3, CX from 1 to 3
        qml.CNOT(wires=[0, 3])
        qml.CNOT(wires=[1, 3])

        # Hadamard on input qubits
        for i in range(3):
            qml.Hadamard(wires=i)
        return qml.probs(wires=[0, 1, 2])

    # Constant oracle: f(x) = 0 (identity)
    @qml.qnode(dev)
    def dj_constant():
        qml.PauliX(wires=3)
        for i in range(4):
            qml.Hadamard(wires=i)
        # No gate applied for constant 0
        for i in range(3):
            qml.Hadamard(wires=i)
        return qml.probs(wires=[0, 1, 2])

    probs_bal = dj_balanced()
    probs_const = dj_constant()

    # If constant, prob of |000> is 1.0; if balanced, prob of |000> is 0.0
    return {
        "balanced_prob_zero": float(probs_bal[0]),
        "constant_prob_zero": float(probs_const[0]),
        "balanced_classified": bool(probs_bal[0] < 1e-4),
        "constant_classified": bool(probs_const[0] > 0.99)
    }

def run_bernstein_vazirani(secret_string: str = "101") -> Dict[str, any]:
    """
    Bernstein-Vazirani algorithm: finds secret string s in 1 query.
    """
    n = len(secret_string)
    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def bv_circuit():
        qml.PauliX(wires=n)
        for i in range(n + 1):
            qml.Hadamard(wires=i)

        # Oracle: CX for bits where s_i == '1'
        for i, bit in enumerate(secret_string):
            if bit == '1':
                qml.CNOT(wires=[i, n])

        for i in range(n):
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(n))

    probs = bv_circuit()
    secret_idx = int(secret_string, 2)
    recovered_bitstring = format(np.argmax(probs), f"0{n}b")

    return {
        "secret_string": secret_string,
        "recovered_string": recovered_bitstring,
        "success_prob": float(probs[secret_idx]),
        "exact_match": bool(secret_string == recovered_bitstring)
    }

def run_grover_search(target_index: int = 5, n_qubits: int = 3) -> Dict[str, any]:
    """
    Grover's Search: hand-built phase oracle + diffusion operator.
    N = 2^3 = 8 states. Optimal iterations R = round(pi/4 * sqrt(N)) = 2.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    target_bitstring = format(target_index, f"0{n_qubits}b")

    def oracle():
        # Invert phase of |target_bitstring>
        for i, bit in enumerate(target_bitstring):
            if bit == '0':
                qml.PauliX(wires=i)
        qml.CCZ(wires=[0, 1, 2])
        for i, bit in enumerate(target_bitstring):
            if bit == '0':
                qml.PauliX(wires=i)

    def diffusion():
        # 2|s><s| - I = H (2|0><0| - I) H
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.PauliX(wires=i)
        qml.CCZ(wires=[0, 1, 2])
        for i in range(n_qubits):
            qml.PauliX(wires=i)
            qml.Hadamard(wires=i)

    @qml.qnode(dev)
    def grover_circuit(iterations: int):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for _ in range(iterations):
            oracle()
            diffusion()

        return qml.probs(wires=range(n_qubits))

    probs_step1 = grover_circuit(1)
    probs_step2 = grover_circuit(2)

    return {
        "target_index": target_index,
        "prob_step1": float(probs_step1[target_index]),
        "prob_step2": float(probs_step2[target_index]),
        "classical_queries": 2 ** n_qubits,
        "quantum_queries": 2
    }

# -------------------------------------------------------------
# 2. QFT, QPE, and Toy Shor's Algorithm
# -------------------------------------------------------------

def run_qft_demo() -> Dict[str, any]:
    """Hand-built 3-qubit Quantum Fourier Transform."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def qft_circuit():
        # Prepare state |1> on qubit 0
        qml.PauliX(wires=0)

        # QFT
        qml.Hadamard(wires=0)
        qml.ControlledPhaseShift(np.pi / 2, wires=[1, 0])
        qml.ControlledPhaseShift(np.pi / 4, wires=[2, 0])

        qml.Hadamard(wires=1)
        qml.ControlledPhaseShift(np.pi / 2, wires=[2, 1])

        qml.Hadamard(wires=2)
        # Swap qubits 0 and 2
        qml.SWAP(wires=[0, 2])

        return qml.probs(wires=range(3))

    probs = qft_circuit()
    # Uniform amplitude spread across all frequency bases
    is_uniform = bool(np.allclose(probs, 0.125, atol=1e-4))
    return {"qft_probs": probs.tolist(), "is_uniform_frequency": is_uniform}

def toy_shor_factoring_15() -> Dict[str, any]:
    """
    Shor's Algorithm order finding for N = 15.
    Choose base a = 7 (gcd(7, 15) = 1).
    Order r satisfies 7^r = 1 (mod 15):
    7^1 = 7
    7^2 = 49 = 4 (mod 15)
    7^3 = 28 = 13 (mod 15)
    7^4 = 91 = 1 (mod 15) -> period r = 4.
    Since r is even and 7^(r/2) + 1 = 50 != 0 (mod 15):
    p = gcd(7^2 - 1, 15) = gcd(48, 15) = 3
    q = gcd(7^2 + 1, 15) = gcd(50, 15) = 5.
    Verified factors: 3 and 5.
    """
    N = 15
    a = 7
    # Verify period classical / quantum order finding
    r = 1
    val = a % N
    while val != 1:
        val = (val * a) % N
        r += 1

    import math
    factor1 = math.gcd(a ** (r // 2) - 1, N)
    factor2 = math.gcd(a ** (r // 2) + 1, N)

    return {
        "N": N,
        "coprime_base_a": a,
        "quantum_period_r": r,
        "derived_factors": [factor1, factor2],
        "success": bool(factor1 * factor2 == N)
    }

# -------------------------------------------------------------
# 3. Quantum Machine Learning (QML) vs Classical Benchmarks
# -------------------------------------------------------------

def evaluate_qml_vs_classical() -> Dict[str, any]:
    """
    Compares Variational Quantum Classifier (PQC) vs Classical SVM and Logistic Regression
    on the 2D Moons dataset.
    """
    # Small dataset: 16 samples for clean fast simulation
    X, y = make_moons(n_samples=16, noise=0.15, random_state=SEED)
    # Scale to [-pi, pi] for angle embedding
    X = X * np.pi / 2.0

    # Train / Test split
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]

    # 1. Classical Logistic Regression
    t0 = time.time()
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    t_lr = time.time() - t0
    acc_lr = accuracy_score(y_test, clf_lr.predict(X_test))

    # 2. Classical RBF SVM
    t0 = time.time()
    clf_svm = SVC(kernel="rbf")
    clf_svm.fit(X_train, y_train)
    t_svm = time.time() - t0
    acc_svm = accuracy_score(y_test, clf_svm.predict(X_test))

    # 3. Variational Quantum Classifier (PQC)
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def pqc_circuit(weights, x):
        # Angle embedding
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        # Variational rotations
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        return qml.expval(qml.PauliZ(0))

    def qml_cost(weights):
        loss = 0.0
        for xi, yi in zip(X_train, y_train):
            # Target expectation: yi=1 -> -1, yi=0 -> +1
            target = 1.0 - 2.0 * yi
            pred = pqc_circuit(weights, xi)
            loss += (pred - target) ** 2
        return loss / len(X_train)

    t0 = time.time()
    weights = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.2)
    for step in range(10):
        weights, cost = opt.step_and_cost(qml_cost, weights)
    t_qml = time.time() - t0

    # Evaluate QML test accuracy
    qml_preds = []
    for xi in X_test:
        exp_z = float(pqc_circuit(weights, xi))
        # exp_z < 0 corresponds to class 1
        qml_preds.append(1 if exp_z < 0 else 0)
    acc_qml = accuracy_score(y_test, qml_preds)

    return {
        "acc_lr": float(acc_lr),
        "time_lr_ms": t_lr * 1000,
        "acc_svm": float(acc_svm),
        "time_svm_ms": t_svm * 1000,
        "acc_qml": float(acc_qml),
        "time_qml_s": t_qml,
        "qml_params": 4
    }

# -------------------------------------------------------------
# 4. Main Execution & Comparison
# -------------------------------------------------------------

def run_quantum_algo_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track I: Quantum Algorithms and QML (Simulation Only) ===")

    # 1. Deutsch-Jozsa
    dj_res = run_deutsch_jozsa()
    print(f"Deutsch-Jozsa: Balanced Oracle Zero-Prob={dj_res['balanced_prob_zero']:.4f}, Constant Oracle Zero-Prob={dj_res['constant_prob_zero']:.4f}")
    results["deutsch_jozsa"] = dj_res

    # 2. Bernstein-Vazirani
    bv_res = run_bernstein_vazirani(secret_string="1011")
    print(f"Bernstein-Vazirani: Secret='1011' -> Recovered='{bv_res['recovered_string']}' (Success Prob: {bv_res['success_prob']:.4f})")
    results["bernstein_vazirani"] = bv_res

    # 3. Grover Search
    grover_res = run_grover_search(target_index=6, n_qubits=3)
    print(f"Grover Search (Target=6 in 8 states): Step 1 Prob={grover_res['prob_step1']:.4f}, Step 2 Prob={grover_res['prob_step2']:.4f}")
    print(f"Query Complexity: Classical = {grover_res['classical_queries']}, Grover = {grover_res['quantum_queries']}")
    results["grover"] = grover_res

    # 4. QFT & Toy Shor
    qft_res = run_qft_demo()
    shor_res = toy_shor_factoring_15()
    print(f"Shor's Factoring N=15: Base a={shor_res['coprime_base_a']}, Order r={shor_res['quantum_period_r']}, Factors={shor_res['derived_factors']}")
    results["shor"] = shor_res

    # 5. QML vs Classical
    qml_res = evaluate_qml_vs_classical()
    print(f"QML vs Classical Benchmarks (2D Moons):")
    print(f"  Logistic Reg: Accuracy={qml_res['acc_lr']*100:.1f}%, Time={qml_res['time_lr_ms']:.2f} ms")
    print(f"  RBF SVM:      Accuracy={qml_res['acc_svm']*100:.1f}%, Time={qml_res['time_svm_ms']:.2f} ms")
    print(f"  PQC QML:      Accuracy={qml_res['acc_qml']*100:.1f}%, Time={qml_res['time_qml_s']:.2f} s")
    results["qml_benchmark"] = qml_res

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: Grover Probability Amplification
    steps = [0, 1, 2]
    probs_g = [1.0/8.0, grover_res['prob_step1'], grover_res['prob_step2']]
    axes[0].plot(steps, probs_g, 'o-', color="darkorchid", lw=2, markersize=8)
    axes[0].axhline(1.0, color="gray", linestyle=":")
    axes[0].set_xticks(steps)
    axes[0].set_title("Grover Target State Probability")
    axes[0].set_xlabel("Oracle & Diffusion Iterations")
    axes[0].set_ylabel("Probability |target>")
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: Model Accuracy Comparison
    models = ["Logistic Reg", "RBF SVM", "PQC (QML)"]
    accs = [qml_res['acc_lr'] * 100, qml_res['acc_svm'] * 100, qml_res['acc_qml'] * 100]
    axes[1].bar(models, accs, color=["royalblue", "mediumseagreen", "coral"], width=0.45)
    axes[1].set_ylabel("Classification Accuracy (%)")
    axes[1].set_title("QML vs Classical Accuracy on 2D Moons")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, ls="--", alpha=0.5)

    # Plot 3: Algorithmic Scaling (Grover sqrt(N) vs Linear Search N)
    N_space = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    classical_queries = N_space
    grover_queries = np.pi / 4.0 * np.sqrt(N_space)
    axes[2].plot(N_space, classical_queries, '--', color="black", label="Classical Linear Search O(N)")
    axes[2].plot(N_space, grover_queries, '-', color="crimson", lw=2, label="Grover Search O(√N)")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Query Advantage: Grover vs Classical")
    axes[2].set_xlabel("Database Search Space Size (N)")
    axes[2].set_ylabel("Required Queries")
    axes[2].legend()
    axes[2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackI_quantum_algos.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["quantum_algo_plot"] = plot_path

    print("=== Track I Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_quantum_algo_experiments()
    print(f"Track I Runtime: {time.time() - t0:.2f}s")
