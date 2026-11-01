# Track I: Quantum Algorithms and Quantum Machine Learning (Simulation Only)

> [!NOTE]
> **Hardware Status:** Simulated using PennyLane's `default.qubit` simulator. No physical quantum processor was accessed.

## Objectives
- Build hand-crafted quantum circuits from fundamental quantum gates:
  1. **Deutsch-Jozsa Algorithm:** Distinguish constant vs balanced oracles in a single evaluation ($1$ quantum query vs $2^{n-1}+1$ classical queries).
  2. **Bernstein-Vazirani Algorithm:** Determine an $n$-bit secret string $s$ via phase kickback in $1$ query (vs $n$ classical queries).
  3. **Grover's Search Algorithm:** Build custom phase oracle and diffusion operator ($2|s\rangle\langle s| - I$) verifying quadratic speedup $\mathcal{O}(\sqrt{N})$.
  4. **Quantum Fourier Transform (QFT):** Controlled phase kickback cascade creating uniform frequency interference.
  5. **Shor's Algorithm Order Finding:** Factoring composite integer $N = 15$ through order finding $a^r \equiv 1 \pmod{15}$.
- Benchmark Variational Quantum Classifier (PQC) against classical Logistic Regression and RBF Support Vector Machines (SVM) on 2D Moons.
- Contrast theoretical asymptotic algorithmic speedups ($O(\sqrt{N})$) with practical simulation wall-clock overhead.

## How to Run
```bash
python trackI_quantum_algos_qml/run.py
pytest trackI_quantum_algos_qml/tests/
```
