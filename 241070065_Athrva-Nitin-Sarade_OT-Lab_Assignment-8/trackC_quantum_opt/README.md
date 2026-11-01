# Track C: Quantum Optimization (Simulation Only)

## Objectives
- Implement the Quantum Approximate Optimization Algorithm (QAOA) with depth $p=1, 2$ on the Max-Cut problem derived from Track B.
- Implement the Variational Quantum Eigensolver (VQE) to compute the ground-state energy of a 2-qubit Ising Hamiltonian $H = Z_0 Z_1 + 0.5 X_0 + 0.5 X_1$.
- Evaluate quantum gradient optimization schemes: Parameter-Shift Rule, Quantum Natural Gradient (QNG), and Simultaneous Perturbation Stochastic Approximation (SPSA).
- Investigate barren plateaus by measuring the exponential decay of gradient variance across 2 to 8 qubits.
- Analyze the impact of finite shot noise ($10^2, 10^3, 10^4$ shots) and depolarizing noise channels.

## Key Theoretical Formulations
1. **QAOA State Evolution:**
   $$|\gamma, \beta\rangle = \prod_{l=1}^p e^{-i \beta_l H_M} e^{-i \gamma_l H_C} |+\rangle^{\otimes n}$$
2. **Parameter-Shift Rule:**
   $$\frac{\partial \langle H \rangle}{\partial \theta_i} = \frac{\langle H \rangle_{\theta_i + \pi/2} - \langle H \rangle_{\theta_i - \pi/2}}{2}$$
3. **Barren Plateau Scaling:**
   $$\text{Var}_{\theta}\left[\frac{\partial \langle H \rangle}{\partial \theta}\right] \in \mathcal{O}(2^{-n})$$

## Physical Embedding Limits on Quantum Annealers
Real quantum annealers (e.g., D-Wave Pegasus / Advantage architecture) have restricted sparse qubit connectivity graphs. Mapping an arbitrary dense QUBO instance onto the hardware requires **minor embedding**, where a single logical variable is chained across multiple physical qubits via strong ferromagnetic couplings ($J_{\text{chain}}$). This inflates the required qubit count quadratically, introduces broken chains during readout, and limits precision due to analog control errors.

## How to Run
```bash
python trackC_quantum_opt/run.py
pytest trackC_quantum_opt/tests/
```
