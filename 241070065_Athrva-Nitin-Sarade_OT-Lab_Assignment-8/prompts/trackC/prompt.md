# Track C Prompt

Implement quantum optimization simulation in PennyLane (default.qubit):
1. QAOA on the Max-Cut problem from Track B for depths p=1 and p=2. Compute expected cut weight and approximation ratio against classical brute force optimum.
2. VQE on 2-qubit Ising Hamiltonian H = Z0 Z1 + 0.5 X0 + 0.5 X1 comparing Adam, Parameter-Shift, Quantum Natural Gradient (QNG), and SPSA.
3. Barren plateau variance scaling across 2, 4, 6, 8 qubits showing exponential decay.
4. Finite shot noise standard deviation scaling (100, 1000, 10000 shots).
5. Document physical embedding limitations on quantum annealers (minor embedding overhead, sparse connectivity).
