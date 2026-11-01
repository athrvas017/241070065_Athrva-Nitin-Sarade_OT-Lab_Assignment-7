# Track B Prompt

Implement discrete optimization algorithms:
1. 0/1 Knapsack selection: Brute Force, Greedy (value/weight density), Dynamic Programming O(nW), LP relaxation Branch-and-Bound, and ILP via scipy.optimize.milp.
2. Max-Cut on seeded random graph (10 nodes, seed 241070065).
3. Formulate Max-Cut as QUBO min x^T Q x and convert to Ising spin Hamiltonian s in {-1, +1}^n via s_i = 1 - 2 x_i.
4. Solve QUBO using classical Simulated Annealing and compare with exact Brute Force.
5. Export baseline graph, QUBO matrix, and optimal cut to results/trackB_maxcut_baseline.json for Track C QAOA.
