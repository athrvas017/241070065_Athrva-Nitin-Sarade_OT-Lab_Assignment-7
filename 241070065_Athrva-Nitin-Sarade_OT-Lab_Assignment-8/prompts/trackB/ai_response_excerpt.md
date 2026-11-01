# Excerpt of Antigravity AI Response for Track B

Formulated knapsack with exact solvers (BF, DP, B&B with fractional LP relaxation bound, ILP via `scipy.optimize.milp`), proving exact consensus on value 520. Formulated Max-Cut QUBO with $Q_{ii} = -d_i, Q_{ij} = w_{ij}$ and derived exact linear transformation to Ising spins $s_i = 1 - 2x_i$. Simulated annealing found the exact 15-edge cut matching brute force across 5 independent runs.
