# Track B: Discrete Optimization

## Objectives
- Compare exact and heuristic combinatorial algorithms for 0/1 Knapsack: Brute Force, Greedy (density), Dynamic Programming $O(nW)$, LP-relaxation Branch-and-Bound, and Integer Linear Programming (ILP via `scipy.optimize.milp`).
- Formulate the Maximum Cut problem on a seeded random graph ($N=10$) as Quadratic Unconstrained Binary Optimization (QUBO) $\min x^T Q x$.
- Perform binary-to-spin variable transformation $s_i = 1 - 2x_i \in \{-1, +1\}$ to establish the equivalent Ising Hamiltonian $H = \sum J_{ij} s_i s_j + \sum h_i s_i + C$.
- Solve QUBO using classical Simulated Annealing and export the optimal cut and baseline graph to `results/trackB_maxcut_baseline.json` for quantum Track C.

## Formulations
1. **Knapsack ILP:**
   $$\max \sum_{i=1}^n v_i x_i \quad \text{s.t.} \quad \sum_{i=1}^n w_i x_i \le W, \quad x_i \in \{0, 1\}$$
2. **Max-Cut QUBO Matrix:**
   $$\text{Cut}(x) = \sum_{(i,j) \in E} w_{ij}(x_i + x_j - 2x_i x_j) = x^T (D - A) x$$
   $$\min x^T Q x \quad \text{where} \quad Q_{ii} = -d_i, \quad Q_{ij} = w_{ij}$$
3. **Ising Hamiltonian:**
   $$H_C = \sum_{(i,j) \in E} \frac{w_{ij}}{2} (1 - s_i s_j)$$

## How to Run
```bash
python trackB_discrete/run.py
pytest trackB_discrete/tests/
```
