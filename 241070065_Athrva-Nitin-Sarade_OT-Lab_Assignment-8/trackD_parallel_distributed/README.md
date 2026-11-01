# Track D: Submodular, Parallel, and Distributed Optimization

## Objectives
- Formulate a facility-location coverage objective $F(S) = \sum_{i \in V} \max_{j \in S} \text{sim}(i, j)$ and verify diminishing returns $F(S \cup \{u\}) - F(S) \ge F(T \cup \{u\}) - F(T)$ for $S \subseteq T$.
- Empirically validate the Nemhauser et al. $(1 - 1/e) \approx 0.632$ greedy approximation guarantee against exact brute force.
- Benchmark multi-core CPU parallel execution using `multiprocessing.Pool` across varying core counts, comparing static chunking vs dynamic task assignment, and relating results to Amdahl's and Gustafson's Laws.
- Implement Federated Averaging (FedAvg) on a non-IID logistic regression task, examining local drift across local step parameters $E \in \{1, 5, 20\}$.
- Compare multi-task optimization strategies: Equal Weighting vs Multiple Gradient Descent Algorithm (MGDA) vs GradNorm on conflicting loss surfaces.
- Simulate 1F1B pipeline parallelism and calculate pipeline bubble fraction $\frac{p - 1}{m + p - 1}$.

## How to Run
```bash
python trackD_parallel_distributed/run.py
pytest trackD_parallel_distributed/tests/
```
