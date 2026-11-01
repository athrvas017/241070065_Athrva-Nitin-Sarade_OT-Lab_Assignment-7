# Track H: Database Management Systems (DBMS) Optimization

## Objectives
- Generate a synthetic TPC-H relational database (Customer, Orders, Lineitem tables with 50,000+ total rows, seed `241070065`) in SQLite.
- Formulate automated physical index design as a 0/1 Knapsack optimization problem under a disk storage budget:
  $$\max \sum_{i=1}^m \Delta \text{Latency}_i \cdot x_i \quad \text{s.t.} \quad \sum_{i=1}^m \text{Size}_i \cdot x_i \le B, \quad x_i \in \{0, 1\}$$
- Compare Greedy selection (benefit-to-cost ratio $\Delta \text{Latency} / \text{Size}$) against exact Integer Linear Programming (ILP via `scipy.optimize.milp`).
- Evaluate join ordering using Dynamic Programming and execute `EXPLAIN QUERY PLAN` verifying transition from table scans to index searches.
- Assert strict relational output equivalence before and after indexing.

## How to Run
```bash
python trackH_dbms/run.py
pytest trackH_dbms/tests/
```
