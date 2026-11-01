# Track J: Software Engineering Optimization

## Objectives
- Formulate **Test Suite Minimization** as a classic Minimum Set Cover optimization problem:
  $$\min \sum_{t \in T} \text{Time}(t) \cdot x_t \quad \text{s.t.} \quad \sum_{t: l \in \text{Lines}(t)} x_t \ge 1 \quad \forall l \in L, \quad x_t \in \{0, 1\}$$
- Guarantee 100% statement coverage preservation while discarding redundant test executions.
- Compare heuristic Greedy Set Cover (maximum marginal coverage gain per millisecond) against exact Integer Linear Programming (ILP via `scipy.optimize.milp`).
- Profile performance bottlenecks using Python's `cProfile` and demonstrate measured runtime gains via algorithmic refactoring.

## How to Run
```bash
python trackJ_software_opt/run.py
pytest trackJ_software_opt/tests/
```
