# Track E: Main Case Study — Exam Timetable Scheduling

## Problem Overview
College exam invigilation and timetable scheduling for a synthetic department instance (30 exams, 200 students, 6 rooms of varied capacity, seed `241070065`).

## Mathematical Formulation
- **Decision Variables:**
  $x_{e, s, r} \in \{0, 1\}$ indicating whether exam $e \in E$ is scheduled in slot $s \in S$ and assigned to room $r \in R$.
- **Hard Constraints:**
  1. *Single Assignment:* $\sum_{s \in S} \sum_{r \in R} x_{e, s, r} = 1 \quad \forall e \in E$.
  2. *No Student Clashes:* $x_{e_1, s, r_1} + x_{e_2, s, r_2} \le 1 \quad \forall (e_1, e_2) \in E_{\text{conflict}}, \forall s \in S$.
  3. *Room Capacity:* $\sum_{e \in E} \text{Size}(e) \cdot x_{e, s, r} \le \text{Cap}(r) \quad \forall s \in S, \forall r \in R$.
  4. *Single Exam per Room per Slot:* $\sum_{e \in E} x_{e, s, r} \le 1 \quad \forall s \in S, \forall r \in R$.
- **Objective:**
  Minimize total time slots used plus soft consecutive exam spread penalties.

## Algorithms Evaluated
1. **Naive Sequential First-Fit:** Greedy sequential room and slot selection.
2. **DSATUR Graph Coloring:** Dynamic saturation degree prioritization.
3. **Simulated Annealing:** Local search optimizing soft student consecutive exam penalties (evaluated across 5 seeded runs with mean $\pm$ std).

## How to Run
```bash
python trackE_case_study/run.py
pytest trackE_case_study/tests/
```
