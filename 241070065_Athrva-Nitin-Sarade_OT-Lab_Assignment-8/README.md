# Optimization Techniques (Lab Assignment 8): Optimize Anything

**Student:** Athrva Nitin Sarade  
**Roll Number:** 241070065  
**Institution:** Veermata Jijabai Technological Institute (VJTI), Mumbai  
**Course:** Optimization Techniques (M.Tech Computer Engineering)  
**Global Random Seed:** `241070065`

---

## Overview

This repository implements **Assignment 8: "Optimize Anything"**, covering comprehensive computational optimization across 10 specialized tracks (Tracks A through J). All experiments are strictly grounded in empirical code executions adhering to the Gravitas engineering reliability runtime.

### Tracks Implemented
- **Track A — Continuous Optimization**: First-order (GD, SGD, Mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam) and second-order (Newton, Gauss-Newton) methods, manual MLP backpropagation, Hessian-vector products (HVP), Fisher diagonal approximations, step-size tuning, and noisy landscapes.
- **Track B — Discrete Optimization**: 0/1 Knapsack & Max-Cut graph partitioning using Greedy, Dynamic Programming, Branch-and-Bound, ILP, and QUBO / Ising spin transformations with Simulated Annealing.
- **Track C — Quantum Optimization**: QAOA on Max-Cut QUBO, VQE ground-state estimation, Parameter-Shift gradients, SPSA, Quantum Natural Gradient (QNG), depolarizing noise channels, finite shot statistics, and barren plateau variance scaling.
- **Track D — Submodular, Parallel & Distributed**: Diminishing returns coverage functions, multi-core CPU multiprocessing (Amdahl & Gustafson scaling), dynamic work scheduling, Federated Averaging (FedAvg), and multi-task optimization (MGDA, GradNorm).
- **Track E — Main Case Study (Exam Timetable Scheduling)**: Real-world college exam invigilation scheduling under student clash and room capacity constraints using DSATUR graph coloring, Simulated Annealing, and ILP.
- **Track F — Compiler Optimization**: Mini-compiler pipeline (Lexer, Parser, 3-Address IR, VM) with Constant Folding, Constant Propagation, Dead-Code Elimination, Common Sub-expression Elimination, and Chaitin-style register allocation graph coloring.
- **Track G — Operating System CPU Scheduling**: Faithful kernel CPU scheduling simulator (Round Robin, Priority, Lottery, MLFQ) across CPU-bound, I/O-bound, and mixed workloads, plus an xv6-riscv kernel patch draft.
- **Track H — DBMS Optimization**: SQLite TPC-H database workload optimization via automated 0/1 Knapsack index selection (Greedy vs ILP) and dynamic programming join ordering with `EXPLAIN QUERY PLAN` validation.
- **Track I — Quantum Algorithms & QML**: Hand-built circuits for Deutsch-Jozsa, Bernstein-Vazirani, Grover Search ($O(\sqrt{N})$ scaling), QFT, QPE, toy Shor's algorithm, and Variational Quantum Classifier (PQC) vs classical SVM/logistic regression.
- **Track J — Software Engineering Optimization**: Test-suite minimization via set cover (Greedy vs ILP) on statement coverage profiles, and `cProfile` bottleneck optimization.

---

## AI Tools Register (Mandatory Disclosure)

| Tool Name | Model Version | Tracks Involved | Dates | Nature of Assistance |
| :--- | :--- | :--- | :--- | :--- |
| **Antigravity** | Gemini 3.8 Flash | Tracks A–J, Docs | 2026-11-01 | Architecture planning, mathematical derivation, code generation, test suite writing, execution logging |
| **Claude / ChatGPT** | *(Student can append)* | Planning prompt | 2026-11-01 | Initial prompt generation & report compilation |

*(Student: Please update this table if any additional external conversational LLMs or copilot assistants were used during the assignment.)*

---

## Environment Setup & Reproduction

This project uses `uv` and Python virtual environments (`venv`).

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.ps1 | iex` on Windows or `pip install uv`)

### 2. Environment Setup
```bash
# Clone the repository and navigate to Assignment 8
cd 241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8

# Create virtual environment using uv
uv venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install all pinned dependencies
uv pip install -r requirements.txt
```

### 3. Run All Tracks
To execute all experiments, generate figures, run tests, and produce `REPORT_DATA.md`:
```bash
# Windows PowerShell
python run_all.py

# Or bash
bash run_all.sh
```

All generated plots and CSV tables are automatically placed in `results/`.

---

## Key Deliverables & Documentation Links
- [COVERAGE_CHECKLIST.md](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-7/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8/COVERAGE_CHECKLIST.md): Complete mapping of all Chapter 6 syllabus concepts to code.
- [FAILURE_LOG.md](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-7/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8/FAILURE_LOG.md): Four documented real AI edge cases, mathematical bugs, and lessons learned.
- [TOOL_COMPARISON.md](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-7/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8/TOOL_COMPARISON.md): Direct head-to-head evaluation of AI tools on QUBO formulations.
- [VIVA_NOTES.md](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-7/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8/VIVA_NOTES.md): 5-line track summaries and 2 live modification tasks per track.
- [REPORT_DATA.md](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-7/241070065_Athrva-Nitin-Sarade_OT-Lab_Assignment-8/REPORT_DATA.md): Comprehensive data export for technical report generation.
