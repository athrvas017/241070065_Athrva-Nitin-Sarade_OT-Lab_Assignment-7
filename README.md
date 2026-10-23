# Lab Assignment 7 — Multiobjective Optimization (37 Algorithms)

**Student Name:** Athrva Nitin Sarade  
**Registration ID:** 241070065  
**Batch:** B  
**Roll Number:** 65  
**Language:** Python 3.10+  
**Libraries allowed:** `numpy`, `matplotlib`, `scipy.optimize` (only for algorithms 1–7 sub-problems), `reportlab` (for PDF generation)

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run all Task A experiments (algorithm-by-algorithm, fixed seeds)
```bash
python experiments/run_task_a.py
```

### 3. Run sensitivity studies for deep-dive algorithms (28, 3, 15)
```bash
python experiments/run_task_c_sensitivity.py
```

### 4. Run head-to-head comparisons (Task D)
```bash
python experiments/run_task_d1_continuous.py    # P3 continuous head-to-head
python experiments/run_task_d2_many_obj.py      # P10 M=5 many-objective comparison
python experiments/run_task_d3_discrete.py      # P8 (TSP 15-city bi-objective)
python experiments/run_task_d_classical.py      # sweeps for algos 1-7, TOPSIS/VIKOR, MGDA/GradNorm
```

### 5. Generate the complete PDF report
```bash
python pdf_gen.py
```
*Outputs `report/Lab7_Report.pdf` (and `Lab7_Report.pdf` in root) automatically.*

### 6. Interactive Jupyter notebook
```bash
jupyter notebook report/Lab7_Report.ipynb
```

---


## Seeds

Global seed list: `SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

P8 city coordinates use: `np.random.seed(7)`

---

## Repository Layout

```
lab7/
├── README.md                      <- this file
├── requirements.txt               <- numpy, matplotlib, scipy, jupyter
├── moo_utils.py                   <- shared toolkit (dominance, HV, IGD, spacing)
├── problems.py                    <- P1-P14 problem bank
├── ga_core.py                     <- SBX crossover, polynomial mutation, tournament
├── classical.py                   <- Algorithms 1-9 (scalarization + MCDM)
├── non_elitist.py                 <- Algorithms 10-13 (VEGA, MOGA, NSGA, NPGA)
├── elitist.py                     <- Algorithms 14-21 (NSGA-II, NSGA-III, SPEA2)
├── swarm.py                       <- Algorithms 22-30 (MOPSO, MOGWO, MOABC)
├── advanced.py                    <- Algorithms 31-37 (RVEA, IBEA, SMS-EMOA, MGDA)
├── algorithms/                    <- One module per algorithm
├── tests/                         <- Unit tests (one per algorithm)
├── experiments/                   <- Run scripts for Tasks A-D
├── results/                       <- Raw CSV/NPY output
├── figures/                       <- PNGs referenced by the report
├── hand_traces/                   <- Markdown traces for algorithms 28, 3, 15
├── report/
│   ├── Lab7_Report.ipynb
│   └── Lab7_Report.pdf
└── implementation_log.csv         <- Task A3 table (auto-generated)
```

---

## Task C Algorithms (Roll 65)

| Slot | Formula | Value | Algorithm |
|---|---|---|---|
| 1 | (64 mod 37)+1 | **28** | MO-TLBO |
| 2 | (76 mod 37)+1 | **3** | Benson's Method |
| 3 | (88 mod 37)+1 | **15** | NSGA-III |

---

## Python and Library Versions

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
jupyter>=1.0
```
