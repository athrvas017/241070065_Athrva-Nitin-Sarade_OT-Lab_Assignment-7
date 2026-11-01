# ANTIGRAVITY MASTER PLAN: Assignment 8 "Optimize Anything" (Time-boxed: 2–3 hours)

**Student:** Athrva Nitin Sarade | **Roll No:** 241070065 | **Course:** Optimization Techniques (VJTI)
**Global random seed for ALL experiments:** `241070065`

---

## 0. HOW YOU (THE AGENT) MUST WORK

1. **Time budget is 2–3 hours total.** Do NOT go deep. Each track = a small, focused, runnable experiment (target 8–15 min each). If something is taking >15 min, implement the minimal version, log it in `LIMITATIONS`, and move on.
2. **Never fabricate results.** Every number in the report must come from code that actually ran. If something could not be run (e.g., real quantum hardware, xv6 if QEMU is missing), say so clearly and use the fallback listed below.
3. **Honesty rules from the assignment:** no claimed speedup without measurement; no theoretical guarantee unless assumptions hold; stochastic methods use ≥5 repeated runs with mean ± std; label every quantum/parallel result as *theoretical / simulated / measured*.
4. **Keep it lightweight:** small problem sizes, few iterations, CPU only, no large downloads. Prefer pure Python + NumPy/SciPy/Matplotlib. Pin versions in `requirements.txt`.
5. **Auto-log everything for the AI-disclosure requirement** (see Section 3). Log as you go, not at the end.
6. **Ask the user only if truly blocked.** Otherwise pick the sensible default and note the assumption.

---

## 1. REPOSITORY STRUCTURE (create first)

```
optimize-anything-241070065/
├── README.md                  # setup, run steps, AI tools register, coverage checklist link
├── requirements.txt
├── run_all.sh                 # runs every track quickly, writes results/
├── COVERAGE_CHECKLIST.md      # Chapter 6 item -> experiment -> code path -> result
├── FAILURE_LOG.md
├── TOOL_COMPARISON.md
├── REPORT_DATA.md             # <<< FINAL OUTPUT the user will paste into Claude for report generation
├── prompts/
│   ├── trackA/ ... trackJ/    # each: prompt.md, ai_response_excerpt.md, revised_prompt.md, verification_note.md, commit link
├── trackA_continuous/
├── trackB_discrete/
├── trackC_quantum_opt/
├── trackD_parallel_distributed/
├── trackE_case_study/
├── trackF_compiler/
├── trackG_xv6/
├── trackH_dbms/
├── trackI_quantum_algos_qml/
├── trackJ_software_opt/
└── results/                   # csv, png plots, logs
```

Git rules: initialize repo, commit after each track. Prefix AI-generated commits `[AI:antigravity-<model>]`; prefix any manual edit by the user `[MANUAL]`. Use real model name shown in the IDE.

---

## 2. TRACK-BY-TRACK SPEC (minimal-but-complete)

Every track folder must contain: `README.md` (objective, baseline, method, how to run), code, `tests/` (at least 2 correctness checks), and saved results (csv/png). Every track must report: **objective, baseline, fair comparison, correctness check.**

### Track A — Continuous optimization (~15 min)
- Objectives: (1) convex quadratic, (2) strongly convex, (3) non-convex (Rosenbrock + a function with a saddle e.g. `x² − y²` and a double-well with local minima). Show landscape plots for convex vs non-convex, local/global minima, saddle point.
- Implement from scratch in NumPy: GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam. Plus Newton and Gauss–Newton (on a small nonlinear least-squares fit).
- Small ML task: logistic regression or 2-layer MLP on a tiny dataset (sklearn digits/moons) with **manual backpropagation** (verify with numerical gradient check). Show ML-as-optimization: loss, parameters, L2 regularization effect.
- Curvature: Hessian-vector product (finite-difference or autograd), diagonal Fisher approximation, one low-rank approximation demo.
- Show: convergence plots (loss vs iterations), step-size sensitivity (3 learning rates), noisy objective (add Gaussian noise to gradients), final objective table + runtime.
- Correctness: gradient check, Newton converges in 1 step on a quadratic.

### Track B — Discrete optimization (~15 min)
- Problem: **0/1 knapsack-style selection** (also reuse for feature selection) and a **Max-Cut on a small graph (≈10–12 nodes, seeded random)**.
- Methods: Greedy, DP (knapsack), ILP (`scipy.optimize.milp` or PuLP), Branch-and-Bound (hand-written, small), brute force as ground truth for small n.
- Formulate Max-Cut (and a feature-selection or portfolio selection) as **QUBO**, convert binary→Ising spin, show penalty term for a constraint (e.g., cardinality constraint). Solve QUBO classically (brute force + simulated annealing) — **save the optimal cut/energy; Track C uses it as baseline.**
- Correctness: all exact methods agree with brute force; greedy gap reported.

### Track C — Quantum optimization (~15 min)
- Use Qiskit (+ Aer simulator) or PennyLane `default.qubit`. **Simulation only** (state this clearly).
- QAOA on the Track B Max-Cut (p=1,2), compare to classical optimum and approximation ratio.
- VQE on a small Hamiltonian (e.g., 2-qubit H₂-like or Ising) with a PQC ansatz.
- Optimizers: parameter-shift gradient, SPSA, quantum natural gradient (PennyLane `QNGOptimizer`), vs COBYLA/Adam baseline. A tiny QNN/PQC classifier is acceptable.
- Show: effect of finite shots (e.g., 100/1000/10000), a depolarizing noise model, and barren-plateau gradient-variance vs number of qubits (e.g., 2–8 qubits, random circuits, ≥20 samples each).
- Note limits of embedding QUBO on annealers (connectivity/minor-embedding) in a short paragraph (no hardware needed).

### Track D — Submodular, parallel, distributed (~15 min)
- Submodular: facility-location or coverage objective on synthetic data; verify diminishing returns numerically on random sets; greedy vs brute force on small instance; check the (1−1/e) guarantee applies (monotone + submodular + cardinality constraint) and report actual ratio.
- Parallel: `multiprocessing` on a CPU-bound task (e.g., Monte-Carlo or independent job batch). Compare static partition vs dynamic queue / work-stealing-style scheduling on uneven tasks. Report makespan, speedup, parallel efficiency for 1,2,4,… cores (up to available), relate to Amdahl and Gustafson. Show a data-parallel example; describe model/pipeline parallelism with a small simulated pipeline (timing model) and label it *simulated*.
- Distributed: **FedAvg vs centralized** on a small logistic regression with non-IID client split (vary local steps ∈ {1,5,20} and rounds). **MGDA and GradNorm vs equal weighting** on a two-task toy model (NumPy or PyTorch CPU).

### Track E — Main case study (~20 min)  *(domain suggestion; may be changed)*
- **Default domain: Exam invigilation/timetable scheduling for a college** (not a systems domain, so it differs from F–J). Decision variables: assignment of exams to slots/rooms; constraints: no student clash, room capacity, max exams per day; objective: minimize clashes + number of slots + spread penalty.
- Synthetic dataset with documented size (e.g., 30 exams, 200 students, 6 rooms, seed 241070065).
- Baseline: naive sequential first-fit assignment.
- Compare ≥2 methods: Greedy graph-colouring (DSATUR) vs ILP vs Simulated Annealing/Genetic Algorithm (≥5 runs, mean±std).
- Include **strategy-selection table** (variable type, size, exact/heuristic, hardware, parallelism) and a short note on why some Chapter 6 methods (e.g., Adam, QAOA) are unsuitable here.

### Track F — Compiler optimization (~20 min)
- Mini-compiler in Python for a toy language (arithmetic, variables, `if`, `while`): lexer → parser → three-address IR → simple VM/code generator.
- Implement **4 passes**: constant folding, constant propagation, dead-code elimination, common sub-expression elimination (optionally strength reduction).
- Optimization formulation: **register allocation as graph colouring (greedy)** or **pass ordering** solved with greedy/DP/brute-force over orderings; compare against naive.
- Measure: instruction count, runtime on VM, code size, compile time on ≥4 test programs. **Verify optimized output == unoptimized output** for every program.

### Track G — xv6 optimization (~25 min, with fallback)
- Check for `qemu-system-riscv64` and RISC-V toolchain (or x86 xv6). If installable quickly (WSL/Linux), clone `mit-pdos/xv6-riscv` and implement **one feature**: a priority or lottery scheduler (+ a `getpinfo`-style syscall) with a user-level benchmark program over **3 workloads** (CPU-bound, I/O-bound, mixed). Measure turnaround/response time before and after. Run `usertests` (or quick subset if full is too slow) and confirm boot. Export `git diff` as `xv6.patch`.
- **FALLBACK (if QEMU/toolchain unavailable within ~10 min):** do NOT fake it. Write a faithful **CPU-scheduler simulator in Python** (RR vs priority vs lottery vs MLFQ) over 3 workloads, produce the xv6 patch as *untested draft code* in `trackG_xv6/draft_patch/`, and clearly label in the report: "xv6 not run; simulation only." Tell the user so they can run it themselves.
- Note OS concepts (scheduling, context switch) and objectives (turnaround, response, fairness). Document at least one AI mistake found in kernel code if any.

### Track H — DBMS optimization (~20 min)
- Use **SQLite** (no setup) or PostgreSQL if already installed. Synthetic TPC-H-like dataset (document size, e.g., 100k–500k rows) with seed 241070065.
- Formulate **index selection** as knapsack/ILP under storage budget: candidate indexes from a workload of ~8–10 queries; compare **greedy (benefit/cost) vs ILP/DP vs default (no indexes)**. Also demonstrate a **join-order** choice via DP vs greedy on one multi-join query (can be a small cost-model + real timing).
- Record `EXPLAIN QUERY PLAN` before/after, latency (median of ≥5 runs), storage cost. **Assert query results identical** before/after.

### Track I — Quantum algorithms & QML (~20 min)
- Simulator only (+ a hardware **noise model** instead of real hardware; state that no real hardware was used unless the user provides IBM credentials).
- Implement **by hand (own circuits, not library black-boxes)** at least: Deutsch–Jozsa and Bernstein–Vazirani, plus Grover (own oracle + diffuser). Also run QFT, Quantum Phase Estimation (small), and a toy/explained Shor (factor 15) using library or hand-built circuits; quantum annealing → describe + small simulated annealing analogue/QUBO.
- QML vs classical: variational classifier **or** quantum kernel (small PennyLane/Qiskit) vs SVM/logistic regression on the same small dataset (e.g., 2D moons or reduced iris). Report accuracy, training time, parameter count, number of circuit evaluations.
- Algorithmic advantage: **Grover vs classical linear search** — query counts (√N vs N) and wall-clock (expected: simulator is slower; report honestly).
- Analysis: when quantum helps / does not (simulation overhead, noise, barren plateaus, data loading) and contrast with Track D speedups. An honest negative result is full credit.

### Track J — Software-engineering optimization (~20 min)
- Choose a **public open-source Python project ≥1,000 lines** (clone small ones, e.g., a utility library) **or** a ≥1,000-line codebase in the repo.
- Default task: **test-suite minimization/prioritization** formulated as set-cover (ILP vs greedy vs naive "run all"), using coverage data (`coverage.py`) and real test timings. Alternative: profiling (`cProfile`) a bottleneck and optimizing it with measured gain.
- Report: baseline profile/time, formulation (variables, objective, constraints), result of greedy vs ILP, coverage preserved, cost in maintainability/correctness.

---

## 3. AI-DISCLOSURE ARTIFACTS (mandatory; keep them cheap)

- **README AI tools register table:** tool, model version, tracks, dates, kind of work. List Antigravity (with the actual model shown in the IDE) and ask the user to add any other tool (ChatGPT/Claude/Gemini etc.) they used, including this planning prompt.
- **prompts/ folder:** per track, save the actual prompt(s) used, a short excerpt of the response, any revised prompt, commit hash, and a 2–3 line verification note. Remind the user to add chat exports/screenshots (cannot be auto-generated). Strip secrets.
- **FAILURE_LOG.md:** at least **3 real** cases where AI-generated code was wrong/unsafe/inefficient/misleading, with fix and lesson. Record them when they actually happen (e.g., failed gradient check, wrong QUBO sign, SQL rewrite changing results). If fewer than 3 occurred naturally, deliberately review the code skeptically to find genuine issues; **do not invent fake failures.**
- **TOOL_COMPARISON.md:** compare ≥2 AI tools on one small task (e.g., Track B QUBO). Antigravity can fill its own half; leave a clearly marked section for the user to run the same prompt in a second tool and paste results. Do not fabricate the second tool's output.
- **Viva note:** add `VIVA_NOTES.md` — for each track, a 5-line explanation of the code a student must be able to explain and modify live, plus 2 likely "modify this live" tasks.

---

## 4. COVERAGE CHECKLIST (`COVERAGE_CHECKLIST.md`)

Table with columns: *Chapter 6 item | Track | Experiment/Code path | Result summary | Status (done / simulated / partial / not applicable-with-reason)*. Must include every item: ML-as-optimization, continuous vs discrete, convex vs non-convex, local/global/saddle, GD, SGD, mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Gauss–Newton, backprop, Fisher/diag/low-rank/HVP, convergence cases, step-size tuning, noisy objectives, Greedy, ILP, B&B, DP, QUBO/Ising, QAOA, VQE, PQC/QNN, parameter-shift, SPSA, QNG, Deutsch–Jozsa, Bernstein–Vazirani, Grover, QFT, QPE, Shor, annealing, variational classifier, quantum kernel, barren plateaus, shots/noise, submodular, parallel (load balancing/makespan/work stealing/Amdahl/Gustafson, data/model/pipeline), strategy selection, FedAvg, MGDA, GradNorm.
**Note for the user:** the final list must be reconciled with the exact syllabus taught in class.

---

## 5. SPEED RULES

- Use tiny sizes + fixed seeds; each script should run < 2 minutes. `run_all.sh` should finish in < 15 minutes total.
- Run independent tracks with parallel sub-tasks/agents if Antigravity supports it (A, B, F, H, J are independent; C depends on B; I is independent).
- Prefer `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `qiskit`/`qiskit-aer` or `pennylane`, `pulp`/`scipy.milp`, `sqlite3`, `coverage`. Avoid heavy deep-learning frameworks (use PyTorch CPU only if truly needed for FedAvg/MGDA/GradNorm).
- Save plots as PNG (dpi ≈ 120) in `results/`.

---

## 6. FINAL DELIVERABLE: `REPORT_DATA.md` (CRITICAL)

At the very end, generate **one single file `REPORT_DATA.md`** that the user will paste into Claude to generate the technical report. It must be self-contained, factual, and contain only measured/true information. Structure:

```
# REPORT_DATA — Assignment 8 (Roll 241070065)
## 0. Environment: OS, CPU cores, RAM, Python version, library versions, seed, date, total wall-clock time spent
## 1. Repository link/structure (placeholder for GitHub URL)
## 2. For EACH track A–J:
   - Objective and process chosen
   - Formulation (decision variables, objective, constraints) in plain text/LaTeX
   - Baseline (what, dataset/size, conditions)
   - Methods compared and why
   - Results TABLE (copy actual numbers: objective, runtime, mean±std over runs, units)
   - Key plots (file names + 1-line description of what each shows)
   - Correctness/verification evidence (tests run, pass/fail)
   - Trade-offs, failures, limitations
   - Label: measured / simulated / theoretical
   - Code location
## 3. Track E strategy-selection table + case-study narrative
## 4. Coverage checklist (copy of COVERAGE_CHECKLIST.md)
## 5. AI tools register, failure log (≥3), tool comparison, commit-prefix convention summary
## 6. Draft answers to the 10 Final Questions (Section 11 of the assignment), each 3–6 sentences grounded in the results above
## 7. Items NOT done / skipped / simulated-instead-of-real (honest list, e.g., xv6, real quantum hardware)
## 8. Items the user must do manually: add chat screenshots/exports, run second AI tool for comparison, record demo video, push to public GitHub, review viva notes
```
Keep this file concise (target ≤ 2,500 lines of content, tables over prose). No invented numbers.

---

## 7. SUGGESTED TIMELINE

| Time | Work |
|---|---|
| 0:00–0:10 | Repo scaffold, env, requirements, check QEMU/Qiskit availability |
| 0:10–0:40 | Tracks A, B, (C) |
| 0:40–1:10 | Tracks D, E |
| 1:10–1:50 | Tracks F, H, J |
| 1:50–2:20 | Tracks G, I |
| 2:20–2:45 | Coverage checklist, failure log, tool comparison, viva notes, README |
| 2:45–3:00 | Generate `REPORT_DATA.md`, final `run_all.sh` sanity run, summary of what to do manually |

**When finished, print:** (1) list of what ran for real vs simulated vs skipped, (2) the path of `REPORT_DATA.md`, (3) the manual to-do list for the student.