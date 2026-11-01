# Viva Notes & Live Modification Guide

**Student:** Athrva Nitin Sarade | **Roll No:** 241070065 | **Course:** Optimization Techniques (VJTI)  
**Seed:** `241070065`

This guide prepares the student for live oral examination (viva voce) and on-the-spot code modifications for all tracks A through J.

---

### Track A — Continuous Optimization
- **5-Line Explanation:**
  1. Implements first-order (GD, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam) and second-order (Newton, Gauss-Newton) algorithms from scratch in pure NumPy.
  2. Newton's method uses $x_{k+1} = x_k - H^{-1} \nabla f(x_k)$, solving $H \Delta x = \nabla f$ via Cholesky/LU for stability, achieving exact 1-step quadratic convergence.
  3. Mini-batch backpropagation manually computes $\frac{\partial \mathcal{L}}{\partial W_2} = a_1^T \delta_2$ and $\frac{\partial \mathcal{L}}{\partial W_1} = X^T (\delta_2 W_2^T \odot \sigma'(z_1))$.
  4. Finite-difference Hessian-Vector Product calculates $H v \approx \frac{\nabla f(x+\epsilon v) - \nabla f(x-\epsilon v)}{2\epsilon}$ in $O(n)$ time without constructing $O(n^2)$ matrix.
  5. Tested on convex quadratic, Rosenbrock valley, double-well, and saddle point ($x^2 - y^2$).
- **Live Modification 1:** *Change classical momentum to Nesterov Accelerated Gradient in the optimizer loop.*  
  $\rightarrow$ Evaluate gradient at look-ahead position $x_k + \beta v_k$ rather than $x_k$: $v_{k+1} = \beta v_k - \eta \nabla f(x_k + \beta v_k)$, $x_{k+1} = x_k + v_{k+1}$.
- **Live Modification 2:** *Add L1 regularization (Lasso / Proximal gradient) to the logistic regression training loop.*  
  $\rightarrow$ Perform standard gradient step, then apply soft-thresholding operator: $\text{prox}_{\eta \lambda}(w) = \text{sign}(w) \max(|w| - \eta \lambda, 0)$.

---

### Track B — Discrete Optimization
- **5-Line Explanation:**
  1. Compares Greedy, Dynamic Programming, LP Branch-and-Bound, and ILP for 0/1 Knapsack and Max-Cut.
  2. Knapsack DP builds table $V[i, w] = \max(V[i-1, w], V[i-1, w-w_i] + v_i)$ in $O(nW)$ time.
  3. Branch-and-Bound solves the LP relaxation ($x_i \in [0, 1]$) at each node using fractional greedy bound to prune subtrees whose upper bound $\le$ best known.
  4. Max-Cut is cast as QUBO $\min x^T Q x$ with $Q_{ii} = \sum_j W_{ij}$, $Q_{ij} = -2 W_{ij}$ (or $Q = -L$).
  5. Solved using classical Simulated Annealing and verified against Brute Force exact cut for $N=10$.
- **Live Modification 1:** *Add a cardinality constraint to the QUBO allowing at most $k$ nodes in the cut partition.*  
  $\rightarrow$ Add quadratic penalty $Q \mathrel{+}= \lambda (\mathbf{1}\mathbf{1}^T - 2k I)$, setting diagonal and off-diagonal elements with multiplier $\lambda > \max(\Delta \text{Cut})$.
- **Live Modification 2:** *Modify the Branch-and-Bound branching rule from most fractional to pseudo-cost or highest value first.*  
  $\rightarrow$ In `branch_and_bound`, change node selection index from $\text{argmax}_i |x_i - 0.5|$ to $\text{argmax}_i v_i$.

---

### Track C — Quantum Optimization (QAOA & VQE)
- **5-Line Explanation:**
  1. Implements QAOA on PennyLane's `default.qubit` to solve the Track B Max-Cut QUBO.
  2. Alternates Cost Hamiltonian evolution $e^{-i \gamma H_C}$ (ZZ rotations) and Mixer evolution $e^{-i \beta H_M}$ (X rotations) for $p$ layers.
  3. Evaluates ground state energies via VQE using a hardware-efficient parameterized quantum circuit (PQC).
  4. Optimizes angles $(\gamma, \beta)$ using Parameter-Shift rule $\frac{\partial \langle H \rangle}{\partial \theta} = \frac{\langle H \rangle_{\theta + \pi/2} - \langle H \rangle_{\theta - \pi/2}}{2}$, SPSA, and Quantum Natural Gradient (QNG).
  5. Demonstrates barren plateau phenomenon: gradient variance decays exponentially with number of qubits ($\text{Var} \sim 2^{-n}$).
- **Live Modification 1:** *Increase QAOA depth from $p=1$ to $p=2$ and observe approximation ratio.*  
  $\rightarrow$ Double parameter vectors $\gamma = [\gamma_1, \gamma_2]$, $\beta = [\beta_1, \beta_2]$ and apply two successive Cost-Mixer blocks in the circuit.
- **Live Modification 2:** *Add depolarizing noise channel to each qubit and measure expectation decay.*  
  $\rightarrow$ In the quantum node, insert `qml.DepolarizingChannel(p_noise, wires=w)` after each gate layer.

---

### Track D — Submodular, Parallel & Distributed Optimization
- **5-Line Explanation:**
  1. Implements facility location coverage function $F(S) = \sum_{i \in V} \max_{j \in S} \text{sim}(i, j)$ and proves diminishing marginal gains $F(S \cup \{u\}) - F(S) \ge F(T \cup \{u\}) - F(T)$ for $S \subseteq T$.
  2. Evaluates Greedy $(1 - 1/e) \approx 0.632$ approximation guarantee vs exhaustive brute force.
  3. Benchmarks CPU multiprocessing (`multiprocessing.Pool`) comparing static chunking vs dynamic work-stealing queue on uneven jobs.
  4. Validates Amdahl's Law $S(p) = \frac{1}{(1-s) + s/p}$ and Gustafson's Law $S(p) = p - \alpha(p - 1)$.
  5. Implements Federated Averaging (FedAvg) with non-IID client partitions and multi-gradient Pareto optimization (MGDA / GradNorm).
- **Live Modification 1:** *Change FedAvg local steps from 1 to 20 and observe client drift.*  
  $\rightarrow$ In `fedavg_step`, iterate the local gradient loop 20 times before broadcasting model weights to the central aggregator.
- **Live Modification 2:** *Modify static parallel chunking to dynamic task dispatching with custom chunksize.*  
  $\rightarrow$ In `pool.map(func, items, chunksize=1)` switch to `pool.imap_unordered(func, items, chunksize=1)` and record worker idle variance.

---

### Track E — Main Case Study: College Exam Scheduling
- **5-Line Explanation:**
  1. Solves exam timetable scheduling for 30 exams, 200 students, and 6 rooms under student conflict and capacity constraints.
  2. Formulates decision variables $x_{e, s, r} \in \{0, 1\}$ denoting exam $e$ in slot $s$ and room $r$.
  3. Hard constraints: no student takes two exams in the same slot; total enrollment in room $r$ at slot $s$ $\le \text{Cap}_r$.
  4. Soft objective: minimize total number of slots used + student consecutive exam spread penalties.
  5. Compares Greedy DSATUR (degree of saturation graph coloring) vs Simulated Annealing vs ILP.
- **Live Modification 1:** *Add a hard constraint that a student can take at most 2 exams in any single day (assuming 3 slots per day).*  
  $\rightarrow$ Add constraint $\sum_{s \in \text{Day}_d} \sum_r x_{e, s, r} \le 2$ for all students enrolled in those exams.
- **Live Modification 2:** *Tune Simulated Annealing cooling schedule from geometric ($T_{k+1} = \alpha T_k$) to logarithmic ($T_k = \frac{T_0}{\ln(1 + k)}$).*  
  $\rightarrow$ In the temperature update function, replace `T *= alpha` with `T = T0 / np.log(2 + step)`.

---

### Track F — Mini-Compiler Optimization
- **5-Line Explanation:**
  1. Implements Lexer, Recursive-Descent Parser, Three-Address Code (TAC) IR generator, and a register/stack Virtual Machine.
  2. Passes: Constant Folding (evaluates compile-time literals), Constant Propagation (forwards known values), Dead-Code Elimination (removes unreferenced stores), and Common Sub-expression Elimination (CSE).
  3. Chaitin-style Register Allocation using greedy graph coloring on the variable live-range interference graph.
  4. Evaluates pass ordering permutations to show non-commutativity of optimization passes.
  5. Asserts behavioral equivalence: VM output of unoptimized IR strictly equals optimized IR across all benchmark programs.
- **Live Modification 1:** *Implement algebraic identity simplification in the constant folding pass ($x + 0 \rightarrow x$, $x \times 1 \rightarrow x$, $x \times 0 \rightarrow 0$).*  
  $\rightarrow$ In `fold_constants`, match binary operations where one operand is 0 or 1 and simplify to target variable without emitting instruction.
- **Live Modification 2:** *Add a loop-invariant code motion (LICM) heuristic.*  
  $\rightarrow$ Identify definitions inside a while loop whose operands are defined strictly outside the loop and hoist them before loop entry.

---

### Track G — Operating System CPU Scheduling
- **5-Line Explanation:**
  1. Simulates OS kernel scheduling algorithms: Round Robin (RR with quantum $q$), Preemptive Priority, Lottery Scheduling, and Multi-Level Feedback Queue (MLFQ).
  2. Evaluates across 3 synthetic workloads: CPU-bound (long compute bursts), I/O-bound (frequent short bursts), and Mixed.
  3. Measures turnaround time $T_{\text{turn}} = T_{\text{completion}} - T_{\text{arrival}}$, response time $T_{\text{resp}} = T_{\text{first\_run}} - T_{\text{arrival}}$, and Jain's Fairness Index $\mathcal{J} = \frac{(\sum x_i)^2}{n \sum x_i^2}$.
  4. Provides an untested, modular xv6-riscv kernel patch in `draft_patch/` adding priority scheduler and `getpinfo` system call.
  5. Clearly documents simulation vs physical kernel execution constraints.
- **Live Modification 1:** *Modify Round Robin time slice quantum $q$ from 1 to 10 and observe the trade-off between response time and context switches.*  
  $\rightarrow$ In `simulate_rr(processes, quantum=10)`, change `quantum` and observe context switch count drop while I/O response latency increases.
- **Live Modification 2:** *Implement aging in the Priority Scheduler to prevent starvation.*  
  $\rightarrow$ At each clock tick, increment priority of all unexecuted ready processes: `p.priority = max(1, p.priority - 1)`.

---

### Track H — DBMS Query Optimization
- **5-Line Explanation:**
  1. Uses SQLite with synthetic TPC-H orders/customers/lineitems database (50,000+ rows, seed 241070065).
  2. Formulates automated index selection as 0/1 Knapsack under a disk storage budget (MB).
  3. Compares Greedy benefit-cost ratio ($\Delta \text{Latency} / \text{Size}$) vs exact ILP solver.
  4. Evaluates join ordering using dynamic programming cost model vs naive left-deep join tree.
  5. Analyzes `EXPLAIN QUERY PLAN` showing transition from `SCAN TABLE` to `SEARCH TABLE USING INDEX`, guaranteeing identical result sets.
- **Live Modification 1:** *Force SQLite to bypass index using `INDEXED BY` or unary `+` operator to prove index impact.*  
  $\rightarrow$ Change query `WHERE customer_id = ?` to `WHERE +customer_id = ?` to suppress index usage and measure latency penalty.
- **Live Modification 2:** *Add a composite covering index and verify query runs without visiting the table B-tree.*  
  $\rightarrow$ Create index `CREATE INDEX idx_cov ON orders(customer_id, order_date, total_price)` and verify `EXPLAIN QUERY PLAN` reports `COVERING INDEX`.

---

### Track I — Quantum Algorithms & QML
- **5-Line Explanation:**
  1. Constructs hand-built PennyLane quantum circuits for Deutsch-Jozsa (constant vs balanced oracle in 1 query) and Bernstein-Vazirani (recovers $n$-bit secret).
  2. Implements Grover's search with custom phase oracle and diffusion reflection operator $(2|s\rangle\langle s| - I)$, demonstrating $O(\sqrt{N})$ scaling.
  3. Implements Quantum Fourier Transform (QFT), Quantum Phase Estimation (QPE), and toy Shor's order finding for $N=15$.
  4. Compares Variational Quantum Classifier (PQC) vs classical SVM and Logistic Regression on 2D Moons dataset.
  5. Simulates depolarizing noise and shot noise, contrasting quantum simulation wall-clock overhead with theoretical speedup.
- **Live Modification 1:** *Change the secret bitstring in Bernstein-Vazirani from `'1011'` to `'1100'` and observe output.*  
  $\rightarrow$ Update the inner-product oracle CX gates: apply CX from qubit $i$ to ancilla if and only if bit $s_i == '1'$.
- **Live Modification 2:** *Increase Grover search iterations beyond the optimal $R \approx \frac{\pi}{4}\sqrt{N}$ and show overcooking.*  
  $\rightarrow$ Run Grover for $2R$ iterations; show target state probability drops from $>95\%$ back down towards zero due to unitary rotation past the target.

---

### Track J — Software Engineering Optimization
- **5-Line Explanation:**
  1. Analyzes real test-suite coverage and execution times using Python's `coverage` tool.
  2. Formulates Test Suite Minimization as Minimum Set Cover: select minimum-time test subset $S \subseteq T$ such that $\bigcup_{t \in S} \text{Lines}(t) = \bigcup_{t \in T} \text{Lines}(t)$.
  3. Compares Greedy set cover (marginal lines per second) vs exact ILP (`scipy.optimize.milp`).
  4. Demonstrates 100% statement coverage preservation while eliminating redundant tests.
  5. Profiles codebase bottlenecks using `cProfile` and implements algorithmic caching/vectorization optimizations.
- **Live Modification 1:** *Add a test failure history weighting to prioritize historically fragile tests.*  
  $\rightarrow$ Modify objective coefficient from $c_t = \text{Time}(t)$ to $c_t = \frac{\text{Time}(t)}{1 + \alpha \cdot \text{Failures}(t)}$.
- **Live Modification 2:** *Constrain the test selection to a hard maximum time budget (e.g. 5 seconds) instead of covering all lines.*  
  $\rightarrow$ Invert formulation to Maximum Coverage under Knapsack budget: maximize covered lines subject to $\sum \text{time}_i x_i \le B$.
