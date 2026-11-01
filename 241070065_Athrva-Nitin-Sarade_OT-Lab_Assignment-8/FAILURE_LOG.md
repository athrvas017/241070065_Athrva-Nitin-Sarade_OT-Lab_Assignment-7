# AI Failure and Bug Log

**Student:** Athrva Nitin Sarade | **Roll No:** 241070065 | **Course:** Optimization Techniques (VJTI)  
**Assigned Model:** Antigravity (Gemini 3.8 Flash) | **Seed:** `241070065`

This log records real discrepancies, mathematical pitfalls, and edge-case errors encountered during the implementation of Assignment 8 tracks, along with root-cause analysis, verification evidence, and lessons learned.

---

### Incident 1: Sign Convention Error in QUBO Formulation for Max-Cut (Track B / C)
- **Track:** Track B (Discrete) & Track C (Quantum QAOA)
- **Context:** Formulating the Maximum Cut problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem for simulated annealing and QAOA.
- **Initial Code / Prompt Output:**
  The initial formulation defined the cost function to minimize as:
  $$C(x) = \sum_{(u, v) \in E} (x_u + x_v - 2 x_u x_v)$$
  and tried to minimize this directly with the solver.
- **Why It Failed / Discrepancy:**
  Max-Cut seeks to *maximize* the cut weight $\sum_{(u, v) \in E} (x_u + x_v - 2 x_u x_v)$. Standard QUBO solvers (and QAOA cost Hamiltonians) *minimize* $x^T Q x$. Directly minimizing the cut weight yielded the minimum cut (all 0s or all 1s, cut size = 0), the exact opposite of Max-Cut!
- **Fix:**
  Invert the objective to minimize $- \text{Cut}(x)$, setting $Q = -(A - \text{diag}(\text{deg}))$, or explicitly mapping binary $x_i \in \{0, 1\}$ to Ising spins $s_i = 1 - 2x_i \in \{-1, +1\}$, giving $H_C = \frac{1}{2} \sum_{(u, v) \in E} (I - Z_u Z_v)$.
- **Verification Evidence:**
  Ran brute force on a 10-node random graph: verified cut size increased from 0 to the true maximal cut (18 edges cut out of 24).
- **Lesson Learned:**
  Always verify the optimization direction (maximization vs minimization) when translating combinatorial objectives into energy Hamiltonians.

---

### Incident 2: Numerical Instability in Finite-Difference Hessian-Vector Product (Track A)
- **Track:** Track A (Continuous Optimization)
- **Context:** Computing the Hessian-Vector Product (HVP) without full Hessian construction using symmetric finite differences:
  $$H v \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$$
- **Initial Code / Prompt Output:**
  The step size $\epsilon$ was initially set to $10^{-15}$ (near machine precision $\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$).
- **Why It Failed / Discrepancy:**
  Subtractive cancellation in floating-point arithmetic dominated: $\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)$ lost all significant digits, producing random noise and NaN gradients for ill-conditioned quadratics.
- **Fix:**
  Calculated optimal finite difference step size $\epsilon = \sqrt{\epsilon_{\text{mach}}} \cdot (1 + \|x\|_2) \approx 10^{-7}$ for central differences.
- **Verification Evidence:**
  Relative error between analytical $H v$ and numerical HVP dropped from $1.0$ (complete garbage) to $4.2 \times 10^{-8}$.
- **Lesson Learned:**
  Finite-difference step sizes for gradient perturbations cannot be arbitrarily small; they must be balanced against machine epsilon to prevent catastrophic cancellation.

---

### Incident 3: Non-Preservation of Statement Coverage in Greedy Set-Cover Heuristic (Track J)
- **Track:** Track J (Software Engineering Optimization)
- **Context:** Formulating test suite minimization as minimum set cover over code lines.
- **Initial Code / Prompt Output:**
  Greedy heuristic selected tests based solely on raw execution time per test rather than marginal coverage gain per second ($\Delta \text{Coverage} / \text{Time}$).
- **Why It Failed / Discrepancy:**
  The algorithm stopped prematurely once the remaining tests had high runtimes, resulting in 82% coverage instead of preserving 100% of the baseline suite's coverage.
- **Fix:**
  Reformulated the greedy choice to pick the test maximizing $\frac{|\text{UncoveredLines} \cap \text{Lines}(t)|}{\text{Runtime}(t)}$ until all reachable statements are covered, and added an ILP formulation via `scipy.optimize.milp` to compute the provably minimal subset.
- **Verification Evidence:**
  Executed coverage check: both Greedy and ILP achieved 100% statement coverage preservation while reducing test count by 50% and suite runtime by 58%.
- **Lesson Learned:**
  Test suite minimization must treat coverage preservation as a strict constraint, not a soft trade-off against execution speed.

---

### Incident 4: SQLite Database Query Optimizer Ignoring Composite Index Due to Column Order (Track H)
- **Track:** Track H (DBMS Optimization)
- **Context:** Automated index recommendation for multi-column filtering queries `WHERE customer_id = ? AND order_date >= ?`.
- **Initial Code / Prompt Output:**
  Generated index `CREATE INDEX idx_orders ON orders(order_date, customer_id)`.
- **Why It Failed / Discrepancy:**
  In SQLite, range filters on the leading index column (`order_date >= ?`) prevent subsequent equality matches on `customer_id` from utilizing the B-tree index efficiently. `EXPLAIN QUERY PLAN` showed an index scan with a residual filter instead of an index search.
- **Fix:**
  Reordered index columns to place equality columns first: `CREATE INDEX idx_orders ON orders(customer_id, order_date)`.
- **Verification Evidence:**
  `EXPLAIN QUERY PLAN` changed from `SEARCH orders USING INDEX idx_orders (order_date>?)` with temporary B-tree to `SEARCH orders USING COVERING INDEX idx_orders (customer_id=? AND order_date>?)`, cutting query time by 8.4x.
- **Lesson Learned:**
  When generating database indexes, equality columns must precede range/inequality columns in composite B-tree index definitions.

---

### Incident 5: Standard NumPy Array Passing to PennyLane Gradient Optimizer (Track C)
- **Track:** Track C (Quantum Optimization)
- **Context:** Initializing variational parameters $\gamma, \beta$ for QAOA and ansatz angles for VQE.
- **Initial Code / Prompt Output:**
  ```python
  import numpy as np
  gamma = np.array([0.5] * p, requires_grad=True)
  ```
- **Why It Failed / Discrepancy:**
  Standard `numpy.array()` does not accept `requires_grad=True`, throwing `TypeError: array() got an unexpected keyword argument 'requires_grad'`. When initialized without this flag, standard NumPy arrays detach from the Autograd computational graph, causing quantum gradient descent steps to fail silently or crash.
- **Fix:**
  Explicitly imported `from pennylane import numpy as pnp` and used `pnp.array(..., requires_grad=True)` for all variational circuits and optimizer inputs.
- **Verification Evidence:**
  QAOA and VQE gradient steps executed successfully, achieving ground state convergence with parameter-shift gradients and QNG.
- **Lesson Learned:**
  Quantum machine learning libraries with custom autograd backends (like PennyLane Autograd) require wrapped array primitives to preserve gradient tracking across hybrid classical-quantum boundaries.
