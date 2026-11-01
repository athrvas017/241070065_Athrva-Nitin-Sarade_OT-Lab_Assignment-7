# Coverage Checklist — Chapter 6 Optimization Techniques

**Student:** Athrva Nitin Sarade | **Roll No:** 241070065 | **Course:** Optimization Techniques (VJTI)  
**Global Random Seed:** `241070065`

| Chapter 6 Item | Track | Experiment / Code Path | Result Summary | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ML-as-Optimization** | Track A | `trackA_continuous/run.py` (2-layer MLP & Logistic Reg) | Verified with numerical gradient check, cross-entropy minimization, L2 reg | Done |
| **Continuous vs Discrete** | Track A vs B | Comparison of gradient landscapes vs combinatorial search | Continuous uses derivatives; discrete uses combinatorial/DP/ILP | Done |
| **Convex vs Non-Convex** | Track A | Quadratic vs Rosenbrock & double-well | Convex finds global min reliably; non-convex sensitive to init | Done |
| **Local / Global / Saddle** | Track A | Landscape analysis of Rosenbrock and $x^2 - y^2$ saddle | Identified local traps, global minima, and escape trajectories | Done |
| **Gradient Descent (GD)** | Track A | Full-batch GD on quadratic and Rosenbrock | Linear convergence on quadratic; oscillations in Rosenbrock valley | Done |
| **Stochastic GD (SGD)** | Track A | Single-sample SGD with noise | Noisy trajectory, converges in expectation with decaying lr | Done |
| **Mini-batch SGD** | Track A | Batch size = 16 on synthetic digits | Balances vectorized throughput and gradient variance | Done |
| **Momentum** | Track A | Polyak classical momentum ($\beta=0.9$) | Dampens orthogonal oscillations in ill-conditioned quadratic | Done |
| **Nesterov Accelerated Grad (NAG)** | Track A | Look-ahead momentum | Faster dampening near minima compared to standard momentum | Done |
| **AdaGrad** | Track A | Adaptive coordinate scaling via $\sum g_t^2$ | Premature stopping on late iterations due to monotonic sum | Done |
| **RMSProp** | Track A | Exponentially decaying average of squared grads | Fixes AdaGrad learning rate decay, handles non-stationary losses | Done |
| **Adam** | Track A | First and second moment estimation with bias correction | Fast, robust convergence across test functions | Done |
| **Newton's Method** | Track A | Full Hessian inversion $H^{-1}\nabla f$ | Quadratic convergence; 1 step on quadratic | Done |
| **Gauss-Newton** | Track A | Nonlinear least squares approximation $(J^TJ)^{-1}J^Tr$ | Eliminates second derivative calculation for sum-of-squares | Done |
| **Backpropagation** | Track A | Manual analytical backprop vs numerical finite differences | Numerical gradient check error $< 10^{-7}$ | Done |
| **Hessian-Vector Product (HVP)** | Track A | Finite-difference HVP $( \nabla f(x+\epsilon v) - \nabla f(x-\epsilon v) ) / (2\epsilon)$ | Exact match with analytical Hessian product | Done |
| **Fisher / Diag / Low-rank** | Track A | Empirical Fisher diagonal & low-rank SVD curvature | Captures dominant curvature directions efficiently | Done |
| **Step-size Tuning** | Track A | Grid of learning rates $[10^{-3}, 10^{-2}, 10^{-1}]$ | Verified stability criteria ($\eta < 2/\lambda_{\max}$) | Done |
| **Noisy Objectives** | Track A | Gaussian noise injection into gradients | Adam and Momentum outperform vanilla GD under noise | Done |
| **Greedy Heuristics** | Track B, D, E, F | Knapsack density, graph coloring, set cover | Fast approximation; bounded greedy gap | Done |
| **Integer Linear Prog (ILP)** | Track B, E, H, J | Exact solver via branch-and-cut (`scipy.optimize.milp`) | Guaranteed global optimum for combinatorial bounds | Done |
| **Branch and Bound (B&B)** | Track B | Hand-written LP-relaxation B&B tree | Matches ILP and brute force; explores pruned subtrees | Done |
| **Dynamic Programming (DP)** | Track B, H | 0/1 Knapsack & join order DP table | Polynomial time pseudo-polynomial optimal solution | Done |
| **QUBO / Ising Spin Model** | Track B, C | Max-Cut QUBO formulation $x^T Q x$ and spin $s_i \in \{-1, +1\}$ | Energy minimized via simulated annealing and exact solver | Done |
| **QAOA** | Track C | Quantum Approximate Optimization Algorithm ($p=1, 2$) | Simulated on Pennylane `default.qubit`; approximation ratio computed | Simulated |
| **VQE** | Track C | Variational Quantum Eigensolver on 2-qubit Hamiltonian | Finds ground state energy using parameterized ansatz | Simulated |
| **PQC / QNN** | Track C, I | Parameterized Quantum Circuit classifier | Trained via classical gradient descent on quantum expectations | Simulated |
| **Parameter-Shift Rule** | Track C, I | Exact analytical gradients $\frac{f(\theta+s) - f(\theta-s)}{2\sin s}$ | Validated against finite differences | Simulated |
| **SPSA** | Track C | Simultaneous Perturbation Stochastic Approximation | 2 evaluations per gradient step regardless of parameter dimension | Simulated |
| **Quantum Natural Gradient (QNG)** | Track C | Fubini-Study metric tensor inversion | Geometry-aware optimization on quantum state manifold | Simulated |
| **Deutsch-Jozsa Algorithm** | Track I | Hand-built circuit distinguishing constant vs balanced oracle | 1 query deterministic evaluation vs $2^{n-1}+1$ classical | Simulated |
| **Bernstein-Vazirani Algorithm** | Track I | Hand-built circuit finding hidden bitstring $s$ | 1 query discovery of $n$-bit secret string | Simulated |
| **Grover Search** | Track I | Hand-built oracle + diffusion operator | $O(\sqrt{N})$ query scaling; verified probabilities | Simulated |
| **Quantum Fourier Transform (QFT)**| Track I | Hand-built phase kickback and Hadamard cascade | Verified periodicity detection on frequency states | Simulated |
| **Quantum Phase Estimation (QPE)** | Track I | Unitary eigenvalue estimation using controlled rotations + QFT | Accurate phase readout for unitary operators | Simulated |
| **Shor's Algorithm (Toy)** | Track I | Factorization of $N=15$ via order finding $a^r \equiv 1 \pmod N$ | Period $r=4$ found, yielding factors 3 and 5 | Simulated |
| **Quantum Annealing & Limitations** | Track C, I | Simulated annealing on QUBO + Chimera/Pegasus embedding discussion | Explains minor-embedding overhead and graph sparsity limits | Theoretical / Sim |
| **Variational Classifier vs Classical**| Track I | PQC classifier vs SVM / Logistic Regression on 2D Moons | Compares decision boundaries, train time, parameter efficiency | Simulated |
| **Quantum Kernel Methods** | Track I | Quantum fidelity kernel matrix $K_{ij} = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$ | Evaluated with classical SVM; shows quantum feature mapping | Simulated |
| **Barren Plateaus** | Track C, I | Gradient variance decay vs number of qubits ($n=2$ to $8$) | Shows $\text{Var}[\partial_\theta \langle H \rangle] \sim \mathcal{O}(2^{-n})$ | Simulated |
| **Finite Shots & Noise Models** | Track C, I | Shot noise ($10^2, 10^3, 10^4$) + depolarizing noise channel | Demonstrates degradation of expectation values and variance | Simulated |
| **Submodular Optimization** | Track D | Facility location / coverage objective | Validated diminishing returns; achieved $(1 - 1/e)$ approximation | Done |
| **Parallel Processing & Efficiency** | Track D | Multi-core CPU multiprocessing (1, 2, 4 cores) | Amdahl's Law and Gustafson's Law evaluation; speedup measured | Done |
| **Work Scheduling** | Track D | Static partition vs Dynamic queue chunking | Demonstrates reduced worker idle time on skewed task times | Done |
| **Distributed Optimization (FedAvg)** | Track D | Federated Averaging across non-IID clients (local steps $\in \{1, 5, 20\}$) | Analyzed communication rounds vs local drift | Done |
| **Multi-Task Opt (MGDA & GradNorm)**| Track D | Multiple Gradient Descent Algorithm vs GradNorm vs Equal Weighting | Pareto stationary point finding vs gradient scale balancing | Done |
| **Case Study (Scheduling)** | Track E | College exam invigilation / timetable scheduling under hard/soft constraints | DSATUR vs Simulated Annealing vs ILP | Done |
| **Compiler Optimization** | Track F | Mini-compiler: Lexer/Parser/IR/VM + 4 passes + register allocation | Verified identical execution output; instruction reduction measured | Done |
| **Operating System Scheduling** | Track G | CPU Scheduler simulator: Round Robin vs Priority vs Lottery vs MLFQ | Evaluated turnaround, response, fairness over 3 workloads | Simulated |
| **Database Query Optimization** | Track H | SQLite synthetic TPC-H: Index selection (Greedy/ILP) + Join ordering | `EXPLAIN QUERY PLAN`, latency reduction, verified query equivalence | Done |
| **Software Eng Optimization** | Track J | Test suite minimization via set cover (Greedy vs ILP) on real coverage | Preserves 100% statement coverage while reducing execution time | Done |
