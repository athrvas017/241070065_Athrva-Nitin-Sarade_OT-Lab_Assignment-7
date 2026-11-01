# Tool Comparison: AI Assistant Evaluation on Optimization Problem

**Student:** Athrva Nitin Sarade | **Roll No:** 241070065 | **Course:** Optimization Techniques (VJTI)  
**Seed:** `241070065`

## Problem Chosen for Comparison
**Track B QUBO Formulation for Constrained Feature Selection / Max-Cut**:
Prompt given:
> "Formulate the Max-Cut problem on an undirected weighted graph $G=(V, E, W)$ as a Quadratic Unconstrained Binary Optimization (QUBO) problem $\min x^T Q x$ where $x \in \{0, 1\}^n$. Explain the sign convention, construct the $Q$ matrix in terms of degree matrix $D$ and adjacency matrix $A$, map to Ising spins $s \in \{-1, +1\}^n$, and provide the penalty term for adding a cardinality constraint $\sum x_i = k$."

---

### Tool 1: Antigravity (Gemini 3.8 Flash)

**Strengths:**
1. **Mathematical Rigor:** Correctly identified that Max-Cut is naturally a maximization problem, meaning the standard minimization QUBO objective must use $Q = -(D - A) = -L$, or equivalently $Q_{ij} = -2 W_{ij}$ for $i \neq j$ and $Q_{ii} = \sum_{j} W_{ij}$, correctly avoiding the inversion trap.
2. **Ising Spin Mapping:** Accurately derived $s_i = 1 - 2x_i \in \{-1, +1\}$ giving cut Hamiltonian $H_C = \sum_{(i,j) \in E} \frac{W_{ij}}{2}(1 - s_i s_j) = \frac{1}{4} \sum_{(i,j)} W_{ij} - \frac{1}{4} s^T L s$.
3. **Constraint Penalty:** Derived the quadratic penalty $P(x) = \lambda (\sum_{i=1}^n x_i - k)^2 = \lambda (x^T \mathbf{1} \mathbf{1}^T x - 2k \mathbf{1}^T x + k^2)$, and correctly noted that binary $x_i^2 = x_i$ allows folding linear terms into the diagonal of $Q$.
4. **End-to-End Runnable Code:** Provided directly runnable, tested Python code using NumPy and SciPy.

**Weaknesses / Limitations:**
- Initial central finite difference suggestion used machine epsilon instead of $\sqrt{\epsilon_{\text{mach}}}$, requiring empirical calibration.

---

### Tool 2: External Model (ChatGPT / Claude / Gemini Web) — User Comparison Section

*(Student Instructions: Paste the identical prompt above into Claude 3.5 Sonnet / ChatGPT 4o and paste the comparison analysis below.)*

- **Tool Name:** 
- **Model Version:** 
- **Response Correctness:** 
- **Sign Convention Accuracy:** 
- **Code Runnability:** 
- **Key Differences vs Antigravity:** 

---

### Evaluation Criteria Matrix

| Criterion | Antigravity (Gemini 3.8 Flash) | Tool 2 (External) |
| :--- | :--- | :--- |
| **QUBO Sign Accuracy** | Correct ($Q = -L$) | *(To be filled by student)* |
| **Ising Transformation** | Exact ($s_i = 1 - 2x_i$) | *(To be filled by student)* |
| **Penalty Formulation** | Exact ($\lambda (x^T \mathbf{1} - k)^2$) | *(To be filled by student)* |
| **Code Executability** | 100% runnable without edits | *(To be filled by student)* |
| **Response Latency** | Fast (~4s) | *(To be filled by student)* |
