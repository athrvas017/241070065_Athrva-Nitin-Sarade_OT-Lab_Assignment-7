# Track D Prompt

Implement submodular, parallel, and distributed optimization:
1. Submodular facility location coverage function: verify diminishing returns numerically, compare greedy vs brute force, and verify the (1 - 1/e) approximation bound.
2. Multiprocessing CPU benchmark: compare 1, 2, 4 cores on uneven workload tasks, comparing static chunking vs dynamic scheduling, and analyze against Amdahl's and Gustafson's Laws.
3. Federated Averaging (FedAvg): simulate decentralized logistic regression across 5 non-IID clients comparing local steps E in {1, 5, 20}.
4. Multi-task optimization: compare Equal Weighting vs MGDA (Multiple Gradient Descent Algorithm) vs GradNorm on conflicting objectives.
5. Simulated 1F1B pipeline parallelism timing model and bubble fraction.
