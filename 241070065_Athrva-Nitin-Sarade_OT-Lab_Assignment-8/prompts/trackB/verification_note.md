# Verification Note for Track B

- Knapsack exact methods agreement: Brute Force, DP, B&B, and ILP all returned identical optimal value 520 (Greedy gap 0.00%).
- Max-Cut verified: Brute force optimal cut = 15 edges out of 19.
- QUBO-Ising equivalence tested across all $2^5 = 32$ configurations with maximum discrepancy $< 10^{-8}$.
- Pytest test suite: 3/3 passed in 3.20s.
