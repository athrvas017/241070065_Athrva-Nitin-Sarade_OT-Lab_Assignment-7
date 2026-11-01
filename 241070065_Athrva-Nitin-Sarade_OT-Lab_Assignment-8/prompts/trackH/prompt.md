# Track H Prompt

Implement database query and storage optimization in SQLite:
1. Generate synthetic TPC-H relational dataset (customers, orders, lineitem with >50k total rows, seed 241070065).
2. Formulate automated index selection as 0/1 Knapsack under a disk storage budget.
3. Compare Greedy benefit/cost selection against exact ILP solver (scipy.optimize.milp).
4. Demonstrate join-order optimization via dynamic programming / cost analysis and measure execution latency.
5. Verify EXPLAIN QUERY PLAN demonstrates index usage and assert that query output tuples remain 100% identical.
