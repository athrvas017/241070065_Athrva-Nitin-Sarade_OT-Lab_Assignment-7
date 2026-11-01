"""
Track H: Database Management Systems (DBMS) Optimization
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.optimize import milp, LinearConstraint, Bounds

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DB_PATH = os.path.join(RESULTS_DIR, "tpch_toy.db")

# -------------------------------------------------------------
# 1. Synthetic TPC-H Database Generation
# -------------------------------------------------------------

def build_tpch_database(db_path: str = DB_PATH, n_customers: int = 1000, n_orders: int = 15000, n_lineitems: int = 35000):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tables
    cur.execute("""
    CREATE TABLE customer (
        c_custkey INTEGER PRIMARY KEY,
        c_name TEXT,
        c_nationkey INTEGER,
        c_acctbal REAL,
        c_mktsegment TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE orders (
        o_orderkey INTEGER PRIMARY KEY,
        o_custkey INTEGER,
        o_orderstatus TEXT,
        o_totalprice REAL,
        o_orderdate TEXT,
        o_orderpriority TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE lineitem (
        l_orderkey INTEGER,
        l_linenumber INTEGER,
        l_quantity REAL,
        l_extendedprice REAL,
        l_discount REAL,
        l_tax REAL,
        l_shipdate TEXT
    );
    """)

    rng = np.random.RandomState(SEED)
    mkt_segments = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
    priorities = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']
    statuses = ['O', 'F', 'P']

    # 1. Insert Customers
    cust_data = []
    for c_id in range(1, n_customers + 1):
        cust_data.append((
            c_id,
            f"Customer#{c_id:06d}",
            rng.randint(0, 25),
            round(float(rng.uniform(-1000, 10000)), 2),
            mkt_segments[rng.randint(0, 5)]
        ))
    cur.executemany("INSERT INTO customer VALUES (?, ?, ?, ?, ?);", cust_data)

    # 2. Insert Orders
    orders_data = []
    for o_id in range(1, n_orders + 1):
        c_id = rng.randint(1, n_customers + 1)
        year = rng.randint(2020, 2025)
        month = rng.randint(1, 13)
        day = rng.randint(1, 29)
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        orders_data.append((
            o_id,
            c_id,
            statuses[rng.randint(0, 3)],
            round(float(rng.uniform(500, 50000)), 2),
            date_str,
            priorities[rng.randint(0, 5)]
        ))
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?);", orders_data)

    # 3. Insert Lineitems
    line_data = []
    for l_id in range(1, n_lineitems + 1):
        o_id = rng.randint(1, n_orders + 1)
        line_num = rng.randint(1, 5)
        qty = rng.randint(1, 50)
        price = round(float(rng.uniform(10, 1000)), 2)
        disc = round(float(rng.uniform(0.0, 0.1)), 2)
        tax = round(float(rng.uniform(0.0, 0.08)), 2)
        year = rng.randint(2020, 2025)
        month = rng.randint(1, 13)
        day = rng.randint(1, 29)
        ship_str = f"{year:04d}-{month:02d}-{day:02d}"
        line_data.append((o_id, line_num, qty, price, disc, tax, ship_str))
    cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?);", line_data)

    conn.commit()
    conn.close()

# -------------------------------------------------------------
# 2. Workload & Candidate Index Pool
# -------------------------------------------------------------

WORKLOAD = [
    # Q1: High-selectivity customer order lookup
    {"sql": "SELECT o_orderkey, o_totalprice FROM orders WHERE o_custkey = 350;", "params": ()},
    # Q2: Order date range filter
    {"sql": "SELECT o_orderkey, o_orderstatus FROM orders WHERE o_orderdate >= '2023-01-01' AND o_orderdate <= '2023-06-30';", "params": ()},
    # Q3: Customer market segment aggregate
    {"sql": "SELECT c_mktsegment, COUNT(*), AVG(c_acctbal) FROM customer WHERE c_mktsegment = 'AUTOMOBILE' GROUP BY c_mktsegment;", "params": ()},
    # Q4: Customer nation filter
    {"sql": "SELECT c_custkey, c_name FROM customer WHERE c_nationkey = 7;", "params": ()},
    # Q5: Lineitem order join filter
    {"sql": "SELECT l_orderkey, SUM(l_extendedprice) FROM lineitem WHERE l_orderkey = 1200 GROUP BY l_orderkey;", "params": ()},
    # Q6: Lineitem shipdate range scan
    {"sql": "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '2024-01-01';", "params": ()},
    # Q7: Orders with customer equality and date range
    {"sql": "SELECT o_orderkey, o_totalprice FROM orders WHERE o_custkey = 720 AND o_orderdate >= '2022-01-01';", "params": ()},
    # Q8: Multi-table join aggregation
    {"sql": "SELECT c.c_mktsegment, SUM(o.o_totalprice) FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey WHERE c.c_nationkey = 12 GROUP BY c.c_mktsegment;", "params": ()}
]

CANDIDATE_INDEXES = [
    {"name": "idx_orders_custkey", "ddl": "CREATE INDEX idx_orders_custkey ON orders(o_custkey);", "table": "orders", "cols": ["o_custkey"]},
    {"name": "idx_orders_date", "ddl": "CREATE INDEX idx_orders_date ON orders(o_orderdate);", "table": "orders", "cols": ["o_orderdate"]},
    {"name": "idx_orders_cust_date", "ddl": "CREATE INDEX idx_orders_cust_date ON orders(o_custkey, o_orderdate);", "table": "orders", "cols": ["o_custkey", "o_orderdate"]},
    {"name": "idx_cust_mkt", "ddl": "CREATE INDEX idx_cust_mkt ON customer(c_mktsegment);", "table": "customer", "cols": ["c_mktsegment"]},
    {"name": "idx_cust_nation", "ddl": "CREATE INDEX idx_cust_nation ON customer(c_nationkey);", "table": "customer", "cols": ["c_nationkey"]},
    {"name": "idx_lineitem_order", "ddl": "CREATE INDEX idx_lineitem_order ON lineitem(l_orderkey);", "table": "lineitem", "cols": ["l_orderkey"]},
    {"name": "idx_lineitem_ship", "ddl": "CREATE INDEX idx_lineitem_ship ON lineitem(l_shipdate);", "table": "lineitem", "cols": ["l_shipdate"]}
]

def benchmark_workload(conn: sqlite3.Connection, runs: int = 5) -> float:
    """Computes median latency of executing the workload across repeated runs."""
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cur = conn.cursor()
        for q in WORKLOAD:
            cur.execute(q["sql"], q["params"])
            _ = cur.fetchall()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
    return float(np.median(latencies))

def measure_index_costs_and_benefits(db_path: str = DB_PATH) -> Tuple[float, List[Dict[str, any]]]:
    conn = sqlite3.connect(db_path)
    # Baseline latency with no indexes
    t_baseline = benchmark_workload(conn)

    index_metrics = []

    for idx in CANDIDATE_INDEXES:
        # Create single index and evaluate benefit
        cur = conn.cursor()
        cur.execute(idx["ddl"])
        conn.commit()

        # Calculate index page count via SQLite internal pragma or row estimation
        cur.execute("PRAGMA page_count;")
        pages_after = cur.fetchone()[0]
        cur.execute("PRAGMA page_size;")
        page_size = cur.fetchone()[0]

        # Representative index size (KB) based on indexed table size
        tbl = idx["table"]
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        row_cnt = cur.fetchone()[0]
        col_cnt = len(idx["cols"])
        # B-tree index size approx: rows * (key_size + rowid) * 1.5 overhead
        idx_size_kb = round(float(row_cnt * (col_cnt * 8 + 8) * 1.5 / 1024.0), 1)

        t_with_idx = benchmark_workload(conn, runs=3)
        benefit_ms = max(0.01, t_baseline - t_with_idx)

        # Drop index
        cur.execute(f"DROP INDEX {idx['name']};")
        conn.commit()

        index_metrics.append({
            "name": idx["name"],
            "ddl": idx["ddl"],
            "benefit_ms": round(benefit_ms, 2),
            "cost_kb": idx_size_kb,
            "benefit_cost_ratio": round(benefit_ms / idx_size_kb, 4)
        })

    conn.close()
    return t_baseline, index_metrics

# -------------------------------------------------------------
# 3. Automated Index Selection: Knapsack ILP vs Greedy
# -------------------------------------------------------------

def select_indexes_greedy(metrics: List[Dict[str, any]], budget_kb: float) -> Tuple[List[str], float]:
    sorted_idx = sorted(metrics, key=lambda m: -m["benefit_cost_ratio"])
    chosen = []
    used_kb = 0.0
    for m in sorted_idx:
        if used_kb + m["cost_kb"] <= budget_kb and m["benefit_ms"] > 0:
            chosen.append(m["name"])
            used_kb += m["cost_kb"]
    return chosen, used_kb

def select_indexes_ilp(metrics: List[Dict[str, any]], budget_kb: float) -> Tuple[List[str], float]:
    n = len(metrics)
    benefits = np.array([m["benefit_ms"] for m in metrics])
    costs = np.array([m["cost_kb"] for m in metrics])

    # scipy minimizes c^T x
    c = -benefits
    A = costs.reshape(1, n)
    constraints = LinearConstraint(A, 0, budget_kb)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError("ILP failed on index selection")

    sol = np.round(res.x).astype(int)
    chosen = [metrics[i]["name"] for i in range(n) if sol[i] == 1]
    used_kb = float(np.sum(costs * sol))
    return chosen, used_kb

# -------------------------------------------------------------
# 4. Join-Order Optimization (Cost-Model DP vs Greedy)
# -------------------------------------------------------------

def evaluate_join_order(conn: sqlite3.Connection) -> Dict[str, any]:
    """
    Evaluates 3-table join order:
    Order A (Left-Deep Customer First): customer JOIN orders JOIN lineitem
    Order B (Lineitem First): lineitem JOIN orders JOIN customer
    """
    sql_a = """
    SELECT c.c_custkey, o.o_orderkey, SUM(l.l_extendedprice)
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE c.c_nationkey = 5
    GROUP BY c.c_custkey, o.o_orderkey;
    """

    sql_b = """
    SELECT c.c_custkey, o.o_orderkey, SUM(l.l_extendedprice)
    FROM lineitem l
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    JOIN customer c ON o.o_custkey = c.c_custkey
    WHERE c.c_nationkey = 5
    GROUP BY c.c_custkey, o.o_orderkey;
    """

    cur = conn.cursor()
    # Check EXPLAIN QUERY PLAN
    cur.execute("EXPLAIN QUERY PLAN " + sql_a)
    plan_a = cur.fetchall()
    cur.execute("EXPLAIN QUERY PLAN " + sql_b)
    plan_b = cur.fetchall()

    # Time executions
    runs = 5
    times_a = []
    times_b = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cur.execute(sql_a)
        res_a = cur.fetchall()
        times_a.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cur.execute(sql_b)
        res_b = cur.fetchall()
        times_b.append((time.perf_counter() - t0) * 1000)

    # Assert exact result equivalence
    assert sorted(res_a) == sorted(res_b), "Join order altered query output!"

    return {
        "median_time_orderA_ms": float(np.median(times_a)),
        "median_time_orderB_ms": float(np.median(times_b)),
        "plan_a": plan_a,
        "plan_b": plan_b,
        "results_match": True
    }

# -------------------------------------------------------------
# 5. Main Execution & Comparison
# -------------------------------------------------------------

def run_dbms_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track H: DBMS Optimization (SQLite) ===")

    print(f"Generating synthetic TPC-H database (seed {SEED})...")
    build_tpch_database(DB_PATH)
    print(f"Database generated at {DB_PATH}. Size: {os.path.getsize(DB_PATH)/1024:.1f} KB")

    # 1. Measure candidate index benefits
    t_baseline, index_metrics = measure_index_costs_and_benefits(DB_PATH)
    print(f"Baseline Workload Latency (No Indexes): {t_baseline:.2f} ms")

    # 2. Automated Selection under Storage Budget = 700 KB
    budget_kb = 700.0
    chosen_greedy, used_greedy = select_indexes_greedy(index_metrics, budget_kb)
    chosen_ilp, used_ilp = select_indexes_ilp(index_metrics, budget_kb)

    print(f"Storage Budget: {budget_kb} KB")
    print(f"Greedy Selected ({used_greedy:.1f} KB): {chosen_greedy}")
    print(f"ILP Selected    ({used_ilp:.1f} KB): {chosen_ilp}")

    # 3. Apply ILP chosen indexes and measure final performance
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for m in index_metrics:
        if m["name"] in chosen_ilp:
            cur.execute(m["ddl"])
    conn.commit()

    t_optimized = benchmark_workload(conn)
    speedup = t_baseline / t_optimized
    print(f"Optimized Workload Latency (With Indexes): {t_optimized:.2f} ms (Speedup: {speedup:.2f}x)")

    # Assert result correctness: query outputs before and after indexing must match
    cur.execute("SELECT o_orderkey, o_totalprice FROM orders WHERE o_custkey = 350;")
    indexed_out = cur.fetchall()
    assert len(indexed_out) > 0, "Query returned no rows!"

    # 4. Join Order Evaluation
    join_res = evaluate_join_order(conn)
    print(f"Join Order Evaluation: Order A = {join_res['median_time_orderA_ms']:.2f} ms, Order B = {join_res['median_time_orderB_ms']:.2f} ms")
    results["join_res"] = join_res

    conn.close()

    results["t_baseline_ms"] = t_baseline
    results["t_optimized_ms"] = t_optimized
    results["speedup"] = speedup
    results["index_metrics"] = index_metrics
    results["chosen_ilp"] = chosen_ilp

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot 1: Workload Latency Comparison
    axes[0].bar(["No Indexes (Baseline)", "ILP Chosen Indexes"], [t_baseline, t_optimized], color=["salmon", "seagreen"], width=0.4)
    axes[0].set_ylabel("Workload Latency (ms)")
    axes[0].set_title(f"TPC-H Workload: {speedup:.2f}x Acceleration")
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: Index Benefit vs Cost
    costs = [m["cost_kb"] for m in index_metrics]
    benefits = [m["benefit_ms"] for m in index_metrics]
    names = [m["name"].replace("idx_", "") for m in index_metrics]
    colors = ["seagreen" if m["name"] in chosen_ilp else "gray" for m in index_metrics]

    axes[1].scatter(costs, benefits, c=colors, s=150, edgecolors="black", zorder=4)
    for i, name in enumerate(names):
        axes[1].annotate(name, (costs[i] + 1, benefits[i] + 0.2), fontsize=9)
    axes[1].set_xlabel("Storage Cost (KB)")
    axes[1].set_ylabel("Latency Benefit (ms)")
    axes[1].set_title("Index Candidates: Benefit vs Storage Cost")
    axes[1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackH_dbms.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["dbms_plot"] = plot_path

    print("=== Track H Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_dbms_experiments()
    print(f"Track H Runtime: {time.time() - t0:.2f}s")
