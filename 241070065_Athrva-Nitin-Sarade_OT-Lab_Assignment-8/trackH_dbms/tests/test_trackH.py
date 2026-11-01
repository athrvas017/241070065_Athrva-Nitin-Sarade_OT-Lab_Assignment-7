"""
Unit Tests for Track H: DBMS Optimization
"""

import os
import sqlite3
import pytest
from trackH_dbms.run import (
    DB_PATH,
    build_tpch_database,
    select_indexes_ilp,
    evaluate_join_order
)

@pytest.fixture(scope="module")
def setup_test_db():
    test_db = os.path.join(os.path.dirname(__file__), "test_db.db")
    build_tpch_database(test_db, n_customers=100, n_orders=500, n_lineitems=1500)
    yield test_db
    if os.path.exists(test_db):
        os.remove(test_db)

def test_query_results_identical_with_index(setup_test_db):
    conn = sqlite3.connect(setup_test_db)
    cur = conn.cursor()

    query = "SELECT o_orderkey, o_totalprice FROM orders WHERE o_custkey = 25;"
    cur.execute(query)
    unindexed_res = cur.fetchall()

    # Create index
    cur.execute("CREATE INDEX IF NOT EXISTS test_idx ON orders(o_custkey);")
    conn.commit()

    cur.execute(query)
    indexed_res = cur.fetchall()

    assert unindexed_res == indexed_res, "Indexing altered query output tuples!"
    conn.close()

def test_ilp_respects_storage_budget():
    metrics = [
        {"name": "idx1", "benefit_ms": 10.0, "cost_kb": 50.0},
        {"name": "idx2", "benefit_ms": 15.0, "cost_kb": 80.0},
        {"name": "idx3", "benefit_ms": 5.0, "cost_kb": 30.0}
    ]
    budget_kb = 100.0
    chosen, used_kb = select_indexes_ilp(metrics, budget_kb)

    assert used_kb <= budget_kb, f"ILP violated storage budget: {used_kb} > {budget_kb}"
    # Best subset under 100 is idx2 (80kb, 15ms) or idx1+idx3 (80kb, 15ms)
    assert len(chosen) >= 1

def test_join_order_semantic_invariance(setup_test_db):
    conn = sqlite3.connect(setup_test_db)
    res = evaluate_join_order(conn)
    assert res["results_match"], "Join reordering altered relational output!"
    conn.close()
