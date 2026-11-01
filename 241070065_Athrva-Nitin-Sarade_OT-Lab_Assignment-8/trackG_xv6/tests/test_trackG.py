"""
Unit Tests for Track G: Operating System CPU Scheduling
"""

import pytest
from trackG_xv6.run import (
    Process,
    Schedulers,
    generate_workloads
)

def test_all_processes_terminate():
    procs = [
        Process(pid=1, arrival_time=0, burst_time=10),
        Process(pid=2, arrival_time=2, burst_time=5),
        Process(pid=3, arrival_time=4, burst_time=8)
    ]
    res_rr = Schedulers.round_robin(procs, quantum=3)
    res_prio = Schedulers.priority_preemptive(procs)
    res_lottery = Schedulers.lottery(procs, quantum=3)
    res_mlfq = Schedulers.mlfq(procs)

    for res in [res_rr, res_prio, res_lottery, res_mlfq]:
        assert res["avg_turnaround"] > 0
        assert res["avg_response"] >= 0
        assert 0.0 < res["jains_fairness"] <= 1.0

def test_priority_preemption_order():
    # Process 1 has low priority (10), Process 2 arrives later with high priority (1)
    procs = [
        Process(pid=1, arrival_time=0, burst_time=20, priority=10),
        Process(pid=2, arrival_time=5, burst_time=4, priority=1)
    ]
    res = Schedulers.priority_preemptive(procs)
    # Process 2 should preempt and finish before Process 1 completes
    assert res["context_switches"] >= 2, "High priority process failed to trigger preemption"

def test_jains_fairness_bounds():
    workloads = generate_workloads()
    res = Schedulers.round_robin(workloads["CPU-Bound"], quantum=4)
    assert 0.0 <= res["jains_fairness"] <= 1.0
