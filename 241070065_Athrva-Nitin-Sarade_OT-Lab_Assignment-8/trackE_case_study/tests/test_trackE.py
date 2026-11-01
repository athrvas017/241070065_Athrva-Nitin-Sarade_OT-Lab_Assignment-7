"""
Unit Tests for Track E: Case Study (Exam Timetable Scheduling)
"""

import pytest
from trackE_case_study.run import (
    ExamSchedulingProblem,
    solve_naive_first_fit,
    solve_dsatur
)

def test_conflict_graph_symmetry_and_clashes():
    prob = ExamSchedulingProblem(n_exams=10, n_students=50, n_rooms=3, seed=42)
    # Conflict graph must be symmetric
    assert (prob.conflict_graph == prob.conflict_graph.T).all()
    # Diagonal must be zero
    assert (prob.conflict_graph.diagonal() == 0).all()

def test_dsatur_schedule_validity():
    prob = ExamSchedulingProblem(n_exams=20, n_students=100, n_rooms=4, seed=42)
    slots_dsatur, rooms_dsatur = solve_dsatur(prob)
    eval_res = prob.evaluate_solution(slots_dsatur, rooms_dsatur)

    assert eval_res["is_valid"], f"DSATUR solution invalid! Clashes: {eval_res['hard_clashes']}, Violations: {eval_res['capacity_violations']}"
    assert eval_res["hard_clashes"] == 0
    assert eval_res["capacity_violations"] == 0

def test_dsatur_outperforms_or_matches_naive():
    prob = ExamSchedulingProblem(n_exams=25, n_students=120, n_rooms=5, seed=42)
    slots_naive, rooms_naive = solve_naive_first_fit(prob)
    slots_dsatur, rooms_dsatur = solve_dsatur(prob)

    eval_naive = prob.evaluate_solution(slots_naive, rooms_naive)
    eval_dsatur = prob.evaluate_solution(slots_dsatur, rooms_dsatur)

    assert eval_dsatur["is_valid"]
    assert eval_naive["is_valid"]
    assert eval_dsatur["slots_used"] <= eval_naive["slots_used"], "DSATUR should use fewer or equal slots than naive first-fit"
