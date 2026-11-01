"""
Track E: Main Case Study — College Exam Timetable & Invigilation Scheduling
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Problem Formulation & Synthetic Dataset Generator
# -------------------------------------------------------------

class ExamSchedulingProblem:
    def __init__(self, n_exams: int = 30, n_students: int = 200, n_rooms: int = 6, seed: int = SEED):
        rng = np.random.RandomState(seed)
        self.n_exams = n_exams
        self.n_students = n_students
        self.n_rooms = n_rooms

        # Room capacities: 6 rooms with varied capacity
        self.room_capacities = [55, 45, 40, 35, 30, 25]

        # Student enrollments: each student takes 3 to 5 exams
        self.exam_enrollments: Dict[int, Set[int]] = {e: set() for e in range(n_exams)}
        self.student_exams: Dict[int, List[int]] = {s: [] for s in range(n_students)}

        for s in range(n_students):
            k = rng.randint(3, 6)
            chosen_exams = rng.choice(n_exams, size=k, replace=False)
            self.student_exams[s] = list(chosen_exams)
            for e in chosen_exams:
                self.exam_enrollments[e].add(s)

        self.exam_sizes = [len(self.exam_enrollments[e]) for e in range(n_exams)]

        # Conflict graph: edge if two exams share at least 1 student
        self.conflict_graph = np.zeros((n_exams, n_exams), dtype=int)
        self.shared_students = np.zeros((n_exams, n_exams), dtype=int)
        for e1 in range(n_exams):
            for e2 in range(e1 + 1, n_exams):
                shared = len(self.exam_enrollments[e1] & self.exam_enrollments[e2])
                if shared > 0:
                    self.conflict_graph[e1, e2] = 1
                    self.conflict_graph[e2, e1] = 1
                    self.shared_students[e1, e2] = shared
                    self.shared_students[e2, e1] = shared

        self.total_conflict_edges = int(np.sum(self.conflict_graph) // 2)

    def evaluate_solution(self, exam_slots: List[int], exam_rooms: List[int]) -> Dict[str, any]:
        """
        Evaluates schedule validity, student clashes, room capacity violations,
        total slots used, and student consecutive exam spread penalty.
        """
        hard_clashes = 0
        capacity_violations = 0

        # 1. Student Clashes (Hard)
        for e1 in range(self.n_exams):
            for e2 in range(e1 + 1, self.n_exams):
                if exam_slots[e1] == exam_slots[e2] and self.conflict_graph[e1, e2]:
                    hard_clashes += self.shared_students[e1, e2]

        # 2. Room Capacity & Room Collisions (Hard)
        # Check if two exams are assigned the same room in the same slot
        slot_room_usage = {}
        for e in range(self.n_exams):
            s = exam_slots[e]
            r = exam_rooms[e]
            key = (s, r)
            if key in slot_room_usage:
                capacity_violations += 1  # Room clash
            slot_room_usage[key] = e

            # Capacity check
            if self.exam_sizes[e] > self.room_capacities[r]:
                capacity_violations += (self.exam_sizes[e] - self.room_capacities[r])

        # 3. Consecutive exam spread penalty (Soft)
        consecutive_penalties = 0
        for s, exams in self.student_exams.items():
            assigned_slots = sorted([exam_slots[e] for e in exams])
            for i in range(len(assigned_slots) - 1):
                if assigned_slots[i + 1] - assigned_slots[i] == 1:
                    consecutive_penalties += 1  # 2 consecutive exams

        slots_used = len(set(exam_slots))
        # Total cost: heavy penalty on hard constraints
        total_cost = hard_clashes * 1000 + capacity_violations * 500 + slots_used * 10 + consecutive_penalties * 2

        return {
            "hard_clashes": hard_clashes,
            "capacity_violations": capacity_violations,
            "slots_used": slots_used,
            "consecutive_penalties": consecutive_penalties,
            "total_cost": total_cost,
            "is_valid": (hard_clashes == 0 and capacity_violations == 0)
        }

# -------------------------------------------------------------
# 2. Scheduling Solvers
# -------------------------------------------------------------

def solve_naive_first_fit(prob: ExamSchedulingProblem) -> Tuple[List[int], List[int]]:
    """Greedy sequential assignment."""
    exam_slots = [-1] * prob.n_exams
    exam_rooms = [-1] * prob.n_exams

    for e in range(prob.n_exams):
        size = prob.exam_sizes[e]
        assigned = False
        slot = 0
        while not assigned:
            # Find eligible room in this slot
            for r in range(prob.n_rooms):
                if prob.room_capacities[r] >= size:
                    # Check room not taken in this slot
                    room_taken = any(exam_slots[other] == slot and exam_rooms[other] == r for other in range(prob.n_exams))
                    # Check student conflict
                    has_clash = any(exam_slots[other] == slot and prob.conflict_graph[e, other] for other in range(prob.n_exams))
                    if not room_taken and not has_clash:
                        exam_slots[e] = slot
                        exam_rooms[e] = r
                        assigned = True
                        break
            if not assigned:
                slot += 1
    return exam_slots, exam_rooms

def solve_dsatur(prob: ExamSchedulingProblem) -> Tuple[List[int], List[int]]:
    """
    DSATUR (Degree of Saturation) graph coloring heuristic for exam scheduling.
    Prioritizes exams with highest saturation degree (most distinct colored neighbors).
    """
    exam_slots = [-1] * prob.n_exams
    exam_rooms = [-1] * prob.n_exams

    degrees = np.sum(prob.conflict_graph, axis=1)

    for _ in range(prob.n_exams):
        # Calculate saturation degree for uncolored exams
        uncolored = [e for e in range(prob.n_exams) if exam_slots[e] == -1]
        best_e = -1
        max_sat = -1
        max_deg = -1

        for e in uncolored:
            neighbor_slots = set(exam_slots[nbr] for nbr in range(prob.n_exams) if prob.conflict_graph[e, nbr] and exam_slots[nbr] != -1)
            sat = len(neighbor_slots)
            deg = degrees[e]
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                max_sat = sat
                max_deg = deg
                best_e = e

        # Assign best slot and room to best_e
        size = prob.exam_sizes[best_e]
        slot = 0
        assigned = False
        while not assigned:
            # Sort rooms by fit (tightest room capacity >= exam size)
            sorted_rooms = sorted([r for r in range(prob.n_rooms) if prob.room_capacities[r] >= size],
                                  key=lambda r: prob.room_capacities[r])
            for r in sorted_rooms:
                room_taken = any(exam_slots[o] == slot and exam_rooms[o] == r for o in range(prob.n_exams))
                has_clash = any(exam_slots[o] == slot and prob.conflict_graph[best_e, o] for o in range(prob.n_exams))
                if not room_taken and not has_clash:
                    exam_slots[best_e] = slot
                    exam_rooms[best_e] = r
                    assigned = True
                    break
            if not assigned:
                slot += 1

    return exam_slots, exam_rooms

def solve_simulated_annealing(prob: ExamSchedulingProblem, iters: int = 2500, seed: int = SEED) -> Tuple[List[int], List[int], List[float]]:
    """Simulated Annealing starting from DSATUR initialization to optimize soft constraints."""
    rng = np.random.RandomState(seed)
    # Start with valid DSATUR solution
    curr_slots, curr_rooms = solve_dsatur(prob)
    max_slots = max(curr_slots) + 1

    curr_eval = prob.evaluate_solution(curr_slots, curr_rooms)
    best_slots = curr_slots.copy()
    best_rooms = curr_rooms.copy()
    best_cost = curr_eval["total_cost"]

    cost_history = [best_cost]
    T = 50.0
    alpha = 0.997

    for step in range(iters):
        e = rng.randint(0, prob.n_exams)
        size = prob.exam_sizes[e]

        # Neighbor move: try new slot in [0, max_slots] or new eligible room
        cand_slots = curr_slots.copy()
        cand_rooms = curr_rooms.copy()

        if rng.rand() < 0.5:
            cand_slots[e] = rng.randint(0, max_slots + 1)
        else:
            eligible_rooms = [r for r in range(prob.n_rooms) if prob.room_capacities[r] >= size]
            if eligible_rooms:
                cand_rooms[e] = rng.choice(eligible_rooms)

        cand_eval = prob.evaluate_solution(cand_slots, cand_rooms)
        delta = cand_eval["total_cost"] - curr_eval["total_cost"]

        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-4)):
            curr_slots = cand_slots
            curr_rooms = cand_rooms
            curr_eval = cand_eval
            if curr_eval["total_cost"] < best_cost and curr_eval["is_valid"]:
                best_cost = curr_eval["total_cost"]
                best_slots = curr_slots.copy()
                best_rooms = curr_rooms.copy()

        cost_history.append(curr_eval["total_cost"])
        T *= alpha

    return best_slots, best_rooms, cost_history

# -------------------------------------------------------------
# 3. Execution & Strategy Selection Table
# -------------------------------------------------------------

def run_case_study() -> Dict[str, any]:
    results = {}
    print("=== Running Track E: Main Case Study (Exam Timetable Scheduling) ===")

    prob = ExamSchedulingProblem(n_exams=30, n_students=200, n_rooms=6, seed=SEED)
    print(f"Problem Scale: {prob.n_exams} Exams, {prob.n_students} Students, {prob.n_rooms} Rooms")
    print(f"Conflict Graph Edges: {prob.total_conflict_edges} (Graph Density: {prob.total_conflict_edges / (30*29/2):.2f})")

    # 1. Naive First-Fit
    t0 = time.time()
    slots_naive, rooms_naive = solve_naive_first_fit(prob)
    t_naive = time.time() - t0
    eval_naive = prob.evaluate_solution(slots_naive, rooms_naive)
    print(f"Naive First-Fit: Slots={eval_naive['slots_used']}, Clashes={eval_naive['hard_clashes']}, Spread={eval_naive['consecutive_penalties']}, Time={t_naive*1000:.1f}ms")

    # 2. DSATUR Graph Coloring
    t0 = time.time()
    slots_dsatur, rooms_dsatur = solve_dsatur(prob)
    t_dsatur = time.time() - t0
    eval_dsatur = prob.evaluate_solution(slots_dsatur, rooms_dsatur)
    print(f"DSATUR Heuristic: Slots={eval_dsatur['slots_used']}, Clashes={eval_dsatur['hard_clashes']}, Spread={eval_dsatur['consecutive_penalties']}, Time={t_dsatur*1000:.1f}ms")

    # 3. Simulated Annealing (5 Runs)
    sa_runs = []
    sa_histories = []
    for r in range(5):
        t0 = time.time()
        s_sa, rm_sa, hist = solve_simulated_annealing(prob, iters=2000, seed=SEED + r * 10)
        t_sa = time.time() - t0
        ev_sa = prob.evaluate_solution(s_sa, rm_sa)
        sa_runs.append({
            "run": r + 1,
            "slots_used": ev_sa["slots_used"],
            "consecutive_penalties": ev_sa["consecutive_penalties"],
            "total_cost": ev_sa["total_cost"],
            "is_valid": ev_sa["is_valid"],
            "time_ms": t_sa * 1000
        })
        sa_histories.append(hist)

    sa_slots_mean = np.mean([r["slots_used"] for r in sa_runs])
    sa_slots_std = np.std([r["slots_used"] for r in sa_runs])
    sa_pen_mean = np.mean([r["consecutive_penalties"] for r in sa_runs])
    sa_pen_std = np.std([r["consecutive_penalties"] for r in sa_runs])

    print(f"Simulated Annealing (5 runs): Slots={sa_slots_mean:.2f}±{sa_slots_std:.2f}, Spread={sa_pen_mean:.2f}±{sa_pen_std:.2f}")

    comparison_table = [
        {"Method": "Naive First-Fit", "Slots Used": eval_naive["slots_used"], "Hard Clashes": eval_naive["hard_clashes"], "Consecutive Penalties": eval_naive["consecutive_penalties"], "Valid": eval_naive["is_valid"], "Time (ms)": round(t_naive * 1000, 2)},
        {"Method": "DSATUR Coloring", "Slots Used": eval_dsatur["slots_used"], "Hard Clashes": eval_dsatur["hard_clashes"], "Consecutive Penalties": eval_dsatur["consecutive_penalties"], "Valid": eval_dsatur["is_valid"], "Time (ms)": round(t_dsatur * 1000, 2)},
        {"Method": "Simulated Annealing (5 runs)", "Slots Used": f"{sa_slots_mean:.2f} ± {sa_slots_std:.2f}", "Hard Clashes": 0, "Consecutive Penalties": f"{sa_pen_mean:.2f} ± {sa_pen_std:.2f}", "Valid": True, "Time (ms)": round(np.mean([r['time_ms'] for r in sa_runs]), 2)},
    ]
    results["comparison_table"] = comparison_table

    # Strategy Selection Matrix
    strategy_matrix = [
        {"Dimension": "Decision Variables", "Exam Timetable Scheduling": "Discrete binary x_{e, s, r} in {0, 1}", "Continuous ML (Track A)": "Continuous real weights W in R^d", "Max-Cut (Track B/C)": "Binary spin s_i in {-1, +1}"},
        {"Dimension": "Problem Scale", "Exam Timetable Scheduling": "30 exams, 6 rooms, 200 students", "Continuous ML (Track A)": "10^2 to 10^7 real parameters", "Max-Cut (Track B/C)": "10 to 100 nodes"},
        {"Dimension": "Dominant Method", "Exam Timetable Scheduling": "Graph Coloring (DSATUR) + Metaheuristic (SA)", "Continuous ML (Track A)": "Stochastic First-Order (Adam / SGD)", "Max-Cut (Track B/C)": "QUBO, ILP Branch-and-Cut, QAOA"},
        {"Dimension": "Constraint Nature", "Exam Timetable Scheduling": "Strict combinatorial clashes & capacity bounds", "Continuous ML (Track A)": "Unconstrained or soft L2/L1 penalties", "Max-Cut (Track B/C)": "Unconstrained quadratic (QUBO)"},
        {"Dimension": "Why Adam/QAOA Fails Here", "Exam Timetable Scheduling": "Non-differentiable discrete step functions; zero informative gradient everywhere; qubit overhead for 30x6x12 variables (2,160 qubits) exceeds NISQ limits.", "Continuous ML (Track A)": "N/A (Adam is optimal)", "Max-Cut (Track B/C)": "QAOA maps directly to single-layer cost Hamiltonian"}
    ]
    results["strategy_matrix"] = strategy_matrix

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot 1: Slot Comparison Bar Chart
    labels = ["Naive First-Fit", "DSATUR", "Simulated Annealing"]
    slots = [eval_naive["slots_used"], eval_dsatur["slots_used"], sa_slots_mean]
    penalties = [eval_naive["consecutive_penalties"], eval_dsatur["consecutive_penalties"], sa_pen_mean]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, slots, width, label="Slots Required", color="royalblue")
    axes[0].bar(x + width/2, penalties, width, label="Consecutive Exam Penalties", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Exam Scheduling Solvers Performance")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # Plot 2: Simulated Annealing Convergence Trace
    axes[1].plot(sa_histories[0], label="SA Run 1", color="teal")
    axes[1].set_title("Simulated Annealing Cost Convergence")
    axes[1].set_xlabel("Annealing Steps")
    axes[1].set_ylabel("Total Objective Penalty")
    axes[1].grid(True, ls="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackE_case_study.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["case_study_plot"] = plot_path

    print("=== Track E Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_case_study()
    print(f"Track E Runtime: {time.time() - t0:.2f}s")
