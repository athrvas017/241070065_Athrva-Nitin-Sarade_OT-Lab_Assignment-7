"""
Track G: Operating System CPU Scheduling (Simulation Only — xv6 Fallback)
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Process Definition & Workload Generators
# -------------------------------------------------------------

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 10         # Lower number = higher priority
    tickets: int = 20          # For lottery scheduling
    remaining_time: int = 0
    first_run_time: int = -1
    completion_time: int = -1
    io_frequency: int = 0      # Blocks for I/O every X ticks
    io_duration: int = 2       # Ticks spent in blocked state
    blocked_until: int = 0
    current_queue: int = 0     # For MLFQ (0 = highest)

    def __post_init__(self):
        self.remaining_time = self.burst_time

def generate_workloads() -> Dict[str, List[Process]]:
    rng = np.random.RandomState(SEED)
    workloads = {}

    # Workload 1: CPU-Bound (5 long compute jobs, minimal I/O)
    workloads["CPU-Bound"] = [
        Process(pid=i, arrival_time=rng.randint(0, 5), burst_time=rng.randint(25, 45), priority=rng.randint(1, 5), tickets=rng.randint(10, 50))
        for i in range(5)
    ]

    # Workload 2: I/O-Bound (8 short jobs with frequent I/O blocks)
    workloads["I/O-Bound"] = [
        Process(pid=i, arrival_time=rng.randint(0, 10), burst_time=rng.randint(8, 16), priority=rng.randint(1, 5), tickets=rng.randint(10, 50), io_frequency=3, io_duration=2)
        for i in range(8)
    ]

    # Workload 3: Mixed (3 CPU-bound + 4 I/O-bound)
    mixed = []
    for i in range(3):
        mixed.append(Process(pid=i, arrival_time=rng.randint(0, 6), burst_time=rng.randint(30, 50), priority=rng.randint(3, 8), tickets=rng.randint(10, 30)))
    for i in range(3, 7):
        mixed.append(Process(pid=i, arrival_time=rng.randint(0, 6), burst_time=rng.randint(10, 20), priority=rng.randint(1, 4), tickets=rng.randint(30, 60), io_frequency=4, io_duration=2))
    workloads["Mixed"] = mixed

    return workloads

# -------------------------------------------------------------
# 2. CPU Schedulers
# -------------------------------------------------------------

def clone_processes(procs: List[Process]) -> List[Process]:
    import copy
    return [copy.deepcopy(p) for p in procs]

class Schedulers:
    @staticmethod
    def round_robin(processes: List[Process], quantum: int = 4) -> Dict[str, any]:
        procs = clone_processes(processes)
        time_tick = 0
        ready_queue: List[Process] = []
        completed: List[Process] = []
        context_switches = 0
        current_proc = None
        quantum_remaining = quantum

        while len(completed) < len(procs):
            # Arrival check
            for p in procs:
                if p.arrival_time == time_tick and p not in ready_queue and p not in completed:
                    ready_queue.append(p)

            # Unblock I/O processes
            for p in procs:
                if p.blocked_until == time_tick and p not in ready_queue and p not in completed and p != current_proc:
                    ready_queue.append(p)

            # Preemption or dispatch
            if current_proc is None or quantum_remaining == 0 or current_proc.remaining_time == 0:
                if current_proc and current_proc.remaining_time > 0 and current_proc.blocked_until <= time_tick:
                    ready_queue.append(current_proc)
                
                if ready_queue:
                    next_proc = ready_queue.pop(0)
                    if next_proc != current_proc:
                        context_switches += 1
                    current_proc = next_proc
                    quantum_remaining = quantum
                else:
                    current_proc = None

            # Execute tick
            if current_proc:
                if current_proc.first_run_time == -1:
                    current_proc.first_run_time = time_tick
                current_proc.remaining_time -= 1
                quantum_remaining -= 1

                # Check completion
                if current_proc.remaining_time == 0:
                    current_proc.completion_time = time_tick + 1
                    completed.append(current_proc)
                    current_proc = None
                # Check I/O trigger
                elif current_proc.io_frequency > 0 and (current_proc.burst_time - current_proc.remaining_time) % current_proc.io_frequency == 0:
                    current_proc.blocked_until = time_tick + 1 + current_proc.io_duration
                    current_proc = None

            time_tick += 1

        return Schedulers._calculate_metrics(completed, context_switches)

    @staticmethod
    def priority_preemptive(processes: List[Process]) -> Dict[str, any]:
        procs = clone_processes(processes)
        time_tick = 0
        completed = []
        context_switches = 0
        current_proc = None

        while len(completed) < len(procs):
            # Check arrivals and unblocks
            ready = [p for p in procs if p.arrival_time <= time_tick and p.blocked_until <= time_tick and p.remaining_time > 0]

            if ready:
                # Pick lowest priority number (highest priority)
                ready.sort(key=lambda p: (p.priority, p.arrival_time))
                best_proc = ready[0]
                if best_proc != current_proc:
                    context_switches += 1
                    current_proc = best_proc

                if current_proc.first_run_time == -1:
                    current_proc.first_run_time = time_tick
                current_proc.remaining_time -= 1

                if current_proc.remaining_time == 0:
                    current_proc.completion_time = time_tick + 1
                    completed.append(current_proc)
                    current_proc = None
                elif current_proc.io_frequency > 0 and (current_proc.burst_time - current_proc.remaining_time) % current_proc.io_frequency == 0:
                    current_proc.blocked_until = time_tick + 1 + current_proc.io_duration
                    current_proc = None
            else:
                current_proc = None

            time_tick += 1

        return Schedulers._calculate_metrics(completed, context_switches)

    @staticmethod
    def lottery(processes: List[Process], quantum: int = 4, seed: int = SEED) -> Dict[str, any]:
        rng = np.random.RandomState(seed)
        procs = clone_processes(processes)
        time_tick = 0
        completed = []
        context_switches = 0
        current_proc = None
        quantum_remaining = quantum

        while len(completed) < len(procs):
            ready = [p for p in procs if p.arrival_time <= time_tick and p.blocked_until <= time_tick and p.remaining_time > 0]

            if current_proc is None or quantum_remaining == 0 or current_proc.remaining_time == 0:
                if ready:
                    total_tickets = sum(p.tickets for p in ready)
                    pick = rng.randint(0, total_tickets)
                    cum = 0
                    chosen = ready[0]
                    for p in ready:
                        cum += p.tickets
                        if cum > pick:
                            chosen = p
                            break
                    if chosen != current_proc:
                        context_switches += 1
                    current_proc = chosen
                    quantum_remaining = quantum
                else:
                    current_proc = None

            if current_proc:
                if current_proc.first_run_time == -1:
                    current_proc.first_run_time = time_tick
                current_proc.remaining_time -= 1
                quantum_remaining -= 1

                if current_proc.remaining_time == 0:
                    current_proc.completion_time = time_tick + 1
                    completed.append(current_proc)
                    current_proc = None
                elif current_proc.io_frequency > 0 and (current_proc.burst_time - current_proc.remaining_time) % current_proc.io_frequency == 0:
                    current_proc.blocked_until = time_tick + 1 + current_proc.io_duration
                    current_proc = None

            time_tick += 1

        return Schedulers._calculate_metrics(completed, context_switches)

    @staticmethod
    def mlfq(processes: List[Process], boost_interval: int = 50) -> Dict[str, any]:
        """Multi-Level Feedback Queue with 3 priority queues and periodic priority boost."""
        procs = clone_processes(processes)
        time_tick = 0
        completed = []
        context_switches = 0
        current_proc = None
        # Queue quantums: Q0=2, Q1=4, Q2=8
        quantums = {0: 2, 1: 4, 2: 8}
        time_allotted = 0

        while len(completed) < len(procs):
            # Priority Boost: prevent starvation
            if time_tick > 0 and time_tick % boost_interval == 0:
                for p in procs:
                    if p.remaining_time > 0:
                        p.current_queue = 0

            ready = [p for p in procs if p.arrival_time <= time_tick and p.blocked_until <= time_tick and p.remaining_time > 0]

            if current_proc is None or time_allotted == 0 or current_proc.remaining_time == 0:
                if ready:
                    # Pick from highest priority non-empty queue (min current_queue)
                    ready.sort(key=lambda p: (p.current_queue, p.arrival_time))
                    chosen = ready[0]
                    if chosen != current_proc:
                        context_switches += 1
                    current_proc = chosen
                    time_allotted = quantums[current_proc.current_queue]
                else:
                    current_proc = None

            if current_proc:
                if current_proc.first_run_time == -1:
                    current_proc.first_run_time = time_tick
                current_proc.remaining_time -= 1
                time_allotted -= 1

                if current_proc.remaining_time == 0:
                    current_proc.completion_time = time_tick + 1
                    completed.append(current_proc)
                    current_proc = None
                elif time_allotted == 0:
                    # Demote to lower priority queue if used full quantum
                    current_proc.current_queue = min(2, current_proc.current_queue + 1)
                    current_proc = None
                elif current_proc.io_frequency > 0 and (current_proc.burst_time - current_proc.remaining_time) % current_proc.io_frequency == 0:
                    current_proc.blocked_until = time_tick + 1 + current_proc.io_duration
                    current_proc = None

            time_tick += 1

        return Schedulers._calculate_metrics(completed, context_switches)

    @staticmethod
    def _calculate_metrics(completed: List[Process], context_switches: int) -> Dict[str, float]:
        turnarounds = [p.completion_time - p.arrival_time for p in completed]
        responses = [p.first_run_time - p.arrival_time for p in completed]

        avg_turnaround = float(np.mean(turnarounds))
        avg_response = float(np.mean(responses))

        # Jain's Fairness Index on normalized turnaround (turnaround / burst)
        norm_turnarounds = np.array([t / p.burst_time for t, p in zip(turnarounds, completed)])
        n = len(norm_turnarounds)
        jains_fairness = float((np.sum(norm_turnarounds) ** 2) / (n * np.sum(norm_turnarounds ** 2) + 1e-12))

        return {
            "avg_turnaround": round(avg_turnaround, 2),
            "avg_response": round(avg_response, 2),
            "jains_fairness": round(jains_fairness, 4),
            "context_switches": context_switches
        }

# -------------------------------------------------------------
# 3. Main Execution & Comparison
# -------------------------------------------------------------

def run_os_scheduling_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track G: Operating System CPU Scheduling (Simulation Only) ===")

    workloads = generate_workloads()
    schedulers = {
        "Round Robin (q=4)": lambda procs: Schedulers.round_robin(procs, quantum=4),
        "Priority Preemptive": Schedulers.priority_preemptive,
        "Lottery Scheduling": Schedulers.lottery,
        "MLFQ (Adaptive)": Schedulers.mlfq
    }

    full_results = {}
    for wl_name, procs in workloads.items():
        full_results[wl_name] = {}
        print(f"\n--- Workload: {wl_name} ({len(procs)} processes) ---")
        for s_name, s_fn in schedulers.items():
            metrics = s_fn(procs)
            full_results[wl_name][s_name] = metrics
            print(f"  {s_name:22s} | Turnaround: {metrics['avg_turnaround']:5.1f} | Response: {metrics['avg_response']:4.1f} | Fairness: {metrics['jains_fairness']:.3f} | Switches: {metrics['context_switches']:3d}")

    results["scheduling_metrics"] = full_results

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sched_names = list(schedulers.keys())
    x = np.arange(len(sched_names))
    width = 0.25

    # 1. Turnaround Comparison across Workloads
    for idx, (wl_name, data) in enumerate(full_results.items()):
        vals = [data[s]["avg_turnaround"] for s in sched_names]
        axes[0].bar(x + (idx - 1) * width, vals, width, label=wl_name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sched_names, rotation=25, ha='right')
    axes[0].set_ylabel("Average Turnaround Time (ticks)")
    axes[0].set_title("Turnaround Time by Scheduler")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # 2. Response Time Comparison
    for idx, (wl_name, data) in enumerate(full_results.items()):
        vals = [data[s]["avg_response"] for s in sched_names]
        axes[1].bar(x + (idx - 1) * width, vals, width, label=wl_name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sched_names, rotation=25, ha='right')
    axes[1].set_ylabel("Average Response Time (ticks)")
    axes[1].set_title("Response Time (Interactivity)")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    # 3. Context Switches
    for idx, (wl_name, data) in enumerate(full_results.items()):
        vals = [data[s]["context_switches"] for s in sched_names]
        axes[2].bar(x + (idx - 1) * width, vals, width, label=wl_name)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sched_names, rotation=25, ha='right')
    axes[2].set_ylabel("Context Switch Count")
    axes[2].set_title("Context Switching Overhead")
    axes[2].legend()
    axes[2].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackG_scheduling.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["scheduling_plot"] = plot_path

    print("\n=== Track G Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_os_scheduling_experiments()
    print(f"Track G Runtime: {time.time() - t0:.2f}s")
