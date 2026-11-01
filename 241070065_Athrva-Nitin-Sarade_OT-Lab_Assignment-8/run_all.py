"""
Master Execution Script for Assignment 8: Optimize Anything
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import sys
import time
import subprocess
import os

TRACKS = [
    ("Track A (Continuous)", "trackA_continuous/run.py"),
    ("Track B (Discrete)", "trackB_discrete/run.py"),
    ("Track C (Quantum Opt)", "trackC_quantum_opt/run.py"),
    ("Track D (Parallel & Distributed)", "trackD_parallel_distributed/run.py"),
    ("Track E (Case Study - Exam Scheduling)", "trackE_case_study/run.py"),
    ("Track F (Mini-Compiler)", "trackF_compiler/run.py"),
    ("Track G (OS CPU Scheduling)", "trackG_xv6/run.py"),
    ("Track H (DBMS Optimization)", "trackH_dbms/run.py"),
    ("Track I (Quantum Algos & QML)", "trackI_quantum_algos_qml/run.py"),
    ("Track J (Software Engineering)", "trackJ_software_opt/run.py"),
]

def main():
    print("=" * 70)
    print("  ASSIGNMENT 8: OPTIMIZE ANYTHING — MASTER EXPERIMENT RUNNER")
    print("  Student: Athrva Nitin Sarade | Roll: 241070065 | VJTI")
    print("=" * 70)

    python_bin = sys.executable
    total_start = time.time()
    summary = []

    for name, script_path in TRACKS:
        print(f"\n>>> Executing {name} ({script_path})...")
        t0 = time.time()
        ret = subprocess.run([python_bin, "-u", script_path], capture_output=False)
        dur = time.time() - t0
        status = "PASSED" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        print(f">>> {name}: {status} in {dur:.2f}s")
        summary.append((name, status, dur))

    print("\n" + "=" * 70)
    print(">>> RUNNING FULL PYTEST SUITE ACROSS ALL TRACKS...")
    print("=" * 70)
    t0 = time.time()
    ret_test = subprocess.run([python_bin, "-m", "pytest", "-v"], capture_output=False)
    test_dur = time.time() - t0

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("  EXPERIMENT EXECUTION SUMMARY")
    print("=" * 70)
    for name, status, dur in summary:
        print(f"  {name:38s} : {status:10s} ({dur:6.2f}s)")
    print(f"  {'Pytest Verification Suite':38s} : {'PASSED' if ret_test.returncode == 0 else 'FAILED':10s} ({test_dur:6.2f}s)")
    print(f"  Total Wall-Clock Time: {total_time:.2f}s ({total_time/60.0:.1f} minutes)")
    print("=" * 70)

if __name__ == "__main__":
    main()
