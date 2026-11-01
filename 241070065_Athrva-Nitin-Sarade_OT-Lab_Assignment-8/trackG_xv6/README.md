# Track G: Operating System CPU Scheduling (Simulation Only)

> [!NOTE]
> **Environment Note:** RISC-V toolchain / `qemu-system-riscv64` is not installed on this host environment. In accordance with Section 2 fallback rules, xv6 was not executed natively. Instead, a faithful kernel CPU scheduling simulator was built and executed in Python, and the accompanying xv6-riscv kernel priority scheduler patch is provided as an untested reference in `draft_patch/`.

## Objectives
- Simulate operating system process scheduling algorithms:
  1. **Round Robin (RR):** Preemptive time-slicing ($q=4$).
  2. **Priority Preemptive:** Priority-driven dispatch with preemption.
  3. **Lottery Scheduling:** Randomized proportional-share scheduling via ticket lottery.
  4. **Multi-Level Feedback Queue (MLFQ):** Multi-queue adaptive aging with periodic priority boosting to eliminate starvation.
- Evaluate across three distinct workloads: CPU-Bound, I/O-Bound, and Mixed.
- Quantitatively evaluate Average Turnaround Time, Average Response Time, Context Switch Overhead, and Jain's Fairness Index $\mathcal{J} = \frac{(\sum x_i)^2}{n \sum x_i^2}$.

## How to Run
```bash
python trackG_xv6/run.py
pytest trackG_xv6/tests/
```
