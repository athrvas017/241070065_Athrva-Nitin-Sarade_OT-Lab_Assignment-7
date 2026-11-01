# Track G Prompt

Implement operating system scheduling optimization:
1. Verify QEMU/RISC-V toolchain availability. If missing, activate fallback: simulate CPU schedulers in Python and write xv6 priority scheduler patch.
2. Implement Round Robin (q=4), Priority Preemptive, Lottery Scheduling, and Multi-Level Feedback Queue (MLFQ).
3. Benchmark across 3 synthetic workloads: CPU-bound, I/O-bound, and Mixed.
4. Measure average turnaround time, response time, context switch overhead, and Jain's Fairness Index.
5. Create draft xv6 patch adding priority scheduling and getpinfo system call.
