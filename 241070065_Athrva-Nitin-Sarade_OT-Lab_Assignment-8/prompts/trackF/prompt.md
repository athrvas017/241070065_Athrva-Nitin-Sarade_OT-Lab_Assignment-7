# Track F Prompt

Implement a mini-compiler pipeline in Python with Three-Address Code (TAC) IR:
1. Implement virtual machine for interpreting TAC instructions.
2. Implement 4 optimization passes: Constant Folding, Constant Propagation, Common Sub-expression Elimination (CSE), and Dead-Code Elimination (DCE).
3. Implement Chaitin-style Register Allocation using greedy graph coloring on the variable live-range interference graph.
4. Benchmark across 4 distinct programs measuring instruction count reduction and execution steps.
5. Strictly verify unoptimized output == optimized output on every program.
