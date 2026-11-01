# Track F: Compiler Optimization

## Objectives
- Build a modular mini-compiler pipeline interpreting Three-Address Code (TAC) Intermediate Representation (IR).
- Implement four foundational scalar and basic-block optimization passes:
  1. **Constant Folding (CF):** Evaluate literal binary expressions at compile time.
  2. **Constant Propagation (CP):** Track known variable values and forward constants.
  3. **Common Sub-expression Elimination (CSE):** Cache compute tuples $(op, arg1, arg2)$ to avoid redundant ALU operations.
  4. **Dead-Code Elimination (DCE):** Strip variable assignments that lack downstream reads.
- Formulate Chaitin-style register allocation as graph coloring on the variable live-range interference graph.
- Demonstrate non-commutativity of compiler pass pipelines (e.g. $CF \to DCE$ vs $DCE \to CF$).
- Verify semantic equivalence ($VM(\text{unoptimized}) \equiv VM(\text{optimized})$) across four diverse benchmark test programs.

## How to Run
```bash
python trackF_compiler/run.py
pytest trackF_compiler/tests/
```
