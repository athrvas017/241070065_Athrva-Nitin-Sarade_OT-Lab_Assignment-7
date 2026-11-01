"""
Track F: Compiler Optimization
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional

SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Three-Address Code (TAC) IR Definition & Virtual Machine
# -------------------------------------------------------------

# Instruction format: (op, arg1, arg2, dest)
# e.g.: ('ASSIGN', 5, None, 'x')
#       ('ADD', 'x', 'y', 't1')
#       ('JUMP_IF_FALSE', 'cond', None, 'L1')
#       ('LABEL', 'L1', None, None)
#       ('PRINT', 'x', None, None)

class TACInstruction:
    def __init__(self, op: str, arg1: Any = None, arg2: Any = None, dest: Any = None):
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.dest = dest

    def __repr__(self):
        if self.op == 'ASSIGN':
            return f"{self.dest} = {self.arg1}"
        elif self.op in ('ADD', 'SUB', 'MUL', 'DIV', 'LT', 'GT', 'EQ'):
            return f"{self.dest} = {self.arg1} {self.op} {self.arg2}"
        elif self.op == 'LABEL':
            return f"{self.arg1}:"
        elif self.op == 'JUMP':
            return f"goto {self.dest}"
        elif self.op == 'JUMP_IF_FALSE':
            return f"if not {self.arg1} goto {self.dest}"
        elif self.op == 'PRINT':
            return f"print {self.arg1}"
        return f"{self.op} {self.arg1} {self.arg2} -> {self.dest}"

class SimpleVM:
    """Interprets TAC instructions and executes them."""
    def __init__(self):
        self.variables = {}
        self.output = []
        self.step_count = 0

    def run(self, code: List[TACInstruction], max_steps: int = 100000) -> List[Any]:
        self.variables = {}
        self.output = []
        self.step_count = 0

        # Build label table
        labels = {}
        for idx, inst in enumerate(code):
            if inst.op == 'LABEL':
                labels[inst.arg1] = idx

        pc = 0
        n = len(code)

        def eval_val(v):
            if isinstance(v, (int, float)):
                return v
            elif isinstance(v, str):
                return self.variables.get(v, 0)
            return v

        while pc < n:
            self.step_count += 1
            if self.step_count > max_steps:
                raise RuntimeError("VM Infinite loop detected")

            inst = code[pc]
            op = inst.op

            if op == 'ASSIGN':
                self.variables[inst.dest] = eval_val(inst.arg1)
                pc += 1
            elif op == 'ADD':
                self.variables[inst.dest] = eval_val(inst.arg1) + eval_val(inst.arg2)
                pc += 1
            elif op == 'SUB':
                self.variables[inst.dest] = eval_val(inst.arg1) - eval_val(inst.arg2)
                pc += 1
            elif op == 'MUL':
                self.variables[inst.dest] = eval_val(inst.arg1) * eval_val(inst.arg2)
                pc += 1
            elif op == 'DIV':
                denom = eval_val(inst.arg2)
                self.variables[inst.dest] = eval_val(inst.arg1) // (denom if denom != 0 else 1)
                pc += 1
            elif op == 'LT':
                self.variables[inst.dest] = int(eval_val(inst.arg1) < eval_val(inst.arg2))
                pc += 1
            elif op == 'GT':
                self.variables[inst.dest] = int(eval_val(inst.arg1) > eval_val(inst.arg2))
                pc += 1
            elif op == 'EQ':
                self.variables[inst.dest] = int(eval_val(inst.arg1) == eval_val(inst.arg2))
                pc += 1
            elif op == 'LABEL':
                pc += 1
            elif op == 'JUMP':
                pc = labels[inst.dest]
            elif op == 'JUMP_IF_FALSE':
                cond_val = eval_val(inst.arg1)
                if not cond_val:
                    pc = labels[inst.dest]
                else:
                    pc += 1
            elif op == 'PRINT':
                self.output.append(eval_val(inst.arg1))
                pc += 1
            else:
                pc += 1

        return self.output

# -------------------------------------------------------------
# 2. Four Compiler Optimization Passes
# -------------------------------------------------------------

class CompilerOptimizer:
    @staticmethod
    def constant_folding(code: List[TACInstruction]) -> List[TACInstruction]:
        """Evaluates operations where all operands are compile-time literals."""
        new_code = []
        for inst in code:
            if inst.op in ('ADD', 'SUB', 'MUL', 'DIV'):
                if isinstance(inst.arg1, (int, float)) and isinstance(inst.arg2, (int, float)):
                    if inst.op == 'ADD':
                        val = inst.arg1 + inst.arg2
                    elif inst.op == 'SUB':
                        val = inst.arg1 - inst.arg2
                    elif inst.op == 'MUL':
                        val = inst.arg1 * inst.arg2
                    elif inst.op == 'DIV':
                        val = inst.arg1 // inst.arg2 if inst.arg2 != 0 else 0
                    new_code.append(TACInstruction('ASSIGN', val, None, inst.dest))
                    continue
            new_code.append(inst)
        return new_code

    @staticmethod
    def constant_propagation(code: List[TACInstruction]) -> List[TACInstruction]:
        """Replaces variables holding known constants with literal values across basic blocks."""
        new_code = []
        constants = {}
        for inst in code:
            # Clear constants across control flow jumps/labels to remain sound
            if inst.op in ('LABEL', 'JUMP', 'JUMP_IF_FALSE'):
                constants.clear()
                new_code.append(inst)
                continue

            a1 = constants[inst.arg1] if inst.arg1 in constants else inst.arg1
            a2 = constants[inst.arg2] if inst.arg2 in constants else inst.arg2

            if inst.op == 'ASSIGN':
                if isinstance(a1, (int, float)):
                    constants[inst.dest] = a1
                elif inst.dest in constants:
                    del constants[inst.dest]
                new_code.append(TACInstruction('ASSIGN', a1, None, inst.dest))
            elif inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'LT', 'GT', 'EQ'):
                if inst.dest in constants:
                    del constants[inst.dest]
                new_code.append(TACInstruction(inst.op, a1, a2, inst.dest))
            elif inst.op == 'PRINT':
                new_code.append(TACInstruction('PRINT', a1, None, None))
            else:
                new_code.append(inst)
        return new_code

    @staticmethod
    def common_subexpression_elimination(code: List[TACInstruction]) -> List[TACInstruction]:
        """Eliminates redundant identical binary calculations within basic blocks."""
        new_code = []
        expr_cache = {}  # (op, a1, a2) -> dest_var

        for inst in code:
            if inst.op in ('LABEL', 'JUMP', 'JUMP_IF_FALSE'):
                expr_cache.clear()
                new_code.append(inst)
                continue

            if inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'LT', 'GT', 'EQ'):
                expr_key = (inst.op, inst.arg1, inst.arg2)
                # For commutative ops, sort operands if both strings or both numbers
                if inst.op in ('ADD', 'MUL') and str(inst.arg1) > str(inst.arg2):
                    expr_key = (inst.op, inst.arg2, inst.arg1)

                if expr_key in expr_cache:
                    # Replace with assignment from earlier result
                    prev_dest = expr_cache[expr_key]
                    new_code.append(TACInstruction('ASSIGN', prev_dest, None, inst.dest))
                else:
                    expr_cache[expr_key] = inst.dest
                    new_code.append(inst)
            else:
                # Invalidate cache if variable is reassigned
                if inst.dest:
                    expr_cache = {k: v for k, v in expr_cache.items() if k[1] != inst.dest and k[2] != inst.dest and v != inst.dest}
                new_code.append(inst)
        return new_code

    @staticmethod
    def dead_code_elimination(code: List[TACInstruction]) -> List[TACInstruction]:
        """Removes assignments whose dest variables are never read in the program."""
        # Find all used variables
        used = set()
        for inst in code:
            if isinstance(inst.arg1, str) and inst.op != 'LABEL' and inst.op != 'JUMP':
                used.add(inst.arg1)
            if isinstance(inst.arg2, str):
                used.add(inst.arg2)

        new_code = []
        for inst in code:
            # Control flow and prints are never dead
            if inst.op in ('LABEL', 'JUMP', 'JUMP_IF_FALSE', 'PRINT'):
                new_code.append(inst)
            elif inst.dest in used:
                new_code.append(inst)
            else:
                # Dead assignment eliminated
                pass
        return new_code

    @classmethod
    def optimize_pipeline(cls, code: List[TACInstruction], passes: List[str]) -> List[TACInstruction]:
        curr = code
        for p in passes:
            if p == 'CF':
                curr = cls.constant_folding(curr)
            elif p == 'CP':
                curr = cls.constant_propagation(curr)
            elif p == 'CSE':
                curr = cls.common_subexpression_elimination(curr)
            elif p == 'DCE':
                curr = cls.dead_code_elimination(curr)
        return curr

# -------------------------------------------------------------
# 3. Register Allocation via Graph Coloring (Chaitin-style)
# -------------------------------------------------------------

def allocate_registers(code: List[TACInstruction], k_registers: int = 4) -> Dict[str, str]:
    """
    Constructs variable live-range interference graph and solves
    register allocation using greedy graph coloring.
    """
    # 1. Collect all variable names (excluding literals and labels)
    variables = set()
    for inst in code:
        if isinstance(inst.dest, str) and inst.op not in ('LABEL', 'JUMP'):
            variables.add(inst.dest)
        if isinstance(inst.arg1, str) and inst.op not in ('LABEL', 'JUMP'):
            variables.add(inst.arg1)
        if isinstance(inst.arg2, str):
            variables.add(inst.arg2)

    var_list = sorted(list(variables))
    n = len(var_list)
    var_to_idx = {v: i for i, v in enumerate(var_list)}

    # 2. Simple live range estimation: first def to last use
    first_seen = {v: 100000 for v in var_list}
    last_seen = {v: -1 for v in var_list}

    for idx, inst in enumerate(code):
        for v in [inst.dest, inst.arg1, inst.arg2]:
            if isinstance(v, str) and v in var_to_idx:
                first_seen[v] = min(first_seen[v], idx)
                last_seen[v] = max(last_seen[v], idx)

    # 3. Interference graph
    interference = np.zeros((n, n), dtype=int)
    for i in range(n):
        v1 = var_list[i]
        for j in range(i + 1, n):
            v2 = var_list[j]
            # Overlapping live ranges interfere
            if not (last_seen[v1] < first_seen[v2] or last_seen[v2] < first_seen[v1]):
                interference[i, j] = 1
                interference[j, i] = 1

    # 4. Greedy Graph Coloring
    colors = [-1] * n
    for i in range(n):
        neighbor_colors = set(colors[j] for j in range(n) if interference[i, j] and colors[j] != -1)
        # Assign lowest available register
        for c in range(k_registers):
            if c not in neighbor_colors:
                colors[i] = c
                break
        if colors[i] == -1:
            # Spill
            colors[i] = -1

    reg_map = {}
    for i, v in enumerate(var_list):
        reg_map[v] = f"R{colors[i]}" if colors[i] != -1 else "SPILL"

    return reg_map

# -------------------------------------------------------------
# 4. Benchmark Programs & Validation
# -------------------------------------------------------------

def get_benchmark_programs() -> Dict[str, List[TACInstruction]]:
    """Defines 4 benchmark programs in TAC IR."""
    progs = {}

    # Program 1: Arithmetic & Redundancies (Constant folding & CSE & DCE)
    progs["Arithmetic_CSE"] = [
        TACInstruction('ASSIGN', 10, None, 'a'),
        TACInstruction('ASSIGN', 20, None, 'b'),
        TACInstruction('ADD', 10, 20, 'c'),       # Constant folding -> 30
        TACInstruction('ADD', 'a', 'b', 't1'),     # CSE target
        TACInstruction('ADD', 'a', 'b', 't2'),     # Redundant CSE
        TACInstruction('MUL', 't1', 't2', 'res'),
        TACInstruction('ASSIGN', 999, None, 'dead_val'), # DCE target
        TACInstruction('PRINT', 'res', None, None)
    ]

    # Program 2: While loop with accumulator
    progs["Loop_Accumulator"] = [
        TACInstruction('ASSIGN', 0, None, 'sum'),
        TACInstruction('ASSIGN', 1, None, 'i'),
        TACInstruction('ASSIGN', 10, None, 'limit'),
        TACInstruction('LABEL', 'LOOP_START'),
        TACInstruction('LT', 'i', 'limit', 'cond'),
        TACInstruction('JUMP_IF_FALSE', 'cond', None, 'LOOP_END'),
        TACInstruction('ADD', 'sum', 'i', 'sum'),
        TACInstruction('ADD', 'i', 1, 'i'),
        TACInstruction('JUMP', None, None, 'LOOP_START'),
        TACInstruction('LABEL', 'LOOP_END'),
        TACInstruction('PRINT', 'sum', None, None)
    ]

    # Program 3: Geometric Progression with Dead Branches
    progs["Geometric_DCE"] = [
        TACInstruction('ASSIGN', 2, None, 'base'),
        TACInstruction('ASSIGN', 3, None, 'unused_const'), # Dead
        TACInstruction('MUL', 'base', 'base', 'sq'),
        TACInstruction('MUL', 'sq', 'base', 'cube'),
        TACInstruction('ADD', 'unused_const', 5, 'dead2'), # Dead
        TACInstruction('PRINT', 'cube', None, None)
    ]

    # Program 4: Multi-Branch If-Else Condition
    progs["Branching_Logic"] = [
        TACInstruction('ASSIGN', 15, None, 'x'),
        TACInstruction('GT', 'x', 10, 'cond'),
        TACInstruction('JUMP_IF_FALSE', 'cond', None, 'ELSE_L'),
        TACInstruction('MUL', 'x', 2, 'ans'),
        TACInstruction('JUMP', None, None, 'END_L'),
        TACInstruction('LABEL', 'ELSE_L'),
        TACInstruction('DIV', 'x', 2, 'ans'),
        TACInstruction('LABEL', 'END_L'),
        TACInstruction('PRINT', 'ans', None, None)
    ]

    return progs

# -------------------------------------------------------------
# 5. Main Execution & Comparison
# -------------------------------------------------------------

def run_compiler_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track F: Compiler Optimization ===")

    progs = get_benchmark_programs()
    vm = SimpleVM()

    # Pass ordering to compare
    standard_pipeline = ['CP', 'CF', 'CSE', 'DCE']
    
    benchmark_table = []
    
    for name, raw_ir in progs.items():
        # 1. Run unoptimized
        t0 = time.time()
        out_unopt = vm.run(raw_ir)
        t_unopt = time.time() - t0
        unopt_steps = vm.step_count
        unopt_inst_count = len(raw_ir)

        # 2. Run optimized
        opt_ir = CompilerOptimizer.optimize_pipeline(raw_ir, standard_pipeline)
        t0 = time.time()
        out_opt = vm.run(opt_ir)
        t_opt = time.time() - t0
        opt_steps = vm.step_count
        opt_inst_count = len(opt_ir)

        # 3. Assert exact output equivalence
        assert out_unopt == out_opt, f"Optimization altered behavior in {name}! {out_unopt} != {out_opt}"

        # 4. Register allocation
        reg_alloc = allocate_registers(opt_ir, k_registers=4)

        reduction_pct = ((unopt_inst_count - opt_inst_count) / unopt_inst_count) * 100

        benchmark_table.append({
            "Program": name,
            "Unopt Inst": unopt_inst_count,
            "Opt Inst": opt_inst_count,
            "Inst Reduction (%)": round(reduction_pct, 1),
            "Unopt Steps": unopt_steps,
            "Opt Steps": opt_steps,
            "Output": out_opt,
            "Registers Used": len(set(reg_alloc.values())),
            "Equivalence Verified": True
        })

    print(f"Benchmark Results:")
    for row in benchmark_table:
        print(f"  {row['Program']}: {row['Unopt Inst']} -> {row['Opt Inst']} insts ({row['Inst Reduction (%)']}% drop), Output={row['Output']}")
    results["compiler_benchmarks"] = benchmark_table

    # Pass Ordering Exploration (Non-commutativity)
    # Compare: (CF -> DCE) vs (DCE -> CF)
    sample_ir = progs["Arithmetic_CSE"]
    p1 = ['CF', 'DCE']
    p2 = ['DCE', 'CF']
    res_p1 = len(CompilerOptimizer.optimize_pipeline(sample_ir, p1))
    res_p2 = len(CompilerOptimizer.optimize_pipeline(sample_ir, p2))
    print(f"Pass Ordering Effect: CF->DCE results in {res_p1} insts, DCE->CF results in {res_p2} insts")
    results["pass_ordering"] = {"CF_then_DCE": res_p1, "DCE_then_CF": res_p2}

    # Plot
    plt.figure(figsize=(10, 4.5))
    names = [r["Program"] for r in benchmark_table]
    unopt_c = [r["Unopt Inst"] for r in benchmark_table]
    opt_c = [r["Opt Inst"] for r in benchmark_table]

    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, unopt_c, width, label="Unoptimized TAC IR", color="salmon")
    plt.bar(x + width/2, opt_c, width, label="Optimized TAC IR", color="seagreen")
    plt.xticks(x, names)
    plt.ylabel("Instruction Count")
    plt.title("Mini-Compiler: IR Instruction Reduction per Program")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "trackF_compiler.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    results["compiler_plot"] = plot_path

    print("=== Track F Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_compiler_experiments()
    print(f"Track F Runtime: {time.time() - t0:.2f}s")
