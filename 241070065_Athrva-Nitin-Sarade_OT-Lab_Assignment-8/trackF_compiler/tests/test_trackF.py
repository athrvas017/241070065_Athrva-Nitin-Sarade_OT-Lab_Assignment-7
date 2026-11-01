"""
Unit Tests for Track F: Compiler Optimization
"""

import pytest
from trackF_compiler.run import (
    TACInstruction,
    SimpleVM,
    CompilerOptimizer,
    allocate_registers,
    get_benchmark_programs
)

def test_semantic_equivalence_on_all_benchmarks():
    progs = get_benchmark_programs()
    vm = SimpleVM()
    pipeline = ['CP', 'CF', 'CSE', 'DCE']

    for name, raw_ir in progs.items():
        out_raw = vm.run(raw_ir)
        opt_ir = CompilerOptimizer.optimize_pipeline(raw_ir, pipeline)
        out_opt = vm.run(opt_ir)

        assert out_raw == out_opt, f"Semantic mismatch in {name}: unoptimized {out_raw} != optimized {out_opt}"

def test_constant_folding_and_propagation():
    ir = [
        TACInstruction('ASSIGN', 12, None, 'x'),
        TACInstruction('ASSIGN', 8, None, 'y'),
        TACInstruction('ADD', 'x', 'y', 'z'),
        TACInstruction('PRINT', 'z', None, None)
    ]
    # CP then CF
    step1 = CompilerOptimizer.constant_propagation(ir)
    step2 = CompilerOptimizer.constant_folding(step1)

    # After CF, 'z' should be directly assigned 20
    z_inst = [inst for inst in step2 if inst.dest == 'z'][0]
    assert z_inst.op == 'ASSIGN' and z_inst.arg1 == 20

def test_register_allocation_no_conflicts():
    ir = [
        TACInstruction('ASSIGN', 1, None, 'a'),
        TACInstruction('ASSIGN', 2, None, 'b'),
        TACInstruction('ADD', 'a', 'b', 'c'),
        TACInstruction('PRINT', 'c', None, None)
    ]
    reg_map = allocate_registers(ir, k_registers=4)
    # Interfering variables 'a' and 'b' must not share the same register
    assert reg_map['a'] != reg_map['b'], f"Variables a and b assigned same register: {reg_map['a']}"
