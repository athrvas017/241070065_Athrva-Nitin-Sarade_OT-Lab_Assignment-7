"""
Unit Tests for Track C: Quantum Optimization
"""

import numpy as np
import pytest
import pennylane as qml
from pennylane import numpy as pnp
from trackC_quantum_opt.run import (
    build_qaoa_circuit,
    run_vqe_demo
)

def test_qaoa_expectation_bounds():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
    circ, _ = build_qaoa_circuit(n_nodes=3, edges=edges, p=1)
    gamma = pnp.array([0.5], requires_grad=True)
    beta = pnp.array([0.5], requires_grad=True)

    energy = float(circ(gamma, beta))
    # Cut = 0.5 * 3 - energy. Cut must be between 0 and 3
    cut = 1.5 - energy
    assert 0.0 <= cut <= 3.0, f"Expected cut {cut} outside theoretical bounds [0, 3]"

def test_parameter_shift_rule():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circ(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta_val = 0.75
    # Analytical: d(cos(theta))/d(theta) = -sin(theta)
    analytical_grad = -np.sin(theta_val)
    ps_grad = float(qml.grad(circ)(pnp.array(theta_val, requires_grad=True)))

    assert np.isclose(analytical_grad, ps_grad, atol=1e-5), f"Parameter shift grad {ps_grad} != analytical {analytical_grad}"

def test_vqe_ground_state_convergence():
    res = run_vqe_demo()
    exact = res["exact_ground_energy"]
    adam_val = res["adam_final"]
    qng_val = res["qng_final"]
    # VQE expectation is an upper bound on ground state by Rayleigh-Ritz
    assert adam_val >= exact - 1e-4, "VQE energy cannot violate variational minimum"
    assert qng_val >= exact - 1e-4, "QNG energy cannot violate variational minimum"
    # QNG achieves faster geometric convergence on quantum manifold
    best_energy = min(adam_val, qng_val)
    assert abs(best_energy - exact) < 0.25, f"VQE failed to converge near ground energy: best {best_energy}, exact {exact}"
