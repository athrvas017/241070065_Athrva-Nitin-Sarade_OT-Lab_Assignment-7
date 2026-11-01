"""
Unit Tests for Track A: Continuous Optimization
"""

import numpy as np
import pytest
from trackA_continuous.run import (
    Optimizers,
    convex_quadratic,
    grad_convex_quadratic,
    TwoLayerMLP,
    hessian_vector_product_fd
)

def test_newton_quadratic_one_step():
    # Symmetric positive definite matrix
    A = np.array([[5.0, 1.0], [1.0, 3.0]])
    b = np.array([2.0, -1.0])
    x_opt = np.linalg.solve(A, b)
    x0 = np.array([10.0, -8.0])

    grad_fn = lambda x: grad_convex_quadratic(x, A, b)
    hess_fn = lambda x: A

    final_x, history = Optimizers.newton(grad_fn, hess_fn, x0, iters=2)
    # History[0] is x0, History[1] is after 1 step
    step1_x = history[1]
    assert np.allclose(step1_x, x_opt, atol=1e-10), "Newton's method must converge in 1 step on quadratic"

def test_manual_backprop_gradient_check():
    np.random.seed(241070065)
    mlp = TwoLayerMLP(in_dim=3, hidden_dim=5, out_dim=2, l2_reg=1e-3)
    X = np.random.randn(16, 3)
    y = np.random.randint(0, 2, size=16)

    rel_error = mlp.numerical_grad_check(X, y, eps=1e-6)
    assert rel_error < 1e-5, f"Manual backprop gradient check failed with rel error {rel_error}"

def test_hvp_accuracy():
    A = np.array([[4.0, 2.0], [2.0, 6.0]])
    b = np.array([1.0, 3.0])
    v = np.array([0.6, -0.8])
    x0 = np.array([1.0, 2.0])

    grad_fn = lambda x: grad_convex_quadratic(x, A, b)
    analytical_hvp = A @ v
    fd_hvp = hessian_vector_product_fd(grad_fn, x0, v, eps=1e-5)

    assert np.allclose(analytical_hvp, fd_hvp, atol=1e-4), "Finite difference HVP does not match analytical"
