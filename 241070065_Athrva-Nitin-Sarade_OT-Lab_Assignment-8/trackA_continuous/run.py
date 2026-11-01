"""
Track A: Continuous Optimization
Student: Athrva Nitin Sarade | Roll No: 241070065
Seed: 241070065
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable

# Set global random seed
SEED = 241070065
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Objective Functions and Landscapes
# -------------------------------------------------------------

def convex_quadratic(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * x.T @ A @ x - b.T @ x)

def grad_convex_quadratic(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return A @ x - b

def hessian_convex_quadratic(A: np.ndarray) -> np.ndarray:
    return A

def rosenbrock(xy: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    x, y = xy[0], xy[1]
    return float((a - x) ** 2 + b * (y - x ** 2) ** 2)

def grad_rosenbrock(xy: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    x, y = xy[0], xy[1]
    df_dx = -2.0 * (a - x) - 4.0 * b * x * (y - x ** 2)
    df_dy = 2.0 * b * (y - x ** 2)
    return np.array([df_dx, df_dy])

def saddle_function(xy: np.ndarray) -> float:
    return float(xy[0] ** 2 - xy[1] ** 2)

def double_well(xy: np.ndarray) -> float:
    return float((xy[0] ** 2 - 1.0) ** 2 + xy[1] ** 2)

def grad_double_well(xy: np.ndarray) -> np.ndarray:
    x, y = xy[0], xy[1]
    return np.array([4.0 * x * (x ** 2 - 1.0), 2.0 * y])

# -------------------------------------------------------------
# 2. First-Order and Second-Order Optimizers from Scratch
# -------------------------------------------------------------

class Optimizers:
    @staticmethod
    def gd(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, iters: int = 100) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        history = [x.copy()]
        for _ in range(iters):
            g = grad_fn(x)
            x -= lr * g
            history.append(x.copy())
        return x, history

    @staticmethod
    def sgd_noisy(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, iters: int = 100, noise_std: float = 0.1) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        history = [x.copy()]
        for _ in range(iters):
            noise = np.random.normal(0, noise_std, size=x.shape)
            g = grad_fn(x) + noise
            x -= lr * g
            history.append(x.copy())
        return x, history

    @staticmethod
    def momentum(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, beta: float = 0.9, iters: int = 100) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        v = np.zeros_like(x)
        history = [x.copy()]
        for _ in range(iters):
            g = grad_fn(x)
            v = beta * v + lr * g
            x -= v
            history.append(x.copy())
        return x, history

    @staticmethod
    def nag(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, beta: float = 0.9, iters: int = 100) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        v = np.zeros_like(x)
        history = [x.copy()]
        for _ in range(iters):
            x_lookahead = x - beta * v
            g = grad_fn(x_lookahead)
            v = beta * v + lr * g
            x -= v
            history.append(x.copy())
        return x, history

    @staticmethod
    def adagrad(grad_fn: Callable, x0: np.ndarray, lr: float = 0.1, iters: int = 100, eps: float = 1e-8) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        G = np.zeros_like(x)
        history = [x.copy()]
        for _ in range(iters):
            g = grad_fn(x)
            G += g ** 2
            x -= (lr / (np.sqrt(G) + eps)) * g
            history.append(x.copy())
        return x, history

    @staticmethod
    def rmsprop(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, beta: float = 0.9, iters: int = 100, eps: float = 1e-8) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        s = np.zeros_like(x)
        history = [x.copy()]
        for _ in range(iters):
            g = grad_fn(x)
            s = beta * s + (1.0 - beta) * (g ** 2)
            x -= (lr / (np.sqrt(s) + eps)) * g
            history.append(x.copy())
        return x, history

    @staticmethod
    def adam(grad_fn: Callable, x0: np.ndarray, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, iters: int = 100, eps: float = 1e-8) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = [x.copy()]
        for t in range(1, iters + 1):
            g = grad_fn(x)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            x -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
            history.append(x.copy())
        return x, history

    @staticmethod
    def newton(grad_fn: Callable, hess_fn: Callable, x0: np.ndarray, iters: int = 10) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        history = [x.copy()]
        for _ in range(iters):
            g = grad_fn(x)
            if np.linalg.norm(g) < 1e-10:
                break
            H = hess_fn(x)
            # Regularize slightly if singular
            H_reg = H + 1e-9 * np.eye(len(x))
            delta = np.linalg.solve(H_reg, g)
            x -= delta
            history.append(x.copy())
        return x, history

    @staticmethod
    def gauss_newton(residual_fn: Callable, jacobian_fn: Callable, x0: np.ndarray, iters: int = 10) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy().astype(float)
        history = [x.copy()]
        for _ in range(iters):
            r = residual_fn(x)
            J = jacobian_fn(x)
            # Gauss-Newton step: delta = (J^T J)^{-1} J^T r
            JT_J = J.T @ J + 1e-8 * np.eye(len(x))
            delta = np.linalg.solve(JT_J, J.T @ r)
            x -= delta
            history.append(x.copy())
        return x, history

# -------------------------------------------------------------
# 3. Curvature: HVP, Fisher Diagonal, Low-Rank Demo
# -------------------------------------------------------------

def hessian_vector_product_fd(grad_fn: Callable, x: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Symmetric finite difference approximation of Hessian-Vector Product."""
    g_plus = grad_fn(x + eps * v)
    g_minus = grad_fn(x - eps * v)
    return (g_plus - g_minus) / (2.0 * eps)

def diagonal_fisher_approx(grads: np.ndarray) -> np.ndarray:
    """Empirical Fisher information diagonal: E[g_i^2]."""
    return np.mean(grads ** 2, axis=0)

def low_rank_curvature_svd(H: np.ndarray, rank: int = 2) -> np.ndarray:
    """Low-rank SVD approximation of curvature."""
    U, S, Vt = np.linalg.svd(H)
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

# -------------------------------------------------------------
# 4. Small ML Model with Manual Backpropagation
# -------------------------------------------------------------

class TwoLayerMLP:
    """2-layer MLP (Input -> Hidden (Tanh) -> Output (Softmax)) with manual backprop."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, l2_reg: float = 1e-4):
        self.W1 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(out_dim)
        self.l2_reg = l2_reg

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # X: (N, in_dim)
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        # Numerically stable softmax
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return z1, a1, probs

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        _, _, probs = self.forward(X)
        log_likelihood = -np.log(probs[np.arange(N), y] + 1e-12)
        data_loss = np.mean(log_likelihood)
        reg_loss = 0.5 * self.l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(data_loss + reg_loss)

    def backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        N = X.shape[0]
        z1, a1, probs = self.forward(X)
        
        # dL/dz2 for softmax cross-entropy
        dprobs = probs.copy()
        dprobs[np.arange(N), y] -= 1.0
        dprobs /= N

        dW2 = a1.T @ dprobs + self.l2_reg * self.W2
        db2 = np.sum(dprobs, axis=0)

        da1 = dprobs @ self.W2.T
        dz1 = da1 * (1.0 - a1 ** 2)  # d(tanh)/dz1 = 1 - tanh^2
        dW1 = X.T @ dz1 + self.l2_reg * self.W1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def numerical_grad_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Computes relative error between analytical and numerical gradients."""
        grads = self.backward(X, y)
        max_rel_error = 0.0

        for param_name, param in [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]:
            ana_grad = grads[param_name]
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                param[idx] = orig_val + eps
                loss_plus = self.loss(X, y)

                param[idx] = orig_val - eps
                loss_minus = self.loss(X, y)

                param[idx] = orig_val
                num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                it.iternext()

            rel_err = np.linalg.norm(ana_grad - num_grad) / (np.linalg.norm(ana_grad) + np.linalg.norm(num_grad) + 1e-12)
            max_rel_error = max(max_rel_error, float(rel_err))
        return max_rel_error

# -------------------------------------------------------------
# 5. Experiment Execution & Visualization
# -------------------------------------------------------------

def run_experiments() -> Dict[str, any]:
    results = {}
    print("=== Running Track A: Continuous Optimization ===")

    # 1. Landscapes Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1a. Convex Quadratic
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_quad = 0.5 * (A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) - (b[0]*X + b[1]*Y)
    axes[0].contour(X, Y, Z_quad, levels=20, cmap='viridis')
    axes[0].plot([np.linalg.solve(A, b)[0]], [np.linalg.solve(A, b)[1]], 'r*', markersize=12, label='Global Min')
    axes[0].set_title("Convex Quadratic (Unique Global Min)")
    axes[0].legend()

    # 1b. Rosenbrock (Non-convex)
    Z_rosen = (1.0 - X)**2 + 100.0 * (Y - X**2)**2
    axes[1].contour(X, Y, np.log1p(Z_rosen), levels=25, cmap='plasma')
    axes[1].plot([1.0], [1.0], 'r*', markersize=12, label='Global Min (1,1)')
    axes[1].set_title("Rosenbrock (Banana Valley)")
    axes[1].legend()

    # 1c. Saddle Point & Double Well
    Z_saddle = X**2 - Y**2
    axes[2].contour(X, Y, Z_saddle, levels=20, cmap='coolwarm')
    axes[2].plot([0], [0], 'ko', markersize=10, label='Saddle Point (0,0)')
    axes[2].set_title("Saddle Surface: x^2 - y^2")
    axes[2].legend()

    plt.tight_layout()
    landscape_path = os.path.join(RESULTS_DIR, "trackA_landscapes.png")
    plt.savefig(landscape_path, dpi=120)
    plt.close()
    results["landscape_plot"] = landscape_path

    # 2. Convergence Comparison on Ill-conditioned Quadratic
    A_ill = np.array([[20.0, 0.0], [0.0, 1.0]])
    b_ill = np.array([0.0, 0.0])
    x0 = np.array([2.0, 2.0])
    iters = 60

    grad_fn_ill = lambda x: grad_convex_quadratic(x, A_ill, b_ill)
    obj_fn_ill = lambda x: convex_quadratic(x, A_ill, b_ill)

    opt_runs = {
        "GD (lr=0.04)": Optimizers.gd(grad_fn_ill, x0, lr=0.04, iters=iters),
        "Momentum (lr=0.04, beta=0.9)": Optimizers.momentum(grad_fn_ill, x0, lr=0.04, beta=0.9, iters=iters),
        "NAG (lr=0.04, beta=0.9)": Optimizers.nag(grad_fn_ill, x0, lr=0.04, beta=0.9, iters=iters),
        "AdaGrad (lr=0.5)": Optimizers.adagrad(grad_fn_ill, x0, lr=0.5, iters=iters),
        "RMSProp (lr=0.1)": Optimizers.rmsprop(grad_fn_ill, x0, lr=0.1, iters=iters),
        "Adam (lr=0.1)": Optimizers.adam(grad_fn_ill, x0, lr=0.1, iters=iters),
        "Newton": Optimizers.newton(grad_fn_ill, lambda x: A_ill, x0, iters=5)
    }

    plt.figure(figsize=(9, 5.5))
    perf_table = []
    for name, (final_x, hist) in opt_runs.items():
        losses = [obj_fn_ill(pt) for pt in hist]
        plt.semilogy(losses, label=name, lw=1.8)
        perf_table.append({
            "Method": name,
            "Final Loss": float(losses[-1]),
            "Steps": len(hist) - 1,
            "Final X": np.round(final_x, 4).tolist()
        })

    plt.title("Convergence Comparison on Ill-Conditioned Quadratic")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value (log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    conv_path = os.path.join(RESULTS_DIR, "trackA_convergence.png")
    plt.savefig(conv_path, dpi=120)
    plt.close()
    results["convergence_plot"] = conv_path
    results["optimizer_table"] = perf_table

    # 3. Step-size sensitivity and noisy gradients
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    for lr in [0.005, 0.04, 0.09]:
        _, hist = Optimizers.gd(grad_fn_ill, x0, lr=lr, iters=50)
        plt.semilogy([obj_fn_ill(p) for p in hist], label=f"GD lr={lr}")
    plt.title("Step-Size Sensitivity (GD)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    _, hist_clean = Optimizers.gd(grad_fn_ill, x0, lr=0.04, iters=50)
    _, hist_noisy = Optimizers.sgd_noisy(grad_fn_ill, x0, lr=0.04, iters=50, noise_std=0.2)
    _, hist_adam_noisy = Optimizers.adam(lambda x: grad_fn_ill(x) + np.random.normal(0, 0.2, size=x.shape), x0, lr=0.1, iters=50)
    plt.semilogy([obj_fn_ill(p) for p in hist_clean], label="GD (clean)", lw=2)
    plt.semilogy([obj_fn_ill(p) for p in hist_noisy], label="SGD (noise std=0.2)", alpha=0.8)
    plt.semilogy([obj_fn_ill(p) for p in hist_adam_noisy], label="Adam (noise std=0.2)", lw=2)
    plt.title("Robustness to Gradient Noise")
    plt.xlabel("Iteration")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    noise_path = os.path.join(RESULTS_DIR, "trackA_step_and_noise.png")
    plt.savefig(noise_path, dpi=120)
    plt.close()
    results["step_and_noise_plot"] = noise_path

    # 4. Neural Network Manual Backpropagation Check
    N, D, H, C = 64, 4, 8, 2
    X_syn = np.random.randn(N, D)
    y_syn = np.random.randint(0, C, size=N)
    mlp = TwoLayerMLP(in_dim=D, hidden_dim=H, out_dim=C, l2_reg=1e-3)
    grad_err = mlp.numerical_grad_check(X_syn, y_syn)
    print(f"Manual Backprop Numerical Gradient Check Max Relative Error: {grad_err:.2e}")
    results["backprop_grad_check_rel_error"] = grad_err

    # 5. Curvature Checks: HVP and Gauss-Newton
    v = np.array([1.0, -1.0])
    hvp_analytical = A_ill @ v
    hvp_fd = hessian_vector_product_fd(grad_fn_ill, x0, v, eps=1e-5)
    hvp_error = float(np.linalg.norm(hvp_analytical - hvp_fd) / np.linalg.norm(hvp_analytical))
    print(f"Hessian-Vector Product Relative Error: {hvp_error:.2e}")
    results["hvp_relative_error"] = hvp_error

    # Gauss-Newton on nonlinear model y = exp(a * t) + b
    t_data = np.linspace(0, 1, 10)
    y_true = np.exp(1.5 * t_data) + 0.5
    res_fn = lambda theta: (np.exp(theta[0] * t_data) + theta[1]) - y_true
    jac_fn = lambda theta: np.column_stack([t_data * np.exp(theta[0] * t_data), np.ones_like(t_data)])
    theta_gn, hist_gn = Optimizers.gauss_newton(res_fn, jac_fn, np.array([0.5, 0.0]), iters=8)
    results["gauss_newton_solution"] = np.round(theta_gn, 4).tolist()
    results["gauss_newton_res_norm"] = float(np.linalg.norm(res_fn(theta_gn)))

    print("=== Track A Complete! ===")
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_experiments()
    print(f"Track A Wall-clock Runtime: {time.time() - t0:.2f}s")
