# Track A: Continuous Optimization

## Objectives
- Compare first-order and second-order continuous optimization methods across convex, ill-conditioned, and non-convex energy landscapes.
- Implement from scratch in NumPy: Gradient Descent (GD), Stochastic Gradient Descent (SGD) with noise, Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, Adam, Newton's Method, and Gauss-Newton for nonlinear least-squares.
- Formulate machine learning as an unconstrained empirical risk minimization problem with manual analytical backpropagation and L2 regularization.
- Analyze curvature approximations including finite-difference Hessian-Vector Products (HVP), empirical Fisher diagonal, and low-rank SVD.

## Key Formulations
1. **Convex Quadratic:** $f(x) = \frac{1}{2} x^T A x - b^T x$, $\nabla f(x) = A x - b$, $\nabla^2 f(x) = A$.
2. **Rosenbrock Function:** $f(x, y) = (a - x)^2 + b(y - x^2)^2$.
3. **Newton's Step:** $x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$.
4. **Hessian-Vector Product:** $H v \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$.

## How to Run
```bash
python trackA_continuous/run.py
pytest trackA_continuous/tests/
```
