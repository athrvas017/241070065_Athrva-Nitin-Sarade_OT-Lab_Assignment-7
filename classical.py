"""
Algorithms 1-9: Classical scalarization + ranking methods (Lab Catalogue Part 1).
Each returns a front (set of (x, f) points) by sweeping a parameter, OR a single
solution plus the preference/priority input that produced it (algorithms 5-9).
"""
import numpy as np
from scipy.optimize import minimize


# ===================== 1. Weighted Sum =====================
def weighted_sum(eval_fn, bounds, n_weights=11, x0=None, seed=0):
    """Scalarize f(x) as w1*f1+w2*f2+...; vary weights on the simplex; solve each
    single-objective sub-problem with scipy.optimize (permitted exception for
    algorithms 1-7 per Section 2A -- the weight sweep itself is our own code)."""
    rng = np.random.default_rng(seed)
    n = len(bounds)
    ws = np.linspace(0, 1, n_weights)
    results = []
    for w1 in ws:
        w = np.array([w1, 1 - w1])
        def scalar_obj(x, w=w):
            f, viol = eval_fn(x)
            return float(w[:len(f)] @ f) + 1e6 * viol
        x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
        res = minimize(scalar_obj, x_start, bounds=bounds, method='L-BFGS-B')
        f, viol = eval_fn(res.x)
        if viol < 1e-6:
            results.append((res.x.copy(), f.copy(), w.copy()))
    return results


# ===================== 2. Epsilon-Constraint =====================
def epsilon_constraint(eval_fn, bounds, eps_range, n_eps=11, x0=None):
    """Minimize f1 subject to f2 <= eps, sweeping eps over eps_range."""
    results = []
    epss = np.linspace(eps_range[0], eps_range[1], n_eps)
    for eps in epss:
        def obj(x):
            f, viol = eval_fn(x)
            return f[0]
        def cons_f(x):
            f, viol = eval_fn(x)
            return eps - f[1]   # >=0 required by scipy 'ineq'
        x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
        res = minimize(obj, x_start, bounds=bounds, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cons_f}])
        if res.success:
            f, viol = eval_fn(res.x)
            if viol < 1e-6:
                results.append((res.x.copy(), f.copy(), eps))
    return results


# ===================== 3. Benson's Method =====================
def bensons_method(eval_fn, bounds, ref_point, n_dirs=11, x0=None):
    """Maximize the weighted distance travelled past a reference point along a
    search direction. From ref_point, maximize sum(w_i*(ref_i - f_i)) over weights.
    Step 1: pick feasible start x0 / reference point.
    Step 2: for each weight vector, maximize distance (minimize negative distance).
    Step 3: result is a Pareto-optimal or dominated point.
    Step 4: sweep weight vectors to build the front."""
    results = []
    ws = np.linspace(0.05, 0.95, n_dirs)
    for w1 in ws:
        w = np.array([w1, 1 - w1])
        def obj(x):
            f, viol = eval_fn(x)
            dist = w @ (ref_point - f[:len(w)])
            return -dist + 1e6 * viol   # maximize distance -> minimize -distance
        x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
        res = minimize(obj, x_start, bounds=bounds, method='L-BFGS-B')
        f, viol = eval_fn(res.x)
        if viol < 1e-6:
            results.append((res.x.copy(), f.copy(), w.copy()))
    return results


# ===================== 4. Weight Metric (p-norm / Tchebycheff) =====================
def weight_metric(eval_fn, bounds, ideal, p=2, n_weights=11, x0=None):
    """Minimize the weighted L_p distance to the ideal point. p=inf = Tchebycheff."""
    results = []
    ws = np.linspace(0.02, 0.98, n_weights)
    for w1 in ws:
        w = np.array([w1, 1 - w1])
        def obj(x):
            f, viol = eval_fn(x)
            d = w * np.abs(f[:len(w)] - ideal)
            val = np.max(d) if p == np.inf else (np.sum(d ** p)) ** (1 / p)
            return val + 1e6 * viol
        x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
        res = minimize(obj, x_start, bounds=bounds, method='Nelder-Mead')
        f, viol = eval_fn(res.x)
        if viol < 1e-6:
            results.append((res.x.copy(), f.copy(), w.copy()))
    return results


# ===================== 5. Value Function =====================
def value_function(eval_fn, bounds, utility_fn, x0=None):
    """Directly optimize a decision-maker's utility function U(f1,f2) (assumed to
    be known/elicited). Returns the single utility-maximizing solution."""
    def obj(x):
        f, viol = eval_fn(x)
        return -utility_fn(f) + 1e6 * viol
    x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
    res = minimize(obj, x_start, bounds=bounds, method='Nelder-Mead')
    f, viol = eval_fn(res.x)
    return res.x, f, utility_fn


# ===================== 6. Goal Programming =====================
def goal_programming(eval_fn, bounds, targets, weights=None, x0=None):
    """Minimize weighted sum of positive deviations d+ (over target) and negative
    deviations d- (under target) from stated goals. targets: array matching f."""
    weights = np.ones(len(targets)) if weights is None else np.asarray(weights)
    def obj(x):
        f, viol = eval_fn(x)
        d_plus = np.maximum(0, f - targets)
        d_minus = np.maximum(0, targets - f)
        return float(weights @ (d_plus + d_minus)) + 1e6 * viol
    x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
    res = minimize(obj, x_start, bounds=bounds, method='Nelder-Mead')
    f, viol = eval_fn(res.x)
    d_plus = np.maximum(0, f - targets); d_minus = np.maximum(0, targets - f)
    return res.x, f, dict(d_plus=d_plus, d_minus=d_minus, shortfall=float(weights@(d_plus+d_minus)))


# ===================== 7. Lexicographic =====================
def lexicographic(eval_fn, bounds, priority_order, tol=1e-3, x0=None):
    """Optimize objectives in strict priority order: objective priority_order[0]
    first; then objective priority_order[1] subject to staying within `tol` of the
    first objective's optimum; etc."""
    x_start = x0 if x0 is not None else [ (b[0]+b[1])/2 for b in bounds ]
    achieved = []
    constraints = []
    cur_x = x_start
    for level, obj_idx in enumerate(priority_order):
        def obj(x, obj_idx=obj_idx):
            f, viol = eval_fn(x)
            return f[obj_idx] + 1e6 * viol
        res = minimize(obj, cur_x, bounds=bounds, method='SLSQP', constraints=constraints)
        f_opt, viol = eval_fn(res.x)
        best_val = f_opt[obj_idx]
        achieved.append(best_val)
        cur_x = res.x
        def make_cons(obj_idx=obj_idx, best_val=best_val):
            def c(x):
                f, viol = eval_fn(x)
                return (best_val + tol) - f[obj_idx]   # f[obj_idx] <= best_val+tol
            return {'type': 'ineq', 'fun': c}
        constraints = constraints + [make_cons()]
    f_final, viol = eval_fn(cur_x)
    return cur_x, f_final, achieved


# ===================== 8. TOPSIS =====================
def topsis(decision_matrix, weights, sense):
    """decision_matrix: (n_alt, n_crit). sense: +1 maximize, -1 minimize per column.
    Returns ranking (best first) and closeness scores."""
    D = np.asarray(decision_matrix, dtype=float)
    norm = D / np.sqrt((D ** 2).sum(axis=0))
    weighted = norm * weights
    ideal_best = np.where(sense > 0, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(sense > 0, weighted.min(axis=0), weighted.max(axis=0))
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = dist_worst / (dist_best + dist_worst + 1e-12)
    rank = np.argsort(-closeness)
    return rank, closeness


# ===================== 9. VIKOR =====================
def vikor(decision_matrix, weights, sense, v=0.5):
    """Compromise ranking via S (group utility), R (individual regret), Q (compromise)."""
    D = np.asarray(decision_matrix, dtype=float)
    best = np.where(sense > 0, D.max(axis=0), D.min(axis=0))
    worst = np.where(sense > 0, D.min(axis=0), D.max(axis=0))
    rng = np.where(np.abs(best - worst) < 1e-12, 1.0, best - worst)
    norm_gap = weights * np.abs(best - D) / np.abs(rng)
    S = norm_gap.sum(axis=1)
    R = norm_gap.max(axis=1)
    Sstar, Sminus = S.min(), S.max()
    Rstar, Rminus = R.min(), R.max()
    Q = v * (S - Sstar) / (Sminus - Sstar + 1e-12) + (1 - v) * (R - Rstar) / (Rminus - Rstar + 1e-12)
    rank = np.argsort(Q)
    return rank, Q, S, R
