"""
problems.py -- Problem bank P1-P14 (Lab Assignment 7, Section 4).
Every function: evaluate(x) -> (f_array, constraint_violation>=0).
g<=0 convention: 0 means feasible, positive = amount of violation.
All objectives returned in their MINIMIZATION form (maximizations already negated).
"""
import numpy as np

RNG_P8 = np.random.default_rng(7)   # fixed seed, as required by P8

# ---------------- P1: Factory production speed ----------------
P1_BOUNDS = [(1.0, 10.0)]
def P1_eval(x):
    x = x[0]
    f1 = (x - 3) ** 2
    f2 = (x - 7) ** 2
    return np.array([f1, f2]), 0.0

# ---------------- P2: Delivery fleet sizing ----------------
P2_BOUNDS = [(1.0, 10.0)]
def P2_eval(x, integer=False):
    x = x[0]
    if integer: x = round(x)
    x = max(1, min(10, x))
    f1 = 100.0 / x
    f2 = x ** 2
    return np.array([f1, f2]), 0.0

# ---------------- P3: ZDT2 (non-convex benchmark, n=10) ----------------
P3_N = 10
P3_BOUNDS = [(0.0, 1.0)] * P3_N
def P3_eval(x):
    x = np.asarray(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (P3_N - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2]), 0.0

def P3_true_front(n_pts=200):
    f1 = np.linspace(0, 1, n_pts)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])

# ---------------- P4: Investment portfolio ----------------
P4_mu = np.array([0.08, 0.12, 0.15, 0.10])
P4_Sigma = np.array([
    [0.010, 0.002, 0.001, 0.003],
    [0.002, 0.040, 0.006, 0.004],
    [0.001, 0.006, 0.090, 0.005],
    [0.003, 0.004, 0.005, 0.020],
])
P4_BOUNDS = [(0.0, 1.0)] * 4
def P4_eval(w, repair=True):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    if repair:
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(4) / 4
        viol = 0.0
    else:
        viol = abs(w.sum() - 1.0)
    risk = float(w @ P4_Sigma @ w)
    ret = float(P4_mu @ w)
    return np.array([risk, -ret]), viol   # minimize risk, minimize -return

# ---------------- P5: Cantilever beam design (constrained) ----------------
P5_L, P5_P, P5_E = 1.0, 1000.0, 200e9
P5_BOUNDS = [(0.01, 0.10), (0.02, 0.20)]
def P5_eval(x):
    b, h = x
    mass = 7850 * P5_L * b * h
    deflection = 4 * P5_P * P5_L ** 3 / (P5_E * b * h ** 3)
    stress = 6 * P5_P * P5_L / (b * h ** 2)
    viol = max(0.0, stress - 250e6)   # 0 if feasible
    return np.array([mass, deflection]), viol

# ---------------- P6: Binh-Korn constrained benchmark ----------------
P6_BOUNDS = [(0.0, 5.0), (0.0, 3.0)]
def P6_eval(x):
    x1, x2 = x
    f1 = 4 * x1 ** 2 + 4 * x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    g1 = (x1 - 5) ** 2 + x2 ** 2 - 25        # <= 0 required
    g2 = 7.7 - ((x1 - 8) ** 2 + (x2 + 3) ** 2)  # <= 0 required
    viol = max(0.0, g1) + max(0.0, g2)
    return np.array([f1, f2]), viol

# ---------------- P7: Water allocation ----------------
P7_BOUNDS = [(10.0, 90.0), (20.0, 90.0)]
def P7_eval(x):
    x1, x2 = x
    f1 = -0.8 * x1    # maximize 0.8x1 -> minimize -0.8x1
    f2 = -1.2 * x2
    viol = max(0.0, x1 + x2 - 90) + max(0.0, 10 - x1) + max(0.0, 20 - x2)
    return np.array([f1, f2]), viol

# ---------------- P8: Bi-objective delivery routing (TSP, 15 cities) ----------------
P8_N = 15
P8_COORDS = RNG_P8.uniform(0, 100, size=(P8_N, 2))
_u = RNG_P8.uniform(0.5, 2.0, size=(P8_N, P8_N))
P8_U = (_u + _u.T) / 2.0   # symmetric
P8_D = np.sqrt(((P8_COORDS[:, None, :] - P8_COORDS[None, :, :]) ** 2).sum(axis=2))
P8_T = P8_D * P8_U

def P8_eval(perm):
    perm = list(perm) + [perm[0]]
    dist = sum(P8_D[perm[i], perm[i+1]] for i in range(P8_N))
    time = sum(P8_T[perm[i], perm[i+1]] for i in range(P8_N))
    return np.array([dist, time]), 0.0

def P8_nearest_neighbour(cost_matrix, start=0):
    unvisited = set(range(P8_N)) - {start}
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: cost_matrix[cur, j])
        tour.append(nxt); unvisited.remove(nxt); cur = nxt
    return tour

# ---------------- P9: Feature selection (Wisconsin breast cancer) ----------------
try:
    from sklearn.datasets import load_breast_cancer
    _data = load_breast_cancer()
    P9_X, P9_y = _data.data, _data.target
except Exception:
    P9_X, P9_y = None, None
P9_D = 30

def _knn_cv_accuracy(mask, k=5, folds=5, seed=0):
    """Hand-coded 5-NN + 5-fold CV (no sklearn classifier/CV used, per the lab's rule)."""
    idx = np.where(mask > 0.5)[0]
    if len(idx) == 0:
        return 0.0
    X = P9_X[:, idx]
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = P9_y
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    fold_sizes = np.full(folds, len(y) // folds); fold_sizes[: len(y) % folds] += 1
    accs = []
    start = 0
    for fs in fold_sizes:
        test_idx = order[start:start + fs]; start += fs
        train_idx = np.setdiff1d(order, test_idx)
        Xtr, ytr, Xte, yte = Xn[train_idx], y[train_idx], Xn[test_idx], y[test_idx]
        d = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2))
        knn_idx = np.argsort(d, axis=1)[:, :k]
        preds = np.array([np.bincount(ytr[row]).argmax() for row in knn_idx])
        accs.append((preds == yte).mean())
    return float(np.mean(accs))

def P9_eval(mask):
    mask = (np.asarray(mask) > 0.5).astype(float)
    if mask.sum() == 0:
        return np.array([1.0, P9_D]), 1.0   # infeasible: no feature selected
    acc = _knn_cv_accuracy(mask)
    return np.array([1 - acc, mask.sum()]), 0.0

# ---------------- P10: DTLZ2 many-objective benchmark ----------------
def P10_eval(x, M=3):
    x = np.asarray(x)
    n = M + 9
    assert len(x) == n
    xM = x[M - 1:]
    g = np.sum((xM - 0.5) ** 2)
    f = np.zeros(M)
    for m in range(M):
        val = (1 + g)
        for i in range(M - m - 1):
            val *= np.cos(x[i] * np.pi / 2)
        if m > 0:
            val *= np.sin(x[M - m - 1] * np.pi / 2)
        f[m] = val
    return f, 0.0

def P10_true_front(M, n_pts=300, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_pts):
        v = rng.normal(size=M)
        v = np.abs(v) / np.linalg.norm(v)
        pts.append(v)
    return np.array(pts)

P10_BOUNDS = lambda M: [(0.0, 1.0)] * (M + 9)

# ---------------- P11: Production planning (Goal Programming / Lexicographic) ----------------
P11_BOUNDS = [(0.0, 400.0), (0.0, 400.0)]
def P11_eval(x):
    xA, xB = x
    profit = 4 * xA + 6 * xB
    units = xA + xB
    viol = max(0.0, 2*xA + 3*xB - 1200) + max(0.0, 4*xA + 2*xB - 1600) + max(0.0, -xA) + max(0.0, -xB)
    return np.array([-profit, -units]), viol   # maximize both -> minimize negatives

# ---------------- P12: Supplier selection (MCDM) ----------------
P12_DATA = np.array([
    [120, 8, 10, 92],
    [100, 7, 14, 88],
    [140, 9,  7, 95],
    [ 90, 6, 18, 80],
    [130, 8,  9, 90],
], dtype=float)
P12_NAMES = ["S1", "S2", "S3", "S4", "S5"]
P12_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.2])
P12_SENSE = np.array([-1, 1, -1, 1])  # -1 = minimize (cost, delivery days), 1 = maximize

# ---------------- P13: Multi-task learning (toy) ----------------
P13_a = np.array([3.0, 3.0]); P13_b = np.array([-3.0, -3.0])
def P13_eval(x):
    x = np.asarray(x, dtype=float)
    f1 = float(np.sum((x - P13_a) ** 2))
    f2 = float(np.sum((x - P13_b) ** 2))
    return np.array([f1, f2]), 0.0
def P13_grad(x):
    x = np.asarray(x, dtype=float)
    return 2 * (x - P13_a), 2 * (x - P13_b)

# ---------------- P14: Laptop design with decision-maker preference ----------------
P14_BOUNDS = [(13.0, 17.0), (40.0, 90.0)]
def P14_eval(x):
    x1, x2 = x
    weight = 0.4 + 0.05 * x1 + 0.012 * x2
    runtime = 0.2 * x2 / (0.5 + 0.03 * x1)
    return np.array([weight, -runtime]), 0.0
