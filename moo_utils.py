"""
moo_utils -- shared multi-objective optimization toolkit (Lab Assignment 7).
Dominance test, fast non-dominated sort, crowding distance, hypervolume (2D exact
+ Monte-Carlo for 3+ objectives), IGD, Spacing, and objective normalization.
All minimization; convert any maximization with f <- -f before using these tools.
"""
import numpy as np


# ---------------------------------------------------------------------------
def dominates(a, b):
    """a, b: 1-D arrays of objective values (minimization). True if a dominates b:
    a is <= b in every objective and strictly < in at least one."""
    a = np.asarray(a); b = np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def constrained_dominates(a, cva, b, cvb):
    """Deb's 3 constrained-dominance rules (minimization, constraint violation >= 0):
    Step 1: feasible beats infeasible.
    Step 2: both infeasible -> smaller total violation wins.
    Step 3: both feasible  -> Pareto dominance applies."""
    # Step 1 & 2
    if cva < 1e-9 and cvb >= 1e-9:
        return True   # a feasible, b not
    if cva >= 1e-9 and cvb < 1e-9:
        return False  # b feasible, a not
    if cva >= 1e-9 and cvb >= 1e-9:
        return cva < cvb  # both infeasible: smaller violation wins
    # Step 3: both feasible
    return dominates(a, b)


def fast_nondominated_sort(F):
    """F: (N, M) objective matrix. Returns a list of fronts (each a list of indices),
    front[0] is the Pareto-optimal set within F. O(MN^2) -- Deb et al. (2002)."""
    N = F.shape[0]
    S = [[] for _ in range(N)]          # solutions dominated by i
    n = np.zeros(N, dtype=int)          # number of solutions dominating i
    rank = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # last appended is always empty
    return fronts, rank


def crowding_distance(F, front_idx):
    """F: (N,M) objective matrix. front_idx: indices of one front.
    Returns a dict {index: crowding distance}; boundary points get np.inf."""
    idx = list(front_idx)
    M = F.shape[1]
    dist = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        for i in idx: dist[i] = np.inf
        return dist
    for m in range(M):
        idx_sorted = sorted(idx, key=lambda i: F[i, m])
        fmin, fmax = F[idx_sorted[0], m], F[idx_sorted[-1], m]
        dist[idx_sorted[0]] = np.inf
        dist[idx_sorted[-1]] = np.inf
        if fmax - fmin < 1e-12:
            continue
        for k in range(1, len(idx_sorted) - 1):
            dist[idx_sorted[k]] += (F[idx_sorted[k+1], m] - F[idx_sorted[k-1], m]) / (fmax - fmin)
    return dist


def normalize(F, fmin=None, fmax=None):
    """Min-max normalize each column of F to [0,1]. Returns (F_norm, fmin, fmax)."""
    F = np.asarray(F, dtype=float)
    if fmin is None: fmin = F.min(axis=0)
    if fmax is None: fmax = F.max(axis=0)
    rng = np.where(fmax - fmin < 1e-12, 1.0, fmax - fmin)
    return (F - fmin) / rng, fmin, fmax


def hypervolume_2d(F, ref):
    """Exact hypervolume for a 2-objective minimization front. F: (N,2), ref: (2,)
    reference point dominated by every point. Only points strictly below ref count.
    Points are sorted ascending by x; since a non-dominated front has y strictly
    decreasing as x increases, the dominated region decomposes into non-overlapping
    rectangles, one per point, each spanning from that point's x to the NEXT point's
    x (or to ref's x for the last point), and from that point's y up to ref's y."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    fronts, _ = fast_nondominated_sort(F)       # HV is only meaningful on the non-dominated subset
    F = F[fronts[0]]
    pts = [tuple(p) for p in F if np.all(p < ref)]
    if not pts:
        return 0.0
    pts = sorted(set(pts), key=lambda p: p[0])
    hv = 0.0
    for i, (x, y) in enumerate(pts):
        next_x = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        width = next_x - x
        height = ref[1] - y
        hv += width * height
    return hv


def hypervolume_mc(F, ref, n_samples=200000, seed=0):
    """Monte-Carlo hypervolume estimate for M>=2 objectives (used for M=3,5 in P10).
    F: (N,M) minimization front, ref: (M,) reference point dominated by every point."""
    rng = np.random.default_rng(seed)
    pts = np.array([p for p in F if np.all(p < ref)])
    if len(pts) == 0:
        return 0.0
    fmin = pts.min(axis=0)
    box_vol = np.prod(ref - fmin)
    samples = rng.uniform(fmin, ref, size=(n_samples, F.shape[1]))
    # a sample is "dominated" by the front if some front point is <= it in every coord
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)
    return box_vol * dominated.mean()


def hypervolume(F, ref):
    F = np.atleast_2d(F)
    return hypervolume_2d(F, ref) if F.shape[1] == 2 else hypervolume_mc(F, ref)


def igd(F_obtained, F_true):
    """Inverted Generational Distance: mean distance from each TRUE-front point to
    its nearest OBTAINED point. Smaller is better (measures convergence + spread)."""
    F_obtained = np.atleast_2d(F_obtained); F_true = np.atleast_2d(F_true)
    d = np.sqrt(((F_true[:, None, :] - F_obtained[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def spacing(F):
    """Spacing metric: std. dev. of nearest-neighbour distances along the obtained
    front. Smaller means more evenly spread. F: (N,M)."""
    F = np.atleast_2d(F)
    N = F.shape[0]
    if N < 2:
        return 0.0
    d = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(nn.std())


def extract_pareto(F, X=None):
    """Given an evaluated population, returns only the non-dominated (front-0) rows."""
    fronts, _ = fast_nondominated_sort(F)
    idx = fronts[0]
    return (F[idx], X[idx] if X is not None else None, idx)


def das_dennis(M, p):
    """Generate structured reference points on the unit simplex (Das & Dennis).
    Returns an array of shape (C(M+p-1, p), M)."""
    def rec(M, p):
        if M == 1: return [[p]]
        pts = []
        for i in range(p + 1):
            for rest in rec(M - 1, p - i):
                pts.append([i] + rest)
        return pts
    pts = np.array(rec(M, p), dtype=float) / p
    return pts
