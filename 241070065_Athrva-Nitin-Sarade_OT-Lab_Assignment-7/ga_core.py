"""
ga_core.py -- shared real-coded GA scaffolding reused by algorithms 10-21, 30
(SBX crossover, polynomial mutation, tournament selection). Each algorithm supplies
its own SELECTION/RANKING logic; this module only supplies the generic GA plumbing.
"""
import numpy as np


def sbx_crossover(p1, p2, bounds, eta=15, rng=None, pc=0.9):
    rng = np.random.default_rng() if rng is None else rng
    p1, p2 = np.array(p1, float), np.array(p2, float)
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        lo, hi = bounds[i]
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rng.random()
        beta = 1 + 2*(x1 - lo)/(x2 - x1 + 1e-14)
        alpha = 2 - beta**(-(eta+1))
        betaq = (u*alpha)**(1/(eta+1)) if u <= 1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        c1v = 0.5*((x1+x2) - betaq*(x2-x1))
        beta = 1 + 2*(hi - x2)/(x2 - x1 + 1e-14)
        alpha = 2 - beta**(-(eta+1))
        betaq = (u*alpha)**(1/(eta+1)) if u <= 1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        c2v = 0.5*((x1+x2) + betaq*(x2-x1))
        c1[i] = np.clip(c1v, lo, hi); c2[i] = np.clip(c2v, lo, hi)
    return c1, c2


def polynomial_mutation(x, bounds, eta=20, rng=None, pm=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.array(x, float)
    n = len(x)
    pm = 1.0/n if pm is None else pm
    for i in range(n):
        if rng.random() > pm:
            continue
        lo, hi = bounds[i]
        u = rng.random()
        delta1 = (x[i]-lo)/(hi-lo+1e-14); delta2 = (hi-x[i])/(hi-lo+1e-14)
        if u < 0.5:
            deltaq = (2*u + (1-2*u)*(1-delta1)**(eta+1))**(1/(eta+1)) - 1
        else:
            deltaq = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta2)**(eta+1))**(1/(eta+1))
        x[i] = np.clip(x[i] + deltaq*(hi-lo), lo, hi)
    return x


def init_population(bounds, N, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return rng.uniform(lo, hi, size=(N, len(bounds)))


def evaluate_pop(eval_fn, pop):
    out = [eval_fn(x) for x in pop]
    F = np.array([o[0] for o in out])
    V = np.array([o[1] for o in out])
    return F, V


def tournament_select(key_values, rng, k=2):
    """key_values: array where LOWER is better (e.g. rank, or -fitness). Binary tournament."""
    N = len(key_values)
    i, j = rng.integers(0, N, size=2)
    return i if key_values[i] <= key_values[j] else j


def make_offspring(pop, parent_key, bounds, rng, eta_c=15, eta_m=20):
    """Generic: binary tournament (by parent_key, lower better) + SBX + polynomial mutation."""
    N = len(pop)
    children = []
    while len(children) < N:
        i1 = tournament_select(parent_key, rng); i2 = tournament_select(parent_key, rng)
        c1, c2 = sbx_crossover(pop[i1], pop[i2], bounds, eta=eta_c, rng=rng)
        c1 = polynomial_mutation(c1, bounds, eta=eta_m, rng=rng)
        c2 = polynomial_mutation(c2, bounds, eta=eta_m, rng=rng)
        children.append(c1); children.append(c2)
    return np.array(children[:N])
