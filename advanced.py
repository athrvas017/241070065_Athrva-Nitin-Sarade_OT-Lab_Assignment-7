"""
Algorithms 31-37: Many-objective, indicator-based, preference-based and
gradient-based methods (Catalogue Bonus Module 5E).
"""
import numpy as np
from ga_core import init_population, evaluate_pop, sbx_crossover, polynomial_mutation
from moo_utils import fast_nondominated_sort, dominates, hypervolume_mc, hypervolume_2d
from elitist import tournament_offspring, das_dennis_ref_points


# ===================== 31. RVEA (Reference Vector Guided EA) =====================
def rvea(eval_fn, bounds, M, N=60, gens=100, p=8, seed=0, alpha=2.0, fr=0.1):
    """Each solution is associated to its nearest reference VECTOR by smallest
    angle (not nearest reference POINT as in NSGA-III); within each reference-
    vector's associated subset, the winner is chosen by the ANGLE-PENALIZED
    DISTANCE (APD), which inflates distance by a growing penalty on angular
    deviation as generations progress -- this shifts emphasis from diversity
    (early) to convergence (late) automatically."""
    rng = np.random.default_rng(seed)
    V0 = das_dennis_ref_points(M, p)
    V0 = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    V = V0.copy()
    N = len(V)
    pop = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = F + 1e6*Viol[:,None]
        children = tournament_offspring(pop, key, bounds, rng, N)
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([Viol, Vc])
        zmin = comb_F.min(axis=0)
        Ft = comb_F - zmin
        norms = np.linalg.norm(Ft, axis=1, keepdims=True); norms[norms<1e-12]=1e-12
        Ft_unit = Ft / norms
        cos_sim = Ft_unit @ V.T
        assoc = np.argmax(cos_sim, axis=1)
        gamma = np.zeros(len(V))
        for j in range(len(V)):
            members = np.where(assoc==j)[0]
            if len(members)==0: continue
            angles = np.arccos(np.clip(cos_sim[members,j], -1, 1))
            gamma[j] = angles.min() if len(angles) else 1.0
        selected = []
        for j in range(len(V)):
            members = np.where(assoc==j)[0]
            if len(members)==0: continue
            theta = np.arccos(np.clip(cos_sim[members,j], -1, 1))
            Mval = Ft.shape[1]
            apd = (1 + Mval*(g/gens)**alpha * theta/max(gamma[j],1e-9)) * np.linalg.norm(Ft[members], axis=1)
            feasible_mask = comb_V[members] < 1e-9
            if feasible_mask.any():
                cand = members[feasible_mask]
                apd_cand = apd[feasible_mask]
            else:
                cand = members; apd_cand = apd
            selected.append(cand[np.argmin(apd_cand)])
        while len(selected) < N:
            selected.append(rng.integers(len(comb_pop)))
        selected = selected[:N]
        pop, F, Viol = comb_pop[selected], comb_F[selected], comb_V[selected]
        if (g+1) % max(1,int(fr*gens)) == 0:
            V = V0 * (comb_F.max(0) - zmin + 1e-9)
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return pop, F, Viol


# ===================== 32. IBEA ($\epsilon$-indicator) =====================
def eps_indicator(fi, fj):
    """Additive epsilon indicator I_eps(i,j) = min eps s.t. fi - eps <= fj componentwise."""
    return np.max(fi - fj)


def ibea(eval_fn, bounds, N=40, gens=100, kappa=0.05, seed=0):
    """Fitness(i) = -sum_{j!=i} exp(-I_eps(j,i)/(kappa*c)), i.e. EVERY pairwise
    epsilon-indicator value directly shapes fitness -- no explicit diversity-
    preservation step (no crowding, no niching) is used at all."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        Feff = F + 1e6*Viol[:,None]
        c = np.max(np.abs(Feff)) + 1e-9
        M = len(pop)
        I = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i != j: I[i,j] = eps_indicator(Feff[j], Feff[i])
        fitness = np.array([ -sum(np.exp(-I[j,i]/(kappa*c)) for j in range(M) if j!=i) for i in range(M)])
        survivors = list(range(M))
        while len(survivors) > N:
            worst = min(survivors, key=lambda i: fitness[i])
            for i in survivors:
                if i != worst:
                    fitness[i] += np.exp(-I[worst,i]/(kappa*c))
            survivors.remove(worst)
        pop, F, Viol = pop[survivors], F[survivors], Viol[survivors]
        key = -fitness[survivors]
        children = tournament_offspring(pop, key, bounds, rng, N)
        Fc, Vc = evaluate_pop(eval_fn, children)
        pop, F, Viol = children, Fc, Vc
    return pop, F, Viol


# ===================== 33. SMS-EMOA =====================
def sms_emoa(eval_fn, bounds, N=40, gens=200, ref_margin=1.1, seed=0):
    """Steady-state (ONE offspring per iteration): combine with the current
    population (N+1), non-dominated sort, and from the WORST front discard the
    point with the LEAST exclusive hypervolume contribution."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        i1, i2 = rng.integers(0, N, size=2)
        c, _ = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
        c = polynomial_mutation(c, bounds, rng=rng)
        fc, vc = eval_fn(c)
        comb_pop = np.vstack([pop, c[None,:]]); comb_F = np.vstack([F, fc[None,:]]); comb_V = np.concatenate([Viol, [vc]])
        Feff = comb_F + 1e6*comb_V[:,None]
        fronts, _ = fast_nondominated_sort(Feff)
        worst_front = fronts[-1]
        if len(worst_front) == 1:
            remove = worst_front[0]
        else:
            ref = Feff[worst_front].max(axis=0) * ref_margin + 1e-6
            contribs = []
            for idx in worst_front:
                rest = [i for i in worst_front if i != idx]
                hv_all = hypervolume_2d(Feff[worst_front], ref) if Feff.shape[1]==2 else hypervolume_mc(Feff[worst_front], ref, n_samples=20000)
                hv_wo = hypervolume_2d(Feff[rest], ref) if Feff.shape[1]==2 else hypervolume_mc(Feff[rest], ref, n_samples=20000)
                contribs.append(hv_all - hv_wo)
            remove = worst_front[int(np.argmin(contribs))]
        keep = [i for i in range(len(comb_pop)) if i != remove]
        pop, F, Viol = comb_pop[keep], comb_F[keep], comb_V[keep]
    return pop, F, Viol


# ===================== 34. HypE =====================
def hype(eval_fn, bounds, N=40, gens=100, n_samples=5000, seed=0):
    """Monte-Carlo ESTIMATE of each individual's exclusive hypervolume
    contribution -- used BOTH as the environmental-selection criterion and as
    parent fitness."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = F + 1e6*Viol[:,None]
        children = tournament_offspring(pop, key, bounds, rng, N)
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([Viol, Vc])
        Feff = comb_F + 1e6*comb_V[:,None]
        ref = Feff.max(axis=0) * 1.1 + 1e-6
        hv_contrib = hype_mc_contributions(Feff, ref, n_samples, rng)
        order = np.argsort(-hv_contrib)
        keep = order[:N]
        pop, F, Viol = comb_pop[keep], comb_F[keep], comb_V[keep]
    return pop, F, Viol


def hype_mc_contributions(F, ref, n_samples, rng):
    fmin = F.min(axis=0)
    samples = rng.uniform(fmin, ref, size=(n_samples, F.shape[1]))
    contrib = np.zeros(len(F))
    dom_counts = np.zeros(n_samples, dtype=int)
    dom_mask = np.zeros((len(F), n_samples), dtype=bool)
    for i, p in enumerate(F):
        dom_mask[i] = np.all(samples >= p, axis=1)
        dom_counts += dom_mask[i]
    vol = np.prod(ref - fmin)
    for i in range(len(F)):
        contrib[i] = vol * np.sum(dom_mask[i] / np.maximum(dom_counts, 1)) / n_samples
    return contrib


# ===================== 35. R-NSGA-II (preference-based) =====================
def r_nsga2(eval_fn, bounds, ref_point, N=40, gens=100, epsilon=0.01, seed=0):
    """Like NSGA-II, but crowding distance is REPLACED by distance to a
    decision-maker-supplied reference point with an epsilon-clearing step."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = F + 1e6*Viol[:,None]
        children = tournament_offspring(pop, key, bounds, rng, N)
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([Viol, Vc])
        Feff = comb_F + 1e6*comb_V[:,None]
        fronts, _ = fast_nondominated_sort(Feff)
        new_idx = []
        for front in fronts:
            if len(new_idx) + len(front) <= N:
                new_idx += front
            else:
                remaining = N - len(new_idx)
                sub = list(front)
                chosen = []
                dists = {i: np.linalg.norm(comb_F[i]-ref_point) for i in sub}
                while len(chosen) < remaining and sub:
                    best = min(sub, key=lambda i: dists[i])
                    chosen.append(best)
                    sub = [i for i in sub if np.linalg.norm(comb_F[i]-comb_F[best]) > epsilon]
                    if best in sub: sub.remove(best)
                if len(chosen) < remaining:
                    rest = [i for i in front if i not in chosen]
                    rest.sort(key=lambda i: dists[i])
                    chosen += rest[:remaining-len(chosen)]
                new_idx += chosen[:remaining]
                break
        pop, F, Viol = comb_pop[new_idx], comb_F[new_idx], comb_V[new_idx]
    return pop, F, Viol


# ===================== 36. MGDA (gradient-based) =====================
def mgda(grad_fn, x0, iters=200, lr=0.05):
    """Multi-Gradient Descent Algorithm: at each step, find the minimum-norm
    convex combination alpha of the per-task gradients (closed form for M=2:
    alpha* = clip( (g2.(g2-g1)) / ||g1-g2||^2 , 0, 1) ), giving a single common
    descent direction d = alpha*g1 + (1-alpha)*g2 that cannot increase EITHER
    task loss to first order."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    alphas = []
    for it in range(iters):
        g1, g2 = grad_fn(x)
        diff = g1 - g2
        denom = diff @ diff
        if denom < 1e-12:
            alpha = 0.5
        else:
            alpha = np.clip((g2 @ diff) / denom, 0.0, 1.0)
        d = alpha*g1 + (1-alpha)*g2
        x = x - lr*d
        history.append(x.copy()); alphas.append(alpha)
    return x, np.array(history), np.array(alphas)


# ===================== 37. GradNorm (gradient-based) =====================
def gradnorm_toy(grad_fn, loss_fn, x0, iters=200, lr=0.05, lr_w=0.025, gn_alpha=1.5):
    """Adaptive task-loss weights w1,w2 (init 1,1): at each step compute each
    task's gradient NORM weighted by w_i, compare to the TARGET norm implied by
    its relative training rate (loss_i(t)/loss_i(0))^alpha, and nudge w_i toward
    equalizing weighted gradient norms across tasks."""
    x = np.array(x0, dtype=float)
    w = np.array([1.0, 1.0])
    L0 = np.array(loss_fn(x))
    history = [x.copy()]; w_history = [w.copy()]
    for it in range(iters):
        g1, g2 = grad_fn(x)
        L = np.array(loss_fn(x))
        x = x - lr * (w[0]*g1 + w[1]*g2)
        gn1, gn2 = np.linalg.norm(w[0]*g1), np.linalg.norm(w[1]*g2)
        Lr = L / (L0 + 1e-9)
        Lr_mean = Lr.mean()
        target1 = (gn1+gn2)/2 * (Lr[0]/Lr_mean)**gn_alpha
        target2 = (gn1+gn2)/2 * (Lr[1]/Lr_mean)**gn_alpha
        gw = np.array([abs(gn1-target1), abs(gn2-target2)])
        w = w - lr_w * gw * np.sign(np.array([gn1-target1, gn2-target2]))
        w = np.clip(w, 0.01, None)
        w = w / w.sum() * 2.0   # renormalize so weights sum to 2
        history.append(x.copy()); w_history.append(w.copy())
    return x, np.array(history), np.array(w_history)
