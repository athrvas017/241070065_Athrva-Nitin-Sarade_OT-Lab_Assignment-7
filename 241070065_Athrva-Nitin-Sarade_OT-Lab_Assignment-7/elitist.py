"""
Algorithms 14-21: Elitist methods (Catalogue Part 3) + Constrained NSGA-II (Part 4).
Elitism = parent+offspring are COMBINED and the best N survive (archive-like
behaviour), unlike the non-elitist family which fully replaces the population.
"""
import numpy as np
from ga_core import init_population, evaluate_pop, sbx_crossover, polynomial_mutation
from moo_utils import fast_nondominated_sort, crowding_distance, dominates


# ===================== 14. NSGA-II (Deb et al., 2002) =====================
def nsga2(eval_fn, bounds, N=50, gens=100, seed=0, constrained=False):
    """1) Combine parent+offspring (2N). 2) Fast non-dominated sort. 3) Fill the
    next generation front-by-front; the LAST front admitted is truncated by
    crowding distance (most-crowded point removed first) to hit exactly N.
    constrained=True switches the comparison to Deb's three feasibility rules
    (used for Algorithm 21, Constrained NSGA-II)."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = deb_rank_key(F, V) if constrained else (F + 1e6*V[:,None])
        children = tournament_offspring(pop, key, bounds, rng, N, constrained, V)
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([V, Vc])
        if constrained:
            fronts, _ = deb_constrained_sort(comb_F, comb_V)
        else:
            Feff = comb_F + 1e6*comb_V[:,None]
            fronts, _ = fast_nondominated_sort(Feff)
        new_idx = []
        for front in fronts:
            if len(new_idx) + len(front) <= N:
                new_idx += front
            else:
                cd = crowding_distance(comb_F, front)
                remaining = N - len(new_idx)
                sorted_front = sorted(front, key=lambda i: -cd[i])
                new_idx += sorted_front[:remaining]
                break
        pop, F, V = comb_pop[new_idx], comb_F[new_idx], comb_V[new_idx]
    return pop, F, V


def tournament_offspring(pop, key, bounds, rng, N, constrained=False, V=None):
    children = []
    while len(children) < N:
        i1 = binary_tournament(key, rng); i2 = binary_tournament(key, rng)
        c1, c2 = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
        children += [polynomial_mutation(c1, bounds, rng=rng), polynomial_mutation(c2, bounds, rng=rng)]
    return np.array(children[:N])


def binary_tournament(key, rng):
    i, j = rng.integers(0, len(key), size=2)
    if np.isscalar(key[i]) or key.ndim == 1:
        return i if key[i] <= key[j] else j
    return i if dominates(key[i], key[j]) else (j if dominates(key[j], key[i]) else (i if rng.random()<0.5 else j))


def deb_rank_key(F, V):
    return F + 1e6*V[:,None]


def deb_constrained_sort(F, V):
    """Deb's three feasibility rules inside non-dominated sorting:
    (1) between two feasible solutions, the one that Pareto-dominates wins;
    (2) a feasible solution always beats an infeasible one;
    (3) between two infeasible solutions, the one with smaller total violation wins."""
    N = len(F)
    def deb_dominates(p, q):
        fp, vp = F[p], V[p]; fq, vq = F[q], V[q]
        if vp < 1e-9 and vq < 1e-9:
            return dominates(fp, fq)
        if vp < 1e-9 and vq >= 1e-9:
            return True
        if vp >= 1e-9 and vq < 1e-9:
            return False
        return vp < vq
    S = [[] for _ in range(N)]; n = np.zeros(N, dtype=int); rank = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q: continue
            if deb_dominates(p, q): S[p].append(q)
            elif deb_dominates(q, p): n[p] += 1
        if n[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i+1; nxt.append(q)
        i += 1; fronts.append(nxt)
    fronts.pop()
    return fronts, rank


# ===================== 21. Constrained NSGA-II =====================
def constrained_nsga2(eval_fn, bounds, N=50, gens=100, seed=0):
    return nsga2(eval_fn, bounds, N=N, gens=gens, seed=seed, constrained=True)


# ===================== 15. NSGA-III (Deb & Jain, 2014) =====================
def das_dennis_ref_points(M, p):
    """Generate structured reference points on the unit simplex (Das & Dennis)."""
    def rec(M, p):
        if M == 1: return [[p]]
        pts = []
        for i in range(p+1):
            for rest in rec(M-1, p-i):
                pts.append([i] + rest)
        return pts
    pts = np.array(rec(M, p), dtype=float) / p
    return pts


def nsga3(eval_fn, bounds, M, N=92, gens=100, p=12, seed=0):
    """Like NSGA-II through front-0..k selection, but once fronts fill PAST N, the
    last front is filled by REFERENCE-POINT NICHING (each point associated to its
    nearest reference line from a Das-Dennis structured set) instead of crowding
    distance -- crowding distance degrades badly once M>=4 (investigated in Task D).
    Step 1: generate Das-Dennis reference points.
    Step 2: combine parent+offspring (2N), sort into fronts.
    Step 3: fill next gen front-by-front.
    Step 4: for the last partial front, apply reference-point niching."""
    rng = np.random.default_rng(seed)
    ref_pts = das_dennis_ref_points(M, p)
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = F + 1e6*V[:,None]
        children = tournament_offspring(pop, key, bounds, rng, N)
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([V, Vc])
        Feff = comb_F + 1e6*comb_V[:,None]
        fronts, _ = fast_nondominated_sort(Feff)
        new_idx = []
        last_front = None
        for front in fronts:
            if len(new_idx) + len(front) <= N:
                new_idx += front
            else:
                last_front = front
                break
        if last_front is not None:
            remaining = N - len(new_idx)
            chosen = ref_point_niching(Feff, new_idx, last_front, ref_pts, remaining)
            new_idx += chosen
        pop, F, V = comb_pop[new_idx], comb_F[new_idx], comb_V[new_idx]
    return pop, F, V


def ref_point_niching(F, selected_idx, last_front, ref_pts, remaining):
    ideal = F.min(axis=0)
    Fn = F - ideal
    scale = Fn.max(axis=0); scale = np.where(scale < 1e-12, 1.0, scale)
    Fn = Fn / scale
    def assoc(idx_set):
        assoc_ref = {}
        for i in idx_set:
            dists = []
            for r in ref_pts:
                rn = r / (np.linalg.norm(r) + 1e-12)
                proj = Fn[i] @ rn
                perp = np.linalg.norm(Fn[i] - proj*rn)
                dists.append(perp)
            j = int(np.argmin(dists))
            assoc_ref[i] = (j, dists[j])
        return assoc_ref
    all_idx = selected_idx + list(last_front)
    assoc_all = assoc(all_idx)
    niche_count = {j: 0 for j in range(len(ref_pts))}
    for i in selected_idx:
        niche_count[assoc_all[i][0]] += 1
    remaining_pool = set(last_front)
    chosen = []
    while len(chosen) < remaining and remaining_pool:
        min_count = min(niche_count.values())
        candidate_refs = [j for j,c in niche_count.items() if c == min_count]
        j_star = candidate_refs[np.random.default_rng(len(chosen)).integers(len(candidate_refs))]
        candidates = [i for i in remaining_pool if assoc_all[i][0] == j_star]
        if not candidates:
            niche_count[j_star] = 10**9
            continue
        if min_count == 0:
            i_star = min(candidates, key=lambda i: assoc_all[i][1])
        else:
            i_star = candidates[np.random.default_rng(len(chosen)+1).integers(len(candidates))]
        chosen.append(i_star); remaining_pool.remove(i_star); niche_count[j_star] += 1
    return chosen


# ===================== 16. SPEA2 (Zitzler et al., 2001) =====================
def spea2(eval_fn, bounds, N=50, archive_size=50, gens=100, seed=0):
    """Strength S(i) = # solutions i dominates. Raw fitness R(i) = sum of S(j) for
    j dominating i (0 for non-dominated). Density D(i) = 1/(sigma_k-th-NN-dist + 2)
    (k = sqrt(pop+archive)). Fitness = R+D (lower better). Fixed-size archive:
    truncate by iteratively removing the point with smallest distance to its
    nearest neighbour when the archive overflows."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    archive = np.zeros((0, pop.shape[1]))
    for g in range(gens):
        F, V = evaluate_pop(eval_fn, pop)
        Fa, Va = evaluate_pop(eval_fn, archive) if len(archive) else (np.zeros((0,F.shape[1])), np.zeros(0))
        comb_pop = np.vstack([pop, archive]) if len(archive) else pop
        comb_F = np.vstack([F, Fa]) if len(archive) else F
        comb_V = np.concatenate([V, Va]) if len(archive) else V
        Feff = comb_F + 1e6*comb_V[:,None]
        M = len(comb_pop)
        S = np.zeros(M)
        for i in range(M):
            S[i] = sum(1 for j in range(M) if i != j and dominates(Feff[i], Feff[j]))
        R = np.zeros(M)
        for i in range(M):
            R[i] = sum(S[j] for j in range(M) if j != i and dominates(Feff[j], Feff[i]))
        k = max(1, int(np.sqrt(M)))
        D = np.zeros(M)
        for i in range(M):
            d = np.sort(np.sqrt(((Feff-Feff[i])**2).sum(1)))
            D[i] = 1.0/(d[min(k,M-1)] + 2.0)
        fitness = R + D
        nd_idx = [i for i in range(M) if fitness[i] < 1.0]
        if len(nd_idx) <= archive_size:
            archive = comb_pop[nd_idx]
            if len(nd_idx) < archive_size:
                rest = sorted(set(range(M))-set(nd_idx), key=lambda i: fitness[i])
                need = archive_size - len(nd_idx)
                archive = np.vstack([archive, comb_pop[rest[:need]]]) if need>0 else archive
        else:
            archive = truncate_by_distance(comb_pop[nd_idx], Feff[nd_idx], archive_size)
        key = fitness
        pop = tournament_offspring(comb_pop, key, bounds, rng, N)
    Fa, Va = evaluate_pop(eval_fn, archive)
    return archive, Fa, Va


def truncate_by_distance(X, F, target_size):
    idx = list(range(len(X)))
    while len(idx) > target_size:
        sub = F[idx]
        d = np.sqrt(((sub[:,None,:]-sub[None,:,:])**2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(1)
        remove = int(np.argmin(nn))
        idx.pop(remove)
    return X[idx]


# ===================== 17. PESA-II (Corne et al., 2001) =====================
def pesa2(eval_fn, bounds, N=50, archive_size=50, gens=100, grid_div=8, seed=0):
    """Maintain a fixed-size, grid-based archive of non-dominated points. Selection
    is REGION-based: pick a hypergrid cell (favoring sparser cells), then a random
    member of that cell."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    archive = pop.copy(); Fa = F.copy(); Va = V.copy()
    archive, Fa, Va = pesa_update_archive(archive, Fa, Va, archive_size)
    for g in range(gens):
        cells = grid_assign(Fa, grid_div)
        children = []
        while len(children) < N:
            i1 = region_select(archive, Fa, cells, rng)
            i2 = region_select(archive, Fa, cells, rng)
            c1, c2 = sbx_crossover(archive[i1], archive[i2], bounds, rng=rng)
            children += [polynomial_mutation(c1, bounds, rng=rng), polynomial_mutation(c2, bounds, rng=rng)]
        children = np.array(children[:N])
        Fc, Vc = evaluate_pop(eval_fn, children)
        archive = np.vstack([archive, children]); Fa = np.vstack([Fa, Fc]); Va = np.concatenate([Va, Vc])
        archive, Fa, Va = pesa_update_archive(archive, Fa, Va, archive_size)
    return archive, Fa, Va


def pesa_update_archive(X, F, V, size):
    Feff = F + 1e6*V[:,None]
    fronts, _ = fast_nondominated_sort(Feff)
    X, F, V = X[fronts[0]], F[fronts[0]], V[fronts[0]]
    if len(X) <= size: return X, F, V
    Feff = F + 1e6*V[:,None]
    cells = grid_assign(Feff, 8)
    while len(X) > size:
        counts = {}
        for c in cells: counts[c] = counts.get(c,0)+1
        densest = max(counts, key=counts.get)
        candidates = [i for i,c in enumerate(cells) if c == densest]
        remove = candidates[np.random.default_rng(len(X)).integers(len(candidates))]
        keep = [i for i in range(len(X)) if i != remove]
        X, F, V = X[keep], F[keep], V[keep]
        cells = [cells[i] for i in keep]
    return X, F, V


def grid_assign(F, div):
    fmin, fmax = F.min(0), F.max(0)
    rng_ = np.where(fmax-fmin < 1e-12, 1.0, fmax-fmin)
    cellidx = np.floor((F-fmin)/rng_*div).clip(0, div-1).astype(int)
    return [tuple(row) for row in cellidx]


def region_select(X, F, cells, rng):
    uniq = list(set(cells))
    counts = {c: cells.count(c) for c in uniq}
    sparse_first = sorted(uniq, key=lambda c: counts[c])
    probs = np.array([1.0/counts[c] for c in sparse_first]); probs /= probs.sum()
    chosen_cell = sparse_first[rng.choice(len(sparse_first), p=probs)]
    members = [i for i,c in enumerate(cells) if c == chosen_cell]
    return members[rng.integers(len(members))]


# ===================== 18/19/20. MOEA/D family =====================
def tchebycheff(f, weight, ideal):
    return np.max(weight * np.abs(f - ideal))


def moead(eval_fn, bounds, N=50, gens=100, T=5, seed=0, variant='basic'):
    """Decompose into N scalar Tchebycheff sub-problems with weight vectors on the
    simplex. Each sub-problem i has a NEIGHBOURHOOD of its T closest weight vectors;
    offspring for sub-problem i are generated FROM its neighbours only (cooperative,
    localized search) and can replace any neighbour whose own Tchebycheff value they
    improve. variant='de' uses differential-evolution-style mutation (MOEA/D-DE);
    variant='aco' uses ant-colony pheromone-weighted sampling (MOEA/D-ACO)."""
    rng = np.random.default_rng(seed)
    M = len(eval_fn(init_population(bounds,1,rng)[0])[0])
    if M == 2:
        w = np.column_stack([np.linspace(0,1,N), 1-np.linspace(0,1,N)])
        w = np.clip(w, 1e-6, None)
    else:
        w = das_dennis_ref_points(M, int(np.ceil(N**(1/(M-1)))))
        w = np.clip(w, 1e-6, None)[:N]
        N = len(w)
    dist = np.sqrt(((w[:,None,:]-w[None,:,:])**2).sum(-1))
    neighbours = np.argsort(dist, axis=1)[:, :T]
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    ideal = F.min(axis=0)
    pheromone = np.ones((N, len(bounds))) if variant == 'aco' else None
    for g in range(gens):
        for i in range(N):
            nb = neighbours[i]
            if variant == 'de':
                r1, r2, r3 = rng.choice(nb, size=3, replace=True)
                child = pop[r1] + 0.5*(pop[r2]-pop[r3])
                child = polynomial_mutation(child, bounds, rng=rng, pm=1.0/len(bounds))
            elif variant == 'aco':
                lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
                base = pop[nb[rng.integers(len(nb))]]
                sigma = (hi-lo)*0.1/ (pheromone[i].mean()+0.1)
                child = np.clip(base + rng.normal(0, sigma), lo, hi)
            else:
                i1, i2 = rng.choice(nb, size=2, replace=True)
                c1, _ = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
                child = polynomial_mutation(c1, bounds, rng=rng)
            fchild, vchild = eval_fn(child)
            feff_child = fchild + 1e6*vchild
            ideal = np.minimum(ideal, fchild)
            for j in nb:
                feff_j = F[j] + 1e6*V[j]
                if tchebycheff(feff_child, w[j], ideal) <= tchebycheff(feff_j, w[j], ideal):
                    pop[j], F[j], V[j] = child, fchild, vchild
                    if variant == 'aco':
                        pheromone[j] = 0.9*pheromone[j] + 0.1
    return pop, F, V


def moead_de(eval_fn, bounds, N=50, gens=100, T=5, seed=0):
    return moead(eval_fn, bounds, N, gens, T, seed, variant='de')


def moead_aco(eval_fn, bounds, N=50, gens=100, T=5, seed=0):
    return moead(eval_fn, bounds, N, gens, T, seed, variant='aco')
