"""
Algorithms 22-30: Swarm, physics-inspired, evolutionary, human-inspired and hybrid
metaheuristics (Catalogue Part 4), all built around a shared EXTERNAL ARCHIVE of
non-dominated solutions that plays the role gbest/leader/teacher plays in the
single-objective versions of these algorithms.
"""
import numpy as np
from ga_core import init_population, evaluate_pop, sbx_crossover, polynomial_mutation
from moo_utils import fast_nondominated_sort, dominates, crowding_distance
from elitist import truncate_by_distance, pesa_update_archive, grid_assign


def select_leader(archive_X, archive_F, rng, grid_div=8):
    """Pick a leader from the archive favoring LESS crowded regions (grid-based,
    as in MOPSO's original design)."""
    if len(archive_X) == 1:
        return archive_X[0]
    cells = grid_assign(archive_F, grid_div)
    uniq = list(set(cells))
    counts = {c: cells.count(c) for c in uniq}
    probs = np.array([1.0/counts[c] for c in cells]); probs /= probs.sum()
    i = rng.choice(len(archive_X), p=probs)
    return archive_X[i]


def update_archive_(aX, aF, aV, nX, nF, nV, cap):
    allX = np.vstack([aX, nX]); allF = np.vstack([aF, nF]); allV = np.concatenate([aV, nV])
    Feff = allF + 1e6*allV[:,None]
    fronts,_ = fast_nondominated_sort(Feff)
    allX, allF, allV = allX[fronts[0]], allF[fronts[0]], allV[fronts[0]]
    if len(allX) > cap:
        Xc = truncate_by_distance(allX, allF, cap)
        keep = [int(np.where((allX==row).all(1))[0][0]) for row in Xc]
        allX, allF, allV = allX[keep], allF[keep], allV[keep]
    return allX, allF, allV


# ===================== 22. MOPSO =====================
def mopso(eval_fn, bounds, N=40, gens=100, archive_cap=50, seed=0, w=0.4, c1=1.5, c2=1.5):
    """Each particle's velocity pulls it toward its OWN personal best (first
    non-dominating replacement) and a LEADER picked from the external archive
    (not a single global best, since there is no single best in MOO)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = init_population(bounds, N, rng)
    V = rng.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, size=(N, len(bounds)))
    F, Viol = evaluate_pop(eval_fn, X)
    pbest_X, pbest_F = X.copy(), F.copy()
    arch_X, arch_F, arch_V = X.copy(), F.copy(), Viol.copy()
    Feff = arch_F + 1e6*arch_V[:,None]
    fronts,_ = fast_nondominated_sort(Feff)
    arch_X, arch_F, arch_V = arch_X[fronts[0]], arch_F[fronts[0]], arch_V[fronts[0]]
    for g in range(gens):
        for i in range(N):
            leader = select_leader(arch_X, arch_F, rng)
            r1, r2 = rng.random(len(bounds)), rng.random(len(bounds))
            V[i] = w*V[i] + c1*r1*(pbest_X[i]-X[i]) + c2*r2*(leader-X[i])
            X[i] = np.clip(X[i] + V[i], lo, hi)
        F, Viol = evaluate_pop(eval_fn, X)
        for i in range(N):
            if dominates(F[i]+1e6*Viol[i], pbest_F[i]) or rng.random()<0.1:
                pbest_X[i], pbest_F[i] = X[i].copy(), F[i].copy()
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, X, F, Viol, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 23. MOGWO =====================
def mogwo(eval_fn, bounds, N=40, gens=100, archive_cap=50, seed=0):
    """alpha/beta/delta leaders are the TOP-3 least-crowded points taken from the
    external archive each iteration (standard GWO encircling equations, applied
    per-wolf toward all three archive leaders)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, X)
    arch_X, arch_F, arch_V = update_archive_(X[:0], F[:0], Viol[:0], X, F, Viol, archive_cap)
    for g in range(gens):
        a = 2.0 - 2.0*g/gens
        if len(arch_X) >= 3:
            cd = crowding_distance(arch_F, range(len(arch_X)))
            top3 = sorted(cd, key=lambda i: -cd[i])[:3]
        else:
            top3 = list(range(len(arch_X))) * 3
        leaders = [arch_X[i] for i in top3[:3]]
        for i in range(N):
            cand = []
            for leader in leaders:
                Avec = 2*a*rng.random(len(bounds)) - a
                Cvec = 2*rng.random(len(bounds))
                D = np.abs(Cvec*leader - X[i])
                cand.append(leader - Avec*D)
            X[i] = np.clip(np.mean(cand, axis=0), lo, hi)
        F, Viol = evaluate_pop(eval_fn, X)
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, X, F, Viol, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 24. MOABC =====================
def moabc(eval_fn, bounds, N=40, gens=100, limit=20, archive_cap=50, seed=0):
    """Employed/onlooker/scout bees as in single-objective ABC, but a trial
    REPLACES the food source only if it DOMINATES it (greedy acceptance becomes
    Pareto-dominance acceptance)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, X)
    trial = np.zeros(N, dtype=int)
    arch_X, arch_F, arch_V = update_archive_(X[:0], F[:0], Viol[:0], X, F, Viol, archive_cap)
    def neighbour_search(i):
        k = rng.choice([j for j in range(N) if j != i])
        j = rng.integers(len(bounds))
        v = X[i].copy()
        v[j] = X[i][j] + rng.uniform(-1,1)*(X[i][j]-X[k][j])
        v = np.clip(v, lo, hi)
        fv, vv = eval_fn(v)
        if dominates(fv+1e6*vv, F[i]+1e6*Viol[i]):
            X[i], F[i], Viol[i], trial[i] = v, fv, vv, 0
        else:
            trial[i] += 1
        return v, fv, vv
    for g in range(gens):
        news = []
        for i in range(N):
            news.append(neighbour_search(i))
        for _ in range(N):
            i = rng.integers(N)
            news.append(neighbour_search(i))
        m = int(np.argmax(trial))
        if trial[m] > limit:
            X[m] = init_population(bounds, 1, rng)[0]
            F[m], Viol[m] = eval_fn(X[m]); trial[m] = 0
        nX = np.array([n[0] for n in news]); nF = np.array([n[1] for n in news]); nV = np.array([n[2] for n in news])
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, nX, nF, nV, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 25. MOACO =====================
def moaco(eval_fn_perm, n_items, N=30, gens=60, archive_cap=30, seed=0, n_obj=2,
         cost_matrices=None, alpha=1.0, beta=2.0, rho=0.1):
    """Ant Colony for PERMUTATION problems (P8/TSP) with TWO pheromone matrices,
    one per objective, combined multiplicatively when choosing the next city."""
    rng = np.random.default_rng(seed)
    pher = [np.ones((n_items, n_items)) for _ in range(n_obj)]
    heur = [1.0/(cost_matrices[o] + 1e-6) for o in range(n_obj)]
    best_archive_X, best_archive_F = [], []
    for g in range(gens):
        tours, objs = [], []
        for ant in range(N):
            unvisited = set(range(n_items)); cur = rng.integers(n_items); tour = [cur]; unvisited.discard(cur)
            while unvisited:
                cand = list(unvisited)
                probs = np.array([ (pher[0][cur,j]**alpha * heur[0][cur,j]**beta) *
                                   (pher[1][cur,j]**alpha * heur[1][cur,j]**beta) for j in cand])
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(cand))/len(cand)
                nxt = cand[rng.choice(len(cand), p=probs)]
                tour.append(nxt); unvisited.discard(nxt); cur = nxt
            f, viol = eval_fn_perm(tour)
            tours.append(tour); objs.append(f)
        objs = np.array(objs)
        allX = best_archive_X + tours
        allF = np.vstack([np.array(best_archive_F)]+[objs]) if best_archive_F else objs
        fronts,_ = fast_nondominated_sort(allF)
        nd = fronts[0]
        best_archive_X = [allX[i] for i in nd]
        best_archive_F = [allF[i] for i in nd]
        if len(best_archive_X) > archive_cap:
            Farr = np.array(best_archive_F)
            Xc = truncate_by_distance(np.array(best_archive_X), Farr, archive_cap)
            best_archive_X = [list(r) for r in Xc]
            best_archive_F = best_archive_F[:archive_cap]
        for o in range(n_obj):
            pher[o] *= (1-rho)
        for tour in best_archive_X:
            for o in range(n_obj):
                tlen = sum(cost_matrices[o][tour[k], tour[(k+1)%n_items]] for k in range(n_items))
                for k in range(n_items):
                    a_idx, b_idx = tour[k], tour[(k+1)%n_items]
                    pher[o][a_idx,b_idx] += 1.0/(tlen+1e-6); pher[o][b_idx,a_idx] += 1.0/(tlen+1e-6)
    return best_archive_X, np.array(best_archive_F)


# ===================== 26. MOSA =====================
def mosa(eval_fn, bounds, archive_cap=50, iters=3000, seed=0, T0=10.0, Tmin=1e-3, cooling=0.995):
    """Single-trajectory simulated annealing generalized to MOO: a perturbed
    neighbour is accepted if it DOMINATES the current point, or (if neither
    dominates) with Boltzmann probability exp(-Delta/T) on a scalarized comparison."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x = init_population(bounds, 1, rng)[0]
    f, v = eval_fn(x)
    arch_X, arch_F, arch_V = update_archive_(x[None,:][:0], f[None,:][:0], np.array([])[:0], x[None,:], f[None,:], np.array([v]), archive_cap)
    T = T0
    for it in range(iters):
        step = (hi-lo)*0.05
        x_new = np.clip(x + rng.normal(0, step), lo, hi)
        f_new, v_new = eval_fn(x_new)
        feff, feff_new = f + 1e6*v, f_new + 1e6*v_new
        if dominates(feff_new, feff):
            x, f, v = x_new, f_new, v_new
        elif not dominates(feff, feff_new):
            w = rng.random(); w = np.array([w, 1-w])
            delta = (w@feff_new[:2]) - (w@feff[:2])
            if rng.random() < np.exp(-max(delta,0)/max(T,1e-9)):
                x, f, v = x_new, f_new, v_new
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, x[None,:], f[None,:], np.array([v]), archive_cap)
        T = max(Tmin, T*cooling)
    return arch_X, arch_F, arch_V


# ===================== 27. MODE =====================
def mode(eval_fn, bounds, N=40, gens=100, archive_cap=50, seed=0, F_scale=0.5, CR=0.9):
    """DE/rand/1/bin mutation+crossover per individual; trial REPLACES target only
    if it DOMINATES the target (Pareto-based greedy selection)."""
    rng = np.random.default_rng(seed)
    X = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, X)
    arch_X, arch_F, arch_V = update_archive_(X[:0], F[:0], Viol[:0], X, F, Viol, archive_cap)
    for g in range(gens):
        for i in range(N):
            others = [j for j in range(N) if j != i]
            r1,r2,r3 = rng.choice(others, size=3, replace=False)
            donor = X[r1] + F_scale*(X[r2]-X[r3])
            donor = np.clip(donor, [b[0] for b in bounds], [b[1] for b in bounds])
            cross = rng.random(len(bounds)) < CR
            cross[rng.integers(len(bounds))] = True
            trial = np.where(cross, donor, X[i])
            ft, vt = eval_fn(trial)
            if dominates(ft+1e6*vt, F[i]+1e6*Viol[i]):
                X[i], F[i], Viol[i] = trial, ft, vt
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, X, F, Viol, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 28. MO-TLBO =====================
def motlbo(eval_fn, bounds, N=40, gens=100, archive_cap=50, seed=0):
    """Teacher phase: move each learner toward a RANDOMLY-CHOSEN non-dominated
    "teacher" from the current class (no single best exists in MOO) minus a
    teaching-factor-weighted class mean. Learner phase: pair with a random peer,
    move toward the Pareto-better one. Both phases accept the move only if it
    dominates (or does not worsen) the learner.
    Step 1: init population.
    Step 2: non-dominated sort -> identify teacher candidates (front 0).
    Step 3: teacher phase - update each learner toward a random teacher minus TF*mean.
    Step 4: learner phase - pair with random peer, move toward better by dominance.
    Step 5: update external archive."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, X)
    arch_X, arch_F, arch_V = update_archive_(X[:0], F[:0], Viol[:0], X, F, Viol, archive_cap)
    for g in range(gens):
        Feff = F + 1e6*Viol[:,None]
        fronts,_ = fast_nondominated_sort(Feff)
        nd = fronts[0]
        mean = X.mean(axis=0)
        # Teacher phase
        for i in range(N):
            teacher = X[rng.choice(nd)]
            TF = rng.integers(1,3)
            new = np.clip(X[i] + rng.random(len(bounds))*(teacher - TF*mean), lo, hi)
            fn, vn = eval_fn(new)
            if dominates(fn+1e6*vn, F[i]+1e6*Viol[i]):
                X[i], F[i], Viol[i] = new, fn, vn
        # Learner phase
        for i in range(N):
            j = rng.choice([k for k in range(N) if k != i])
            r = rng.random(len(bounds))
            if dominates(F[i]+1e6*Viol[i], F[j]+1e6*Viol[j]):
                new = X[i] + r*(X[i]-X[j])
            else:
                new = X[i] + r*(X[j]-X[i])
            new = np.clip(new, lo, hi)
            fn, vn = eval_fn(new)
            if dominates(fn+1e6*vn, F[i]+1e6*Viol[i]):
                X[i], F[i], Viol[i] = new, fn, vn
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, X, F, Viol, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 29. MOBA =====================
def moba(eval_fn, bounds, N=40, gens=100, archive_cap=50, seed=0, fmin=0.0, fmax=2.0,
         alpha=0.9, gamma=0.9, A0=1.0, r0=0.5):
    """Bat algorithm: velocity update pulls each bat toward a LEADER chosen from
    the external archive (replacing the single global best of single-objective BA)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = init_population(bounds, N, rng)
    F, Viol = evaluate_pop(eval_fn, X)
    Vvel = np.zeros_like(X); vmax = (hi-lo)*0.2
    A = np.full(N, A0); r = np.full(N, r0)
    arch_X, arch_F, arch_V = update_archive_(X[:0], F[:0], Viol[:0], X, F, Viol, archive_cap)
    for g in range(1, gens+1):
        for i in range(N):
            leader = select_leader(arch_X, arch_F, rng)
            freq = fmin + (fmax-fmin)*rng.random()
            Vvel[i] = np.clip(Vvel[i] + (X[i]-leader)*freq, -vmax, vmax)
            x_new = np.clip(X[i] + Vvel[i], lo, hi)
            if rng.random() > r[i]:
                x_new = np.clip(leader + 0.02*(hi-lo)*rng.uniform(-1,1,len(bounds))*A.mean(), lo, hi)
            fn, vn = eval_fn(x_new)
            if rng.random() < A[i] and dominates(fn+1e6*vn, F[i]+1e6*Viol[i]):
                X[i], F[i], Viol[i] = x_new, fn, vn
                A[i] *= alpha; r[i] = r0*(1-np.exp(-gamma*g))
        arch_X, arch_F, arch_V = update_archive_(arch_X, arch_F, arch_V, X, F, Viol, archive_cap)
    return arch_X, arch_F, arch_V


# ===================== 30. MOMA (memetic hybrid) =====================
def moma(eval_fn, bounds, N=40, gens=60, local_iters=5, archive_cap=50, seed=0):
    """NSGA-II global search, PLUS a hill-climbing local-search refinement applied
    to a sample of offspring each generation (Lamarckian: the improved individual
    replaces its parent in the population)."""
    from elitist import tournament_offspring
    from moo_utils import crowding_distance
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    for g in range(gens):
        key = F + 1e6*V[:,None]
        children = tournament_offspring(pop, key, bounds, rng, N)
        n_local = max(1, N//5)
        local_idx = rng.choice(N, size=n_local, replace=False)
        for i in local_idx:
            x = children[i].copy()
            fx, vx = eval_fn(x)
            for _ in range(local_iters):
                x_try = np.clip(x + rng.normal(0,(hi-lo)*0.02), lo, hi)
                fx_try, vx_try = eval_fn(x_try)
                if dominates(fx_try+1e6*vx_try, fx+1e6*vx):
                    x, fx, vx = x_try, fx_try, vx_try
            children[i] = x
        Fc, Vc = evaluate_pop(eval_fn, children)
        comb_pop = np.vstack([pop, children]); comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([V, Vc])
        Feff = comb_F + 1e6*comb_V[:,None]
        fronts,_ = fast_nondominated_sort(Feff)
        new_idx = []
        for front in fronts:
            if len(new_idx)+len(front) <= N:
                new_idx += front
            else:
                cd = crowding_distance(comb_F, front)
                remaining = N - len(new_idx)
                new_idx += sorted(front, key=lambda i: -cd[i])[:remaining]
                break
        pop, F, V = comb_pop[new_idx], comb_F[new_idx], comb_V[new_idx]
    return pop, F, V
