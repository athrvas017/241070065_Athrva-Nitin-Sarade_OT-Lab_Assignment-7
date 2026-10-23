"""
Algorithms 10-13: Non-elitist evolutionary methods (Catalogue Part 2).
No archive, no parent+offspring combination -- each generation's population is
entirely replaced, which is exactly the property that distinguishes this family
from the elitist family (14-21) and is worth demonstrating experimentally.
"""
import numpy as np
from ga_core import init_population, evaluate_pop, sbx_crossover, polynomial_mutation
from moo_utils import fast_nondominated_sort, dominates, extract_pareto


# ===================== 10. VEGA (Vector Evaluated GA) =====================
def vega(eval_fn, bounds, N=50, gens=100, seed=0):
    """Split the population into M sub-populations, each selected using ONLY one
    objective (fitness-proportionate-like tournament on that single objective),
    then shuffle and recombine with crossover+mutation across the whole population.
    Known weakness: speciation drifts to the extremes of each objective (tested in Task D)."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    F, V = evaluate_pop(eval_fn, pop)
    M = F.shape[1]
    history_front_sizes = []
    for g in range(gens):
        sub_size = N // M
        selected = []
        for m in range(M):
            key = F[:, m] + 1e6 * V   # lower is better on objective m, penalize infeasible
            for _ in range(sub_size):
                i, j = rng.integers(0, N, size=2)
                selected.append(pop[i] if key[i] <= key[j] else pop[j])
        while len(selected) < N:
            selected.append(pop[rng.integers(0, N)])
        rng.shuffle(selected)
        children = []
        for i in range(0, N - 1, 2):
            c1, c2 = sbx_crossover(selected[i], selected[i+1], bounds, rng=rng)
            c1 = polynomial_mutation(c1, bounds, rng=rng)
            c2 = polynomial_mutation(c2, bounds, rng=rng)
            children += [c1, c2]
        pop = np.array(children[:N])
        F, V = evaluate_pop(eval_fn, pop)
        fronts, _ = fast_nondominated_sort(F)
        history_front_sizes.append(len(fronts[0]))
    return pop, F, V, history_front_sizes


# ===================== 11. MOGA (Fonseca & Fleming) =====================
def moga(eval_fn, bounds, N=50, gens=100, sigma_share=0.1, seed=0):
    """Rank(i) = 1 + (number of individuals that dominate i). Raw fitness assigned
    by rank (lower rank = better), then DEGRADED by fitness sharing within each
    rank to preserve diversity (niching in objective space)."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    for g in range(gens):
        F, V = evaluate_pop(eval_fn, pop)
        Feff = F + 1e6 * V[:, None]
        rank = np.ones(N)
        for i in range(N):
            rank[i] = 1 + sum(1 for j in range(N) if dominates(Feff[j], Feff[i]))
        # fitness sharing within each rank group (niche count on normalized objectives)
        Fn = (Feff - Feff.min(0)) / (Feff.max(0) - Feff.min(0) + 1e-12)
        shared_fitness = rank.copy()
        for r in np.unique(rank):
            idx = np.where(rank == r)[0]
            if len(idx) <= 1: continue
            d = np.sqrt(((Fn[idx,None,:]-Fn[None,idx,:])**2).sum(-1))
            sh = np.maximum(0, 1 - d/sigma_share)
            niche_count = sh.sum(1)
            shared_fitness[idx] = rank[idx] * niche_count / niche_count.mean()
        key = shared_fitness  # lower is better
        children = []
        while len(children) < N:
            i1 = _pick(key, rng); i2 = _pick(key, rng)
            c1, c2 = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
            children += [polynomial_mutation(c1, bounds, rng=rng),
                        polynomial_mutation(c2, bounds, rng=rng)]
        pop = np.array(children[:N])
    F, V = evaluate_pop(eval_fn, pop)
    return pop, F, V


def _pick(key, rng):
    i, j = rng.integers(0, len(key), size=2)
    return i if key[i] <= key[j] else j


# ===================== 12. NSGA (Srinivas & Deb, 1994 -- pre-elitist) =====================
def nsga(eval_fn, bounds, N=50, gens=100, sigma_share=0.1, seed=0):
    """Classify the WHOLE population into non-dominated fronts (not just rank-by-
    domination-count like MOGA). All individuals in front k share a dummy fitness
    LOWER than every individual in front k+1; within a front, fitness is degraded
    by sharing (niching), so no front-k individual can ever outcompete a front-(k-1)
    individual -- but there is still NO elitism/archive across generations."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    for g in range(gens):
        F, V = evaluate_pop(eval_fn, pop)
        Feff = F + 1e6 * V[:, None]
        fronts, _ = fast_nondominated_sort(Feff)
        Fn = (Feff - Feff.min(0)) / (Feff.max(0) - Feff.min(0) + 1e-12)
        dummy = np.zeros(N)
        base = N + 1.0   # front 0 gets the highest dummy fitness
        for front in fronts:
            if not front: continue
            idx = np.array(front)
            if len(idx) > 1:
                d = np.sqrt(((Fn[idx,None,:]-Fn[None,idx,:])**2).sum(-1))
                sh = np.maximum(0, 1 - d/sigma_share)
                niche_count = sh.sum(1)
            else:
                niche_count = np.ones(1)
            shared = base / niche_count
            dummy[idx] = shared
            base = shared.min() - 1e-6
        key = -dummy  # convert to "lower is better" for _pick()
        children = []
        while len(children) < N:
            i1 = _pick(key, rng); i2 = _pick(key, rng)
            c1, c2 = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
            children += [polynomial_mutation(c1, bounds, rng=rng),
                        polynomial_mutation(c2, bounds, rng=rng)]
        pop = np.array(children[:N])
    F, V = evaluate_pop(eval_fn, pop)
    return pop, F, V


# ===================== 13. NPGA (Niched Pareto GA) =====================
def npga(eval_fn, bounds, N=50, gens=100, t_dom=10, sigma_share=0.1, seed=0):
    """Pareto tournament: pick 2 candidates + a random comparison set of size t_dom;
    whichever candidate is dominated by none of the comparison set wins; if both
    (or neither) qualify, break the tie by the LOWER niche count (fewer crowded
    neighbours wins) -- niching is used only as a tie-breaker, not for full ranking."""
    rng = np.random.default_rng(seed)
    pop = init_population(bounds, N, rng)
    for g in range(gens):
        F, V = evaluate_pop(eval_fn, pop)
        Feff = F + 1e6 * V[:, None]
        Fn = (Feff - Feff.min(0)) / (Feff.max(0) - Feff.min(0) + 1e-12)
        def pareto_tournament():
            i, j = rng.integers(0, N, size=2)
            comp = rng.integers(0, N, size=t_dom)
            i_dom = any(dominates(Feff[c], Feff[i]) for c in comp)
            j_dom = any(dominates(Feff[c], Feff[j]) for c in comp)
            if i_dom and not j_dom: return j
            if j_dom and not i_dom: return i
            # tie-break: niche count (fewer neighbours within sigma_share wins)
            ni = np.sum(np.sqrt(((Fn-Fn[i])**2).sum(1)) < sigma_share)
            nj = np.sum(np.sqrt(((Fn-Fn[j])**2).sum(1)) < sigma_share)
            return i if ni <= nj else j
        children = []
        while len(children) < N:
            i1 = pareto_tournament(); i2 = pareto_tournament()
            c1, c2 = sbx_crossover(pop[i1], pop[i2], bounds, rng=rng)
            children += [polynomial_mutation(c1, bounds, rng=rng),
                        polynomial_mutation(c2, bounds, rng=rng)]
        pop = np.array(children[:N])
    F, V = evaluate_pop(eval_fn, pop)
    return pop, F, V
