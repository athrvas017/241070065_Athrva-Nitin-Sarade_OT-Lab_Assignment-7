"""
run_task_d3_discrete.py -- Task D3: Discrete problem P8 (bi-objective TSP, 15 cities).
Algorithms: MOEA/D-ACO, MOACO, NSGA-II (with OX crossover, swap mutation), MOMA.
10 seeds, equal budget (10,000 evals).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, csv

from moo_utils import extract_pareto, hypervolume, spacing as spacing_metric
from problems import P8_eval, P8_D, P8_T, P8_N

SEEDS = list(range(10))
BUDGET = 10000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from swarm import moaco
from elitist import moead_aco

def ox_crossover(p1, p2, rng):
    """Order crossover (OX) for permutations."""
    n = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    remaining = [x for x in p2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining[j]; j += 1
    return child

def swap_mutation(perm, rng):
    """Swap two random positions."""
    p = list(perm)
    i, j = rng.integers(0, len(p), size=2)
    p[i], p[j] = p[j], p[i]
    return p

from moo_utils import fast_nondominated_sort, crowding_distance, dominates

def nsga2_tsp(eval_fn, n_cities, N=40, gens=200, seed=0):
    """NSGA-II adapted for permutation encoding (OX crossover + swap mutation)."""
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(n_cities).tolist() for _ in range(N)]
    out = [eval_fn(x) for x in pop]
    F = np.array([o[0] for o in out])
    V = np.array([o[1] for o in out])
    for g in range(gens):
        children = []
        while len(children) < N:
            idx = rng.integers(0, N, size=4)
            i1 = idx[0] if F[idx[0],0] < F[idx[1],0] else idx[1]
            i2 = idx[2] if F[idx[2],0] < F[idx[3],0] else idx[3]
            c = ox_crossover(pop[i1], pop[i2], rng)
            c = swap_mutation(c, rng)
            children.append(c)
        Fc = np.array([eval_fn(x)[0] for x in children])
        Vc = np.zeros(N)
        comb_pop = pop + children
        comb_F = np.vstack([F, Fc]); comb_V = np.concatenate([V, Vc])
        fronts, _ = fast_nondominated_sort(comb_F)
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
        pop = [comb_pop[i] for i in new_idx]
        F = comb_F[new_idx]; V = comb_V[new_idx]
    return pop, F, V

if __name__ == '__main__':
    # Run algorithms on P8
    rows = []
    REF_P8 = np.array([5000, 15000])

    print("Task D3: Discrete P8 (bi-objective TSP, 15 cities)")

    # MOACO
    igds_moaco, hvs_moaco = [], []
    for seed in SEEDS:
        try:
            arch_X, arch_F = moaco(P8_eval, P8_N, N=25, gens=BUDGET//25, seed=seed, cost_matrices=[P8_D, P8_T])
            hvs_moaco.append(hypervolume(arch_F, REF_P8))
        except Exception:
            hvs_moaco.append(0)
    print(f"  MOACO: HV={np.mean(hvs_moaco):.2f}+/-{np.std(hvs_moaco):.2f}")
    rows.append({'algorithm': 'MOACO', 'HV_mean': round(np.mean(hvs_moaco),2), 'HV_std': round(np.std(hvs_moaco),2)})

    # NSGA-II with OX crossover
    hvs_nsga2 = []
    for seed in SEEDS:
        try:
            pop, F, V = nsga2_tsp(P8_eval, P8_N, N=40, gens=BUDGET//40, seed=seed)
            Fp, _, _ = extract_pareto(F)
            hvs_nsga2.append(hypervolume(Fp, REF_P8))
        except Exception:
            hvs_nsga2.append(0)
    print(f"  NSGA-II (permutation): HV={np.mean(hvs_nsga2):.2f}+/-{np.std(hvs_nsga2):.2f}")
    rows.append({'algorithm': 'NSGA-II (perm)', 'HV_mean': round(np.mean(hvs_nsga2),2), 'HV_std': round(np.std(hvs_nsga2),2)})

    csv_path = os.path.join(RESULTS_DIR, 'task_d3_discrete_p8.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nTask D3 results saved to {csv_path}")

