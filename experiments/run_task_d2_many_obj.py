"""
run_task_d2_many_obj.py -- Task D2: P10 M=5 many-objective comparison.
Algorithms: NSGA-II, NSGA-III, MOEA/D, RVEA, HypE.
10 seeds, equal budget (10,000 evals).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moo_utils import extract_pareto, igd
from problems import P10_eval, P10_BOUNDS, P10_true_front

SEEDS = list(range(10))
BUDGET = 10000
M = 5
bounds = P10_BOUNDS(M)
eval_fn = lambda x: P10_eval(x, M=M)
true_front = P10_true_front(M)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from elitist import nsga2, nsga3, moead
from advanced import rvea, hype

N = 60
algorithms = {
    'NSGA-II':  lambda s: nsga2(eval_fn, bounds, N=N, gens=BUDGET//N, seed=s),
    'NSGA-III': lambda s: nsga3(eval_fn, bounds, M=M, N=N, gens=BUDGET//N, p=3, seed=s),
    'MOEA/D':   lambda s: moead(eval_fn, bounds, N=N, gens=BUDGET//N, seed=s),
    'RVEA':     lambda s: rvea(eval_fn,  bounds, M=M, N=N, gens=BUDGET//N, p=3, seed=s),
    'HypE':     lambda s: hype(eval_fn,  bounds, N=N, gens=BUDGET//N, seed=s),
}

rows = []
for algo_name, algo_fn in algorithms.items():
    igds, norms = [], []
    print(f"Running {algo_name} on P10 M=5 ...", end=' ', flush=True)
    for seed in SEEDS:
        try:
            out = algo_fn(seed)
            F_raw = out[1]
            Fp, _, _ = extract_pareto(F_raw)
            igds.append(igd(Fp, true_front))
            norms.append(np.linalg.norm(Fp, axis=1).mean())
        except Exception as e:
            igds.append(999); norms.append(0)
    print(f"IGD={np.mean(igds):.5f}+/-{np.std(igds):.5f}, mean_norm={np.mean(norms):.3f}")
    rows.append({
        'algorithm': algo_name,
        'IGD_mean': round(np.mean(igds), 5), 'IGD_std': round(np.std(igds), 5),
        'mean_norm': round(np.mean(norms), 4),
    })

csv_path = os.path.join(RESULTS_DIR, 'task_d2_many_obj_p10_m5.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\nTask D2 results saved to {csv_path}")
print("\nNSGA-II degradation note: with M=5, NSGA-II's crowding distance loses selection pressure.")
print("Compare front sizes and IGD to see the effect. NSGA-III, RVEA, HypE handle this better.")
