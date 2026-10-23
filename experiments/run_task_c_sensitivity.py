"""
run_task_c_sensitivity.py -- Task C deep-dive sensitivity studies for
algorithms 28 (MO-TLBO), 3 (Benson's Method), and 15 (NSGA-III).
Fixed budget per run: 2000 evals. 10 seeds each, 3 parameter values each.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moo_utils import extract_pareto, hypervolume, igd
from problems import P3_eval, P3_BOUNDS, P3_true_front, P10_eval, P10_BOUNDS

SEEDS = list(range(10))
BUDGET = 2000     # fixed per run (keep it fast; enough to see trends)
TRUE_FRONT = P3_true_front()
REF = np.array([1.1, 1.1])
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# Algorithm 28: MO-TLBO — sensitivity to POPULATION SIZE (N = 20, 50, 100)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Algorithm 28: MO-TLBO Sensitivity (population size N) ===")
from swarm import motlbo

pop_sizes = [20, 50, 100]
motlbo_results = {}
for N in pop_sizes:
    gens = max(1, BUDGET // N)
    hvs, igds = [], []
    for seed in SEEDS:
        try:
            arch_X, arch_F, arch_V = motlbo(P3_eval, P3_BOUNDS, N=N, gens=gens, seed=seed)
            Fp, _, _ = extract_pareto(arch_F)
            hvs.append(hypervolume(Fp, REF))
            igds.append(igd(Fp, TRUE_FRONT))
        except Exception:
            hvs.append(0.0); igds.append(9.9)
    motlbo_results[N] = {'HV': hvs, 'IGD': igds}
    print(f"  N={N:3d} (gens={gens:3d}): HV={np.mean(hvs):.4f}±{np.std(hvs):.4f}  "
          f"IGD={np.mean(igds):.5f}±{np.std(igds):.5f}")

np.save(os.path.join(RESULTS_DIR, 'motlbo_sensitivity.npy'), motlbo_results)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ['HV', 'IGD']):
    means = [np.mean(motlbo_results[N][metric]) for N in pop_sizes]
    stds  = [np.std(motlbo_results[N][metric])  for N in pop_sizes]
    ax.errorbar(pop_sizes, means, yerr=stds, marker='o', capsize=5, color='tab:blue')
    ax.set_xlabel('Population size N'); ax.set_ylabel(metric)
    ax.set_title(f'MO-TLBO (A28): {metric} vs N on P3'); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'motlbo_sensitivity.png'), dpi=100)
plt.close(fig)
print("  Saved motlbo_sensitivity.png and .npy")

# ════════════════════════════════════════════════════════════════════════════
# Algorithm 3: Benson's Method — sensitivity to N_DIRS (5, 10, 20)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Algorithm 3: Benson's Method Sensitivity (n_dirs) ===")
from classical import bensons_method

n_dirs_vals = [5, 10, 20]
ref_pt = np.array([2.0, 2.0])
benson_results = {}
for n_dirs in n_dirs_vals:
    hvs = []
    for seed in SEEDS:
        try:
            res = bensons_method(P3_eval, P3_BOUNDS, ref_point=ref_pt, n_dirs=n_dirs)
            if res:
                pts = np.array([f for x, f, w in res])
                hvs.append(hypervolume(pts, REF))
            else:
                hvs.append(0.0)
        except Exception:
            hvs.append(0.0)
    benson_results[n_dirs] = {'HV': hvs}
    print(f"  n_dirs={n_dirs:2d}: HV={np.mean(hvs):.4f}±{np.std(hvs):.4f}")

np.save(os.path.join(RESULTS_DIR, 'benson_sensitivity.npy'), benson_results)

fig, ax = plt.subplots(figsize=(5, 4))
means = [np.mean(benson_results[n]['HV']) for n in n_dirs_vals]
stds  = [np.std(benson_results[n]['HV'])  for n in n_dirs_vals]
ax.errorbar(n_dirs_vals, means, yerr=stds, marker='s', capsize=5, color='tab:orange')
ax.set_xlabel('Number of directions'); ax.set_ylabel('HV')
ax.set_title("Benson's Method (A3): HV vs n_dirs on P3"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'benson_sensitivity.png'), dpi=100)
plt.close(fig)
print("  Saved benson_sensitivity.png and .npy")

# ════════════════════════════════════════════════════════════════════════════
# Algorithm 15: NSGA-III — sensitivity to divisions p (4, 8, 12) on P10 M=3
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Algorithm 15: NSGA-III Sensitivity (divisions p) on P10 M=3 ===")
from elitist import nsga3

M = 3
bounds_p10 = P10_BOUNDS(M)
eval_p10   = lambda x: P10_eval(x, M=M)
p_vals = [4, 8, 12]
nsga3_results = {}
for p in p_vals:
    N = max(20, p * M)
    gens = max(1, BUDGET // N)
    igds, norms = [], []
    for seed in SEEDS:
        try:
            pop, F, V = nsga3(eval_p10, bounds_p10, M=M, N=N, gens=gens, p=p, seed=seed)
            Fp, _, _ = extract_pareto(F)
            norms.append(np.linalg.norm(Fp, axis=1).mean())
            # Build a small true front for IGD reference
            rng = np.random.default_rng(seed)
            v = np.abs(rng.normal(size=(200, M)))
            true_f = v / np.linalg.norm(v, axis=1, keepdims=True)
            igds.append(igd(Fp, true_f))
        except Exception as e:
            norms.append(0.0); igds.append(9.9)
    nsga3_results[p] = {'mean_norm': norms, 'IGD': igds}
    print(f"  p={p:2d} (N={N}, gens={gens}): mean_norm={np.mean(norms):.3f}±{np.std(norms):.3f}  "
          f"IGD={np.mean(igds):.5f}±{np.std(igds):.5f}")

np.save(os.path.join(RESULTS_DIR, 'nsga3_sensitivity.npy'), nsga3_results)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, label in zip(axes,
                              ['mean_norm', 'IGD'],
                              ['Mean ||f|| (proxy for convergence)', 'IGD']):
    means = [np.mean(nsga3_results[p][metric]) for p in p_vals]
    stds  = [np.std(nsga3_results[p][metric])  for p in p_vals]
    ax.errorbar(p_vals, means, yerr=stds, marker='^', capsize=5, color='tab:green')
    ax.set_xlabel('Divisions p'); ax.set_ylabel(label)
    ax.set_title(f'NSGA-III (A15): {label} vs p on P10 M=3'); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'nsga3_sensitivity.png'), dpi=100)
plt.close(fig)
print("  Saved nsga3_sensitivity.png and .npy")

print("\nTask C sensitivity study complete.")
print(f"Results: {RESULTS_DIR}")
print(f"Figures: {FIGURES_DIR}")
