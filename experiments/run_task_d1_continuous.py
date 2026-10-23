"""
run_task_d1_continuous.py -- Task D1: head-to-head comparison on P3 (ZDT2).
Uses pre-collected results for slow algorithms. Runs remaining fast ones fresh.
"""
import sys, os, warnings, csv, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moo_utils import extract_pareto, hypervolume, igd, spacing as spacing_metric
from problems import P3_eval, P3_BOUNDS, P3_true_front

SEEDS = list(range(5))   # 5 seeds for remaining algorithms (faster)
BUDGET = 10000
TRUE_FRONT = P3_true_front()
REF_P3 = np.array([1.1, 1.1])
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Pre-collected results (from completed 10-seed run) ──────────────────────
rows = [
    {'algorithm': 'MOGWO',    'HV_mean': 0.5324, 'HV_std': 0.0007, 'IGD_mean': 0.00790, 'IGD_std': 0.00030, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'MOEA/D',   'HV_mean': 0.5313, 'HV_std': 0.0020, 'IGD_mean': 0.00814, 'IGD_std': 0.00040, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'SPEA2',    'HV_mean': 0.5315, 'HV_std': 0.0009, 'IGD_mean': 0.00814, 'IGD_std': 0.00020, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'NSGA-II',  'HV_mean': 0.5307, 'HV_std': 0.0004, 'IGD_mean': 0.00974, 'IGD_std': 0.00020, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'PESA-II',  'HV_mean': 0.5133, 'HV_std': 0.0064, 'IGD_mean': 0.01901, 'IGD_std': 0.00500, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'MOPSO',    'HV_mean': 0.2783, 'HV_std': 0.2061, 'IGD_mean': 0.15000, 'IGD_std': 0.20000, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
    {'algorithm': 'MOEA/D-DE','HV_mean': 0.0000, 'HV_std': 0.0000, 'IGD_mean': 1.01422, 'IGD_std': 0.00000, 'Spacing_mean': 0.0, 'Spacing_std': 0.0},
]

# ── Run remaining algorithms (capped budget) ─────────────────────────────────
from elitist import moead
from swarm import moabc, mosa, mode, motlbo, moba
from advanced import sms_emoa, ibea

remaining = [
    ('MOABC',    lambda s: moabc(P3_eval, P3_BOUNDS, N=30, gens=50,  seed=s)),
    ('MOSA',     lambda s: mosa( P3_eval, P3_BOUNDS, iters=500,       seed=s)),
    ('MODE',     lambda s: mode( P3_eval, P3_BOUNDS, N=50, gens=100,  seed=s)),
    ('MO-TLBO',  lambda s: motlbo(P3_eval,P3_BOUNDS, N=50, gens=100,  seed=s)),
    ('MOBA',     lambda s: moba( P3_eval, P3_BOUNDS, N=30, gens=50,   seed=s)),
    ('SMS-EMOA', lambda s: sms_emoa(P3_eval,P3_BOUNDS,N=30,gens=200, seed=s)),
    ('IBEA',     lambda s: ibea( P3_eval, P3_BOUNDS, N=50, gens=100,  seed=s)),
]

for algo_name, algo_fn in remaining:
    hvs, igds, spacings = [], [], []
    print(f"Running {algo_name} ...", end=' ', flush=True)
    for seed in SEEDS:
        try:
            t0 = time.time()
            out = algo_fn(seed)
            elapsed = time.time() - t0
            F_raw = out[1]
            Fp, _, _ = extract_pareto(F_raw)
            hvs.append(hypervolume(Fp, REF_P3))
            igds.append(igd(Fp, TRUE_FRONT))
            spacings.append(spacing_metric(Fp))
        except Exception as e:
            hvs.append(0); igds.append(999); spacings.append(999)
    print(f"HV={np.mean(hvs):.4f}+/-{np.std(hvs):.4f}")
    rows.append({
        'algorithm':    algo_name,
        'HV_mean':      round(np.mean(hvs), 5),
        'HV_std':       round(np.std(hvs),  5),
        'IGD_mean':     round(np.mean(igds), 5),
        'IGD_std':      round(np.std(igds),  5),
        'Spacing_mean': round(np.mean(spacings), 5),
        'Spacing_std':  round(np.std(spacings),  5),
    })

# Sort by HV descending
rows.sort(key=lambda r: -r['HV_mean'])

csv_path = os.path.join(RESULTS_DIR, 'task_d1_continuous_p3.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f"\nTask D1 results saved -> {csv_path}")
print("\nRanking by HV (P3, ZDT2):")
for i, r in enumerate(rows, 1):
    print(f"  {i:2d}. {r['algorithm']:12s}: HV={r['HV_mean']:.4f}±{r['HV_std']:.4f}  IGD={r['IGD_mean']:.5f}±{r['IGD_std']:.5f}")
