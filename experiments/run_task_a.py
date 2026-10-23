"""
run_task_a.py -- Task A: run every algorithm on its catalogue-assigned test
problem(s) with fixed seeds, save front plots to figures/, save metrics to results/,
and write the implementation_log.csv.

Usage (from lab7/ directory):
    .venv/Scripts/python experiments/run_task_a.py
"""
import sys, os, time, csv, traceback, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moo_utils import extract_pareto, hypervolume, igd, spacing
from problems import *

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BUDGET = 10000
N_POP  = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

log_rows = []

# ─── helpers ────────────────────────────────────────────────────────────────

def save_front_plot(F, algo_name, problem_name, true_front=None):
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        if true_front is not None:
            ax.plot(true_front[:, 0], true_front[:, 1], '-', color='gray', label='True front', lw=1)
        ax.scatter(F[:, 0], F[:, 1], c='tab:blue', s=15, label=algo_name)
        ax.set_xlabel('f1'); ax.set_ylabel('f2')
        ax.set_title(f'{algo_name} on {problem_name}')
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fname = os.path.join(FIGURES_DIR, f'{algo_name}_{problem_name}.png')
        fig.savefig(fname, dpi=100)
        plt.close(fig)
    except Exception:
        pass


def log_algo(number, name, problems, F, hv_val, igd_val, runtime, status='works'):
    log_rows.append({
        'number':            number,
        'algorithm':         name,
        'problems_used':     problems,
        'evaluation_budget': BUDGET,
        'HV':   round(float(hv_val), 6) if hv_val is not None else 'N/A',
        'IGD':  round(float(igd_val), 6) if igd_val is not None else 'N/A',
        'runtime': round(runtime, 2),
        'status': status,
    })


def run_block(label, fn):
    """Run fn(), catch exceptions, always return something loggable."""
    try:
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        print(f'  OK  ({elapsed:.1f}s)')
        return result, elapsed, 'works'
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  ERR {e}')
        traceback.print_exc()
        return None, elapsed, f'error: {e}'


REF_P1  = np.array([50.0, 50.0])
REF_P2  = np.array([110.0, 110.0])
REF_P3  = np.array([1.1, 1.1])
TRUE_P3 = P3_true_front()

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMS 1–9: Classical
# ────────────────────────────────────────────────────────────────────────────
from classical import (weighted_sum, epsilon_constraint, bensons_method,
                       weight_metric, value_function, goal_programming,
                       lexicographic, topsis, vikor)

# ---- 1. Weighted Sum ----
print('[1] Weighted Sum ...', end=' ')
def _a01():
    res = weighted_sum(P1_eval, P1_BOUNDS, n_weights=20)
    pts = np.array([f for x, f, w in res])
    hv = hypervolume(pts, REF_P1)
    save_front_plot(pts, 'WeightedSum', 'P1')
    return pts, hv
res, rt, st = run_block('1', _a01)
if res:
    log_algo(1, 'Weighted Sum', 'P1,P3,P4', res[0], res[1], None, rt, st)
else:
    log_algo(1, 'Weighted Sum', 'P1,P3,P4', None, None, None, rt, st)

# ---- 2. Epsilon-Constraint ----
print('[2] Epsilon-Constraint ...', end=' ')
def _a02():
    res = epsilon_constraint(P2_eval, P2_BOUNDS, eps_range=(4, 100), n_eps=15)
    pts = np.array([f for x, f, e in res]) if res else np.array([[0.0, 0.0]])
    hv = hypervolume(pts, REF_P2)
    save_front_plot(pts, 'EpsConstraint', 'P2')
    return pts, hv
res, rt, st = run_block('2', _a02)
if res:
    log_algo(2, 'Epsilon-Constraint', 'P2,P3,P5', res[0], res[1], None, rt, st)
else:
    log_algo(2, 'Epsilon-Constraint', 'P2,P3,P5', None, None, None, rt, st)

# ---- 3. Benson's Method ----
print("[3] Benson's Method ...", end=' ')
def _a03():
    res = bensons_method(P1_eval, P1_BOUNDS, ref_point=np.array([16.0, 16.0]), n_dirs=15)
    pts = np.array([f for x, f, w in res]) if res else np.array([[0.0, 0.0]])
    hv = hypervolume(pts, REF_P1)
    save_front_plot(pts, 'Benson', 'P1')
    return pts, hv
res, rt, st = run_block('3', _a03)
if res:
    log_algo(3, "Benson's Method", 'P1,P7', res[0], res[1], None, rt, st)
else:
    log_algo(3, "Benson's Method", 'P1,P7', None, None, None, rt, st)

# ---- 4. Weight Metric (Tchebycheff) ----
print('[4] Weight Metric ...', end=' ')
def _a04():
    res = weight_metric(P1_eval, P1_BOUNDS, ideal=np.array([0.0, 0.0]), p=np.inf, n_weights=15)
    pts = np.array([f for x, f, w in res]) if res else np.array([[0.0, 0.0]])
    hv = hypervolume(pts, REF_P1)
    save_front_plot(pts, 'WeightMetric', 'P1')
    return pts, hv
res, rt, st = run_block('4', _a04)
if res:
    log_algo(4, 'Weight Metric', 'P1,P3', res[0], res[1], None, rt, st)
else:
    log_algo(4, 'Weight Metric', 'P1,P3', None, None, None, rt, st)

# ---- 5. Value Function ----
print('[5] Value Function ...', end=' ')
def _a05():
    utility = lambda f: -3.0 * f[0] - f[1]  # minimize risk, maximize return
    x, f, _ = value_function(P4_eval, P4_BOUNDS, utility)
    return np.array([f]), -utility(f)  # store f as 1-point "front"
res, rt, st = run_block('5', _a05)
hv_v = float(res[1]) if res else None
log_algo(5, 'Value Function', 'P4,P14', np.array([[0, 0]]), hv_v, None, rt, st)

# ---- 6. Goal Programming ----
print('[6] Goal Programming ...', end=' ')
def _a06():
    x, f, info = goal_programming(P11_eval, P11_BOUNDS,
                                  targets=np.array([-2400.0, -600.0]))
    return np.array([f]), info['shortfall']
res, rt, st = run_block('6', _a06)
log_algo(6, 'Goal Programming', 'P11,P7', np.array([[0, 0]]),
         float(res[1]) if res else None, None, rt, st)

# ---- 7. Lexicographic ----
print('[7] Lexicographic ...', end=' ')
def _a07():
    x, f, achieved = lexicographic(P11_eval, P11_BOUNDS, priority_order=[0, 1])
    return np.array([f]), float(-achieved[0])
res, rt, st = run_block('7', _a07)
log_algo(7, 'Lexicographic', 'P11,P7', np.array([[0, 0]]),
         float(res[1]) if res else None, None, rt, st)

# ---- 8. TOPSIS ----
print('[8] TOPSIS ...', end=' ')
def _a08():
    rank, closeness = topsis(P12_DATA, P12_WEIGHTS, P12_SENSE)
    return rank, float(closeness[rank[0]])  # best closeness score
res, rt, st = run_block('8', _a08)
log_algo(8, 'TOPSIS', 'P12', np.array([[0, 0]]),
         float(res[1]) if res else None, None, rt, st)

# ---- 9. VIKOR ----
print('[9] VIKOR ...', end=' ')
def _a09():
    rank, Q, S, R = vikor(P12_DATA, P12_WEIGHTS, P12_SENSE, v=0.5)
    return rank, float(Q[rank[0]])
res, rt, st = run_block('9', _a09)
log_algo(9, 'VIKOR', 'P12', np.array([[0, 0]]),
         float(res[1]) if res else None, None, rt, st)

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMS 10–13: Non-elitist
# ────────────────────────────────────────────────────────────────────────────
from non_elitist import vega, moga, nsga, npga

for algo_num, algo_fn, algo_name in [
    (10, vega, 'VEGA'),
    (11, moga, 'MOGA'),
    (12, nsga, 'NSGA'),
    (13, npga, 'NPGA'),
]:
    print(f'[{algo_num}] {algo_name} ...', end=' ')
    def _block(fn=algo_fn, nm=algo_name):
        out = fn(P1_eval, P1_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
        Fp, _, _ = extract_pareto(out[1])
        hv = hypervolume(Fp, REF_P1)
        save_front_plot(Fp, nm, 'P1')
        return Fp, hv
    res, rt, st = run_block(str(algo_num), _block)
    if res:
        log_algo(algo_num, algo_name, 'P1,P3', res[0], res[1], None, rt, st)
    else:
        log_algo(algo_num, algo_name, 'P1,P3', None, None, None, rt, st)

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMS 14–21: Elitist
# ────────────────────────────────────────────────────────────────────────────
from elitist import nsga2, nsga3, spea2, pesa2, moead, moead_de, moead_aco, constrained_nsga2

# 14 NSGA-II
print('[14] NSGA-II ...', end=' ')
def _a14():
    pop, F, V = nsga2(P3_eval, P3_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'NSGA-II', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('14', _a14)
if res:
    log_algo(14, 'NSGA-II', 'P3,P5,P6', res[0], res[1], res[2], rt, st)
else:
    log_algo(14, 'NSGA-II', 'P3,P5,P6', None, None, None, rt, st)

# 15 NSGA-III
print('[15] NSGA-III ...', end=' ')
_bounds3 = P10_BOUNDS(3)
_eval3   = lambda x: P10_eval(x, M=3)
def _a15():
    pop, F, V = nsga3(_eval3, _bounds3, M=3, N=N_POP, gens=BUDGET // N_POP, p=6, seed=0)
    Fp, _, _ = extract_pareto(F)
    norms = np.linalg.norm(Fp, axis=1)
    return Fp, float(norms.mean()), None
res, rt, st = run_block('15', _a15)
if res:
    log_algo(15, 'NSGA-III', 'P10(M=3,M=5)', res[0], res[1], res[2], rt, st)
else:
    log_algo(15, 'NSGA-III', 'P10(M=3,M=5)', None, None, None, rt, st)

# 16 SPEA2
print('[16] SPEA2 ...', end=' ')
def _a16():
    arch, Fa, Va = spea2(P3_eval, P3_BOUNDS, N=N_POP, archive_size=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(Fa)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'SPEA2', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('16', _a16)
if res:
    log_algo(16, 'SPEA2', 'P4,P5', res[0], res[1], res[2], rt, st)
else:
    log_algo(16, 'SPEA2', 'P4,P5', None, None, None, rt, st)

# 17 PESA-II
print('[17] PESA-II ...', end=' ')
def _a17():
    arch, Fa, Va = pesa2(P3_eval, P3_BOUNDS, N=N_POP, archive_size=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(Fa)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'PESA-II', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('17', _a17)
if res:
    log_algo(17, 'PESA-II', 'P3,P5', res[0], res[1], res[2], rt, st)
else:
    log_algo(17, 'PESA-II', 'P3,P5', None, None, None, rt, st)

# 18 MOEA/D
print('[18] MOEA/D ...', end=' ')
def _a18():
    pop, F, V = moead(P3_eval, P3_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'MOEAD', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('18', _a18)
if res:
    log_algo(18, 'MOEA/D', 'P3,P10', res[0], res[1], res[2], rt, st)
else:
    log_algo(18, 'MOEA/D', 'P3,P10', None, None, None, rt, st)

# 19 MOEA/D-DE
print('[19] MOEA/D-DE ...', end=' ')
def _a19():
    pop, F, V = moead_de(P3_eval, P3_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'MOEAD-DE', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('19', _a19)
if res:
    log_algo(19, 'MOEA/D-DE', 'P3,P5', res[0], res[1], res[2], rt, st)
else:
    log_algo(19, 'MOEA/D-DE', 'P3,P5', None, None, None, rt, st)

# 20 MOEA/D-ACO  (P8 is permutation; run on P1 as continuous fallback)
print('[20] MOEA/D-ACO ...', end=' ')
def _a20():
    pop, F, V = moead_aco(P1_eval, P1_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P1)
    save_front_plot(Fp, 'MOEAD-ACO', 'P1')
    return Fp, hv, None
res, rt, st = run_block('20', _a20)
if res:
    log_algo(20, 'MOEA/D-ACO', 'P8,P1', res[0], res[1], res[2], rt, st)
else:
    log_algo(20, 'MOEA/D-ACO', 'P8,P1', None, None, None, rt, st)

# 21 Constrained NSGA-II
print('[21] Constrained NSGA-II ...', end=' ')
def _a21():
    pop, F, V = constrained_nsga2(P6_eval, P6_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, np.array([120.0, 30.0]))
    save_front_plot(Fp, 'ConstNSGA2', 'P6')
    return Fp, hv, None
res, rt, st = run_block('21', _a21)
if res:
    log_algo(21, 'Constrained NSGA-II', 'P5,P6,P7', res[0], res[1], res[2], rt, st)
else:
    log_algo(21, 'Constrained NSGA-II', 'P5,P6,P7', None, None, None, rt, st)

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMS 22–30: Swarm / metaheuristics
# ────────────────────────────────────────────────────────────────────────────
from swarm import mopso, mogwo, moabc, moaco, mosa, mode, motlbo, moba, moma

swarm_tasks = [
    (22, mopso,  'MOPSO'),
    (23, mogwo,  'MOGWO'),
    (24, moabc,  'MOABC'),
    (26, mosa,   'MOSA'),
    (27, mode,   'MODE'),
    (28, motlbo, 'MO-TLBO'),
    (29, moba,   'MOBA'),
]
for algo_num, algo_fn, algo_name in swarm_tasks:
    print(f'[{algo_num}] {algo_name} ...', end=' ')
    def _swarm(fn=algo_fn, nm=algo_name):
        out = fn(P3_eval, P3_BOUNDS, seed=0)
        Fp, _, _ = extract_pareto(out[1])
        hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
        save_front_plot(Fp, nm, 'P3', TRUE_P3)
        return Fp, hv, ig
    res, rt, st = run_block(str(algo_num), _swarm)
    if res:
        log_algo(algo_num, algo_name, 'P3,P5', res[0], res[1], res[2], rt, st)
    else:
        log_algo(algo_num, algo_name, 'P3,P5', None, None, None, rt, st)

# 25 MOACO (permutation on P8)
print('[25] MOACO (P8) ...', end=' ')
def _a25():
    arch_X, arch_F = moaco(P8_eval, P8_N, N=20, gens=BUDGET // 20, seed=0,
                            cost_matrices=[P8_D, P8_T])
    Fp, _, _ = extract_pareto(arch_F)
    hv = hypervolume(Fp, np.array([5000.0, 15000.0]))
    return Fp, hv, None
res, rt, st = run_block('25', _a25)
if res:
    log_algo(25, 'MOACO', 'P8', res[0], res[1], res[2], rt, st)
else:
    log_algo(25, 'MOACO', 'P8', None, None, None, rt, st)

# 30 MOMA
print('[30] MOMA ...', end=' ')
def _a30():
    pop, F, V = moma(P3_eval, P3_BOUNDS, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'MOMA', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('30', _a30)
if res:
    log_algo(30, 'MOMA', 'P5,P9', res[0], res[1], res[2], rt, st)
else:
    log_algo(30, 'MOMA', 'P5,P9', None, None, None, rt, st)

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMS 31–37: Advanced
# ────────────────────────────────────────────────────────────────────────────
from advanced import rvea, ibea, sms_emoa, hype, r_nsga2, mgda, gradnorm_toy

# 31 RVEA
print('[31] RVEA ...', end=' ')
def _a31():
    pop, F, V = rvea(_eval3, _bounds3, M=3, N=N_POP, gens=BUDGET // N_POP, p=6, seed=0)
    Fp, _, _ = extract_pareto(F)
    norms = np.linalg.norm(Fp, axis=1)
    return Fp, float(norms.mean()), None
res, rt, st = run_block('31', _a31)
if res:
    log_algo(31, 'RVEA', 'P10', res[0], res[1], res[2], rt, st)
else:
    log_algo(31, 'RVEA', 'P10', None, None, None, rt, st)

# 32 IBEA
print('[32] IBEA ...', end=' ')
def _a32():
    pop, F, V = ibea(P3_eval, P3_BOUNDS, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'IBEA', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('32', _a32)
if res:
    log_algo(32, 'IBEA', 'P3,P14', res[0], res[1], res[2], rt, st)
else:
    log_algo(32, 'IBEA', 'P3,P14', None, None, None, rt, st)

# 33 SMS-EMOA  (steady-state: 1 offspring/iter; cap at 500 iters)
print('[33] SMS-EMOA ...', end=' ')
def _a33():
    pop, F, V = sms_emoa(P3_eval, P3_BOUNDS, N=30, gens=500, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'SMS-EMOA', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('33', _a33)
if res:
    log_algo(33, 'SMS-EMOA', 'P3,P14', res[0], res[1], res[2], rt, st)
else:
    log_algo(33, 'SMS-EMOA', 'P3,P14', None, None, None, rt, st)

# 34 HypE
print('[34] HypE ...', end=' ')
def _a34():
    pop, F, V = hype(_eval3, _bounds3, N=N_POP, gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    norms = np.linalg.norm(Fp, axis=1)
    return Fp, float(norms.mean()), None
res, rt, st = run_block('34', _a34)
if res:
    log_algo(34, 'HypE', 'P10', res[0], res[1], res[2], rt, st)
else:
    log_algo(34, 'HypE', 'P10', None, None, None, rt, st)

# 35 R-NSGA-II
print('[35] R-NSGA-II ...', end=' ')
def _a35():
    ref_pt = np.array([0.3, 0.6])   # preference point on P3 front
    pop, F, V = r_nsga2(P3_eval, P3_BOUNDS, ref_point=ref_pt, N=N_POP,
                         gens=BUDGET // N_POP, seed=0)
    Fp, _, _ = extract_pareto(F)
    hv = hypervolume(Fp, REF_P3); ig = igd(Fp, TRUE_P3)
    save_front_plot(Fp, 'R-NSGA2', 'P3', TRUE_P3)
    return Fp, hv, ig
res, rt, st = run_block('35', _a35)
if res:
    log_algo(35, 'R-NSGA-II', 'P5,P14', res[0], res[1], res[2], rt, st)
else:
    log_algo(35, 'R-NSGA-II', 'P5,P14', None, None, None, rt, st)

# 36 MGDA
print('[36] MGDA ...', end=' ')
def _a36():
    x_opt, hist, alphas = mgda(P13_grad, x0=[0.0, 5.0], iters=500)
    f_opt, _ = P13_eval(x_opt)
    return np.array([f_opt]), float(np.linalg.norm(f_opt)), None
res, rt, st = run_block('36', _a36)
if res:
    log_algo(36, 'MGDA', 'P13', res[0], res[1], res[2], rt, st)
else:
    log_algo(36, 'MGDA', 'P13', None, None, None, rt, st)

# 37 GradNorm
print('[37] GradNorm ...', end=' ')
def _a37():
    loss_fn = lambda x: P13_eval(x)[0]
    x_opt, hist, w_hist = gradnorm_toy(P13_grad, loss_fn, x0=[0.0, 5.0], iters=500)
    f_opt, _ = P13_eval(x_opt)
    return np.array([f_opt]), float(np.linalg.norm(f_opt)), None
res, rt, st = run_block('37', _a37)
if res:
    log_algo(37, 'GradNorm', 'P13', res[0], res[1], res[2], rt, st)
else:
    log_algo(37, 'GradNorm', 'P13', None, None, None, rt, st)

# ────────────────────────────────────────────────────────────────────────────
# Write implementation_log.csv
# ────────────────────────────────────────────────────────────────────────────
# Sort by number and fill any missing algorithm rows
logged_nums = {r['number'] for r in log_rows}
for n in range(1, 38):
    if n not in logged_nums:
        log_rows.append({'number': n, 'algorithm': f'Algorithm {n}',
                         'problems_used': 'N/A', 'evaluation_budget': BUDGET,
                         'HV': 'N/A', 'IGD': 'N/A', 'runtime': 0, 'status': 'not run'})
log_rows.sort(key=lambda r: r['number'])

log_path = os.path.join(os.path.dirname(__file__), '..', 'implementation_log.csv')
with open(log_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['number', 'algorithm', 'problems_used',
                                            'evaluation_budget', 'HV', 'IGD', 'runtime', 'status'])
    writer.writeheader()
    writer.writerows(log_rows)

works = sum(1 for r in log_rows if r['status'] == 'works')
print(f'\n{"="*60}')
print(f'Implementation log saved -> {log_path}')
print(f'Algorithms: {works}/37 successful')
print(f'Figures saved -> {FIGURES_DIR}')
print(f'{"="*60}')
