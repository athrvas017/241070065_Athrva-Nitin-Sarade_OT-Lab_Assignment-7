"""
run_task_d_classical.py -- Task D: Classical methods sweeps and TOPSIS/VIKOR analysis.
- Algorithms 1-4: sweep weights / epsilon / p settings (>=10 values each)
- Algorithms 5-7: report solution for stated preference inputs
- Algorithms 8-9: apply TOPSIS and VIKOR to NSGA-II front on P4, shift one weight by 0.1
- Algorithms 36-37: compare MGDA vs GradNorm vs equal weights on P13
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from problems import *

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== Task D: Classical Methods ===\n")

# ---- Algorithms 1-4: Sweeps ----
from classical import weighted_sum, epsilon_constraint, bensons_method, weight_metric

print("[1] Weighted Sum on P1 (20 weights):")
res = weighted_sum(P1_eval, P1_BOUNDS, n_weights=20)
for x,f,w in res:
    print(f"   w1={w[0]:.2f} -> x={x[0]:.3f}, f=({f[0]:.3f},{f[1]:.3f})")

print("\n[2] Epsilon-Constraint on P2 (20 eps values):")
res = epsilon_constraint(P2_eval, P2_BOUNDS, eps_range=(4,100), n_eps=20)
for x,f,eps in res:
    print(f"   eps={eps:.1f} -> x={x[0]:.3f}, f=({f[0]:.3f},{f[1]:.3f})")

print("\n[3] Benson's Method on P1 (15 directions):")
res = bensons_method(P1_eval, P1_BOUNDS, ref_point=np.array([16.0,16.0]), n_dirs=15)
for x,f,w in res:
    print(f"   w=({w[0]:.2f},{w[1]:.2f}) -> x={x[0]:.3f}, f=({f[0]:.3f},{f[1]:.3f})")

print("\n[4] Weight Metric (p=1,2,inf) on P1 (11 weights each):")
for p in [1, 2, np.inf]:
    res = weight_metric(P1_eval, P1_BOUNDS, ideal=np.array([0.0,0.0]), p=p, n_weights=11)
    print(f"   p={p}: {len(res)} points found")

# ---- Algorithms 5-7: Preference inputs ----
from classical import value_function, goal_programming, lexicographic

print("\n[5] Value Function on P4 (utility = -3*risk + 1*return):")
utility = lambda f: -3.0*f[0] - 1.0*f[1]
x, f, _ = value_function(P4_eval, P4_BOUNDS, utility)
print(f"   Optimal weights: {np.round(x,3)}, risk={f[0]:.4f}, return={-f[1]:.4f}")

print("\n[6] Goal Programming on P11 (profit target=2400, units target=600):")
x, f, info = goal_programming(P11_eval, P11_BOUNDS, targets=np.array([-2400,-600]))
print(f"   x=({x[0]:.1f},{x[1]:.1f}), profit={-f[0]:.1f}, units={-f[1]:.1f}, shortfall={info['shortfall']:.1f}")

print("\n[7] Lexicographic on P11 (profit first, units second):")
x, f, achieved = lexicographic(P11_eval, P11_BOUNDS, priority_order=[0,1])
print(f"   x=({x[0]:.1f},{x[1]:.1f}), profit={-f[0]:.1f}, units={-f[1]:.1f}")
print(f"   Achieved at each priority level: {[-a for a in achieved]}")

# ---- Algorithms 8-9: TOPSIS and VIKOR on P12 ----
from classical import topsis, vikor
from problems import P12_DATA, P12_WEIGHTS, P12_SENSE, P12_NAMES

print("\n[8] TOPSIS on P12 (supplier selection):")
rank, closeness = topsis(P12_DATA, P12_WEIGHTS, P12_SENSE)
for r in rank:
    print(f"   {P12_NAMES[r]}: closeness = {closeness[r]:.4f}")

print("\n   Shift weight[0] by +0.1 (from 0.3 to 0.4):")
w_shifted = P12_WEIGHTS.copy(); w_shifted[0] += 0.1; w_shifted /= w_shifted.sum()
rank2, closeness2 = topsis(P12_DATA, w_shifted, P12_SENSE)
print(f"   Original top-3: {[P12_NAMES[r] for r in rank[:3]]}")
print(f"   Shifted  top-3: {[P12_NAMES[r] for r in rank2[:3]]}")
print(f"   Ranking changed: {[P12_NAMES[r] for r in rank[:3]] != [P12_NAMES[r] for r in rank2[:3]]}")

print("\n[9] VIKOR on P12 (v = 0.25, 0.5, 0.75):")
for v in [0.25, 0.5, 0.75]:
    rank_v, Q, S, R = vikor(P12_DATA, P12_WEIGHTS, P12_SENSE, v=v)
    print(f"   v={v}: {[P12_NAMES[i] for i in rank_v]}")

# ---- Algorithms 36-37: MGDA vs GradNorm vs equal weights on P13 ----
from advanced import mgda, gradnorm_toy
print("\n[36] MGDA on P13:")
x_mgda, hist, alphas = mgda(P13_grad, x0=[0,5], iters=200)
f_mgda, _ = P13_eval(x_mgda)
print(f"   x*={np.round(x_mgda,4)}, f=({f_mgda[0]:.4f},{f_mgda[1]:.4f})")

print("\n[37] GradNorm on P13:")
loss_fn = lambda x: P13_eval(x)[0]
x_gn, hist_gn, w_hist = gradnorm_toy(P13_grad, loss_fn, x0=[0,5], iters=200)
f_gn, _ = P13_eval(x_gn)
print(f"   x*={np.round(x_gn,4)}, f=({f_gn[0]:.4f},{f_gn[1]:.4f}), final_weights={np.round(w_hist[-1],3)}")

print("\n   Equal weights (w=[1,1]) baseline (no adaptive balancing):")
x_eq = np.array([0.0, 5.0])
lr = 0.05
for _ in range(200):
    g1, g2 = P13_grad(x_eq)
    x_eq = x_eq - lr * (g1 + g2)
f_eq, _ = P13_eval(x_eq)
print(f"   x*={np.round(x_eq,4)}, f=({f_eq[0]:.4f},{f_eq[1]:.4f})")

print("\nTask D Classical methods complete.")
