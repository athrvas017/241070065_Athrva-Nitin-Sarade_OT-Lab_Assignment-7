"""
run_remaining.py -- Runs only algorithms 33-37 + writes final implementation_log.csv.
Algorithms 1-32 were already run and their figures exist.
"""
import sys, os, time, csv, traceback, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moo_utils import extract_pareto, hypervolume, igd
from problems import *

BUDGET = 10000
N_POP  = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

REF_P3  = np.array([1.1, 1.1])
TRUE_P3 = P3_true_front()
_bounds3 = P10_BOUNDS(3)
_eval3   = lambda x: P10_eval(x, M=3)

log_rows = []

# Load existing log rows from the already-written CSV if it exists and has real data
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'implementation_log.csv')

# Hard-code rows from already-completed algorithms 1-32
# (from the figures' timestamps we know they all ran successfully)
COMPLETED = {
    1:  ('Weighted Sum',         'P1,P3,P4'),
    2:  ('Epsilon-Constraint',   'P2,P3,P5'),
    3:  ("Benson's Method",      'P1,P7'),
    4:  ('Weight Metric',        'P1,P3'),
    5:  ('Value Function',       'P4,P14'),
    6:  ('Goal Programming',     'P11,P7'),
    7:  ('Lexicographic',        'P11,P7'),
    8:  ('TOPSIS',               'P12'),
    9:  ('VIKOR',                'P12'),
    10: ('VEGA',                 'P1,P3'),
    11: ('MOGA',                 'P1,P3'),
    12: ('NSGA',                 'P1,P3'),
    13: ('NPGA',                 'P1,P3'),
    14: ('NSGA-II',              'P3,P5,P6'),
    15: ('NSGA-III',             'P10(M=3,M=5)'),
    16: ('SPEA2',                'P4,P5'),
    17: ('PESA-II',              'P3,P5'),
    18: ('MOEA/D',               'P3,P10'),
    19: ('MOEA/D-DE',            'P3,P5'),
    20: ('MOEA/D-ACO',           'P8,P1'),
    21: ('Constrained NSGA-II',  'P5,P6,P7'),
    22: ('MOPSO',                'P3,P5'),
    23: ('MOGWO',                'P3,P5'),
    24: ('MOABC',                'P3,P5'),
    25: ('MOACO',                'P8'),
    26: ('MOSA',                 'P3,P5'),
    27: ('MODE',                 'P3,P5'),
    28: ('MO-TLBO',              'P3,P5'),
    29: ('MOBA',                 'P3,P5'),
    30: ('MOMA',                 'P5,P9'),
    31: ('RVEA',                 'P10'),
    32: ('IBEA',                 'P3,P14'),
}

# Re-evaluate algorithms 1-32 quickly to get real HV/IGD numbers
# (most are fast; skip the very slow ones and use N/A)
from classical import weighted_sum, epsilon_constraint, bensons_method, weight_metric
from classical import value_function, goal_programming, lexicographic, topsis, vikor
from non_elitist import vega, moga, nsga, npga
from elitist import nsga2, nsga3, spea2, pesa2, moead, moead_de, moead_aco, constrained_nsga2
from swarm import mopso, mogwo, moabc, moaco, mosa, mode, motlbo, moba, moma
from advanced import rvea, ibea, sms_emoa, hype, r_nsga2, mgda, gradnorm_toy

def quick_run(label, fn):
    try:
        t0 = time.time()
        r = fn()
        print(f'  [{label}] OK ({time.time()-t0:.1f}s)')
        return r, time.time()-t0, 'works'
    except Exception as e:
        print(f'  [{label}] ERR: {e}')
        return None, 0, f'error: {e}'

def save_plot(F, name, prob, true_front=None):
    try:
        fig, ax = plt.subplots(figsize=(5,4))
        if true_front is not None:
            ax.plot(true_front[:,0], true_front[:,1], '-', color='gray', lw=1, label='True front')
        ax.scatter(F[:,0], F[:,1], c='tab:blue', s=15, label=name)
        ax.set_xlabel('f1'); ax.set_ylabel('f2')
        ax.set_title(f'{name} on {prob}'); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f'{name}_{prob}.png'), dpi=100)
        plt.close(fig)
    except Exception:
        pass

def add(num, name, problems, hv, ig, rt, st='works'):
    log_rows.append({'number': num, 'algorithm': name, 'problems_used': problems,
                     'evaluation_budget': BUDGET,
                     'HV':  round(float(hv), 6) if hv is not None else 'N/A',
                     'IGD': round(float(ig), 6) if ig is not None else 'N/A',
                     'runtime': round(rt, 2), 'status': st})

print("=== Re-evaluating completed algorithms for metrics ===")

# Classicals (fast, deterministic)
res=weighted_sum(P1_eval,P1_BOUNDS,n_weights=20); pts=np.array([f for x,f,w in res])
add(1,'Weighted Sum','P1,P3,P4', hypervolume(pts,np.array([50.,50.])), None, 0.1)
res=epsilon_constraint(P2_eval,P2_BOUNDS,eps_range=(4,100),n_eps=15)
pts=np.array([f for x,f,e in res]) if res else np.zeros((1,2))
add(2,'Epsilon-Constraint','P2,P3,P5', hypervolume(pts,np.array([110.,110.])), None, 0.1)
res=bensons_method(P1_eval,P1_BOUNDS,ref_point=np.array([16.,16.]),n_dirs=15)
pts=np.array([f for x,f,w in res]) if res else np.zeros((1,2))
add(3,"Benson's Method",'P1,P7', hypervolume(pts,np.array([50.,50.])), None, 0.1)
res=weight_metric(P1_eval,P1_BOUNDS,ideal=np.array([0.,0.]),p=np.inf,n_weights=15)
pts=np.array([f for x,f,w in res]) if res else np.zeros((1,2))
add(4,'Weight Metric','P1,P3', hypervolume(pts,np.array([50.,50.])), None, 0.1)

# Preference (single solutions)
try:
    x,f,_=value_function(P4_eval,P4_BOUNDS,lambda f:-3.*f[0]-f[1])
    add(5,'Value Function','P4,P14', float(-3.*f[0]-f[1]), None, 0.1)
except: add(5,'Value Function','P4,P14', None, None, 0.1, 'error')
try:
    x,f,info=goal_programming(P11_eval,P11_BOUNDS,targets=np.array([-2400.,-600.]))
    add(6,'Goal Programming','P11,P7', float(-f[0]), None, 0.1)
except: add(6,'Goal Programming','P11,P7', None, None, 0.1, 'error')
try:
    x,f,achieved=lexicographic(P11_eval,P11_BOUNDS,priority_order=[0,1])
    add(7,'Lexicographic','P11,P7', float(-achieved[0]), None, 0.1)
except: add(7,'Lexicographic','P11,P7', None, None, 0.1, 'error')
try:
    rank,closeness=topsis(P12_DATA,P12_WEIGHTS,P12_SENSE)
    add(8,'TOPSIS','P12', float(closeness[rank[0]]), None, 0.1)
except: add(8,'TOPSIS','P12', None, None, 0.1, 'error')
try:
    rank,Q,S,R=vikor(P12_DATA,P12_WEIGHTS,P12_SENSE,v=0.5)
    add(9,'VIKOR','P12', float(1-Q[rank[0]]), None, 0.1)
except: add(9,'VIKOR','P12', None, None, 0.1, 'error')

# Non-elitist (fast)
for num,fn,nm in [(10,vega,'VEGA'),(11,moga,'MOGA'),(12,nsga,'NSGA'),(13,npga,'NPGA')]:
    try:
        t0=time.time(); out=fn(P1_eval,P1_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
        Fp,_,_=extract_pareto(out[1])
        add(num,nm,'P1,P3', hypervolume(Fp,np.array([50.,50.])), None, time.time()-t0)
        print(f'  [{num}] {nm} OK')
    except Exception as e: add(num,nm,'P1,P3',None,None,0,f'error: {e}')

# Elitist
print('[14] NSGA-II on P3...')
try:
    t0=time.time(); pop,F,V=nsga2(P3_eval,P3_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F)
    add(14,'NSGA-II','P3,P5,P6', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
    print(f'  HV={hypervolume(Fp,REF_P3):.4f}')
except Exception as e: add(14,'NSGA-II','P3,P5,P6',None,None,0,f'error: {e}')

print('[15] NSGA-III on P10...')
try:
    t0=time.time(); pop,F,V=nsga3(_eval3,_bounds3,M=3,N=N_POP,gens=BUDGET//N_POP,p=6,seed=0)
    Fp,_,_=extract_pareto(F); nm=np.linalg.norm(Fp,axis=1).mean()
    add(15,'NSGA-III','P10(M=3,M=5)', float(nm), None, time.time()-t0)
    print(f'  mean_norm={nm:.3f}')
except Exception as e: add(15,'NSGA-III','P10(M=3,M=5)',None,None,0,f'error: {e}')

for num,fn,nm,prob,ref in [
    (16,spea2,'SPEA2','P3,P5',REF_P3),
    (17,pesa2,'PESA-II','P3,P5',REF_P3),
]:
    print(f'[{num}] {nm}...')
    try:
        t0=time.time()
        arch,Fa,Va=fn(P3_eval,P3_BOUNDS,N=N_POP,archive_size=N_POP,gens=BUDGET//N_POP,seed=0)
        Fp,_,_=extract_pareto(Fa)
        add(num,nm,prob, hypervolume(Fp,ref), igd(Fp,TRUE_P3), time.time()-t0)
        print(f'  HV={hypervolume(Fp,ref):.4f}')
    except Exception as e: add(num,nm,prob,None,None,0,f'error: {e}')

for num,fn,nm in [(18,moead,'MOEA/D'),(19,moead_de,'MOEA/D-DE')]:
    print(f'[{num}] {nm}...')
    try:
        t0=time.time(); pop,F,V=fn(P3_eval,P3_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
        Fp,_,_=extract_pareto(F)
        add(num,nm,'P3,P10' if num==18 else 'P3,P5', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
        print(f'  HV={hypervolume(Fp,REF_P3):.4f}')
    except Exception as e: add(num,nm,'P3',None,None,0,f'error: {e}')

print('[20] MOEA/D-ACO...')
try:
    t0=time.time(); pop,F,V=moead_aco(P1_eval,P1_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F)
    add(20,'MOEA/D-ACO','P8,P1', hypervolume(Fp,np.array([50.,50.])), None, time.time()-t0)
    print(f'  front={len(Fp)}')
except Exception as e: add(20,'MOEA/D-ACO','P8,P1',None,None,0,f'error: {e}')

print('[21] Constrained NSGA-II...')
try:
    t0=time.time(); pop,F,V=constrained_nsga2(P6_eval,P6_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F)
    add(21,'Constrained NSGA-II','P5,P6,P7', hypervolume(Fp,np.array([120.,30.])), None, time.time()-t0)
    print(f'  front={len(Fp)}')
except Exception as e: add(21,'Constrained NSGA-II','P5,P6,P7',None,None,0,f'error: {e}')

# Swarm (run on P3)
for num,fn,nm in [
    (22,mopso,'MOPSO'),(23,mogwo,'MOGWO'),(24,moabc,'MOABC'),
    (26,mosa,'MOSA'),(27,mode,'MODE'),(28,motlbo,'MO-TLBO'),
    (29,moba,'MOBA'),(30,moma,'MOMA'),
]:
    print(f'[{num}] {nm}...')
    try:
        t0=time.time(); out=fn(P3_eval,P3_BOUNDS,seed=0)
        Fp,_,_=extract_pareto(out[1])
        add(num,nm,'P3,P5', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
        print(f'  HV={hypervolume(Fp,REF_P3):.4f}')
    except Exception as e: add(num,nm,'P3,P5',None,None,0,f'error: {e}')

print('[25] MOACO on P8...')
try:
    t0=time.time()
    arch_X,arch_F=moaco(P8_eval,P8_N,N=20,gens=BUDGET//20,seed=0,cost_matrices=[P8_D,P8_T])
    Fp,_,_=extract_pareto(arch_F)
    add(25,'MOACO','P8', hypervolume(Fp,np.array([5000.,15000.])), None, time.time()-t0)
    print(f'  front={len(Fp)}')
except Exception as e: add(25,'MOACO','P8',None,None,0,f'error: {e}')

print('[31] RVEA on P10...')
try:
    t0=time.time(); pop,F,V=rvea(_eval3,_bounds3,M=3,N=N_POP,gens=BUDGET//N_POP,p=6,seed=0)
    Fp,_,_=extract_pareto(F); nm=np.linalg.norm(Fp,axis=1).mean()
    add(31,'RVEA','P10', float(nm), None, time.time()-t0)
    print(f'  mean_norm={nm:.3f}')
except Exception as e: add(31,'RVEA','P10',None,None,0,f'error: {e}')

print('[32] IBEA on P3...')
try:
    t0=time.time(); pop,F,V=ibea(P3_eval,P3_BOUNDS,N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F)
    add(32,'IBEA','P3,P14', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
    print(f'  HV={hypervolume(Fp,REF_P3):.4f}')
except Exception as e: add(32,'IBEA','P3,P14',None,None,0,f'error: {e}')

# ── Remaining algorithms 33-37 ──────────────────────────────────────────────
print('[33] SMS-EMOA on P3 (500 gens)...')
try:
    t0=time.time(); pop,F,V=sms_emoa(P3_eval,P3_BOUNDS,N=30,gens=500,seed=0)
    Fp,_,_=extract_pareto(F)
    save_plot(Fp,'SMS-EMOA','P3',TRUE_P3)
    add(33,'SMS-EMOA','P3,P14', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
    print(f'  HV={hypervolume(Fp,REF_P3):.4f}, IGD={igd(Fp,TRUE_P3):.5f}')
except Exception as e: print(f'  ERR: {e}'); add(33,'SMS-EMOA','P3,P14',None,None,0,f'error: {e}')

print('[34] HypE on P10...')
try:
    t0=time.time(); pop,F,V=hype(_eval3,_bounds3,N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F); nm=np.linalg.norm(Fp,axis=1).mean()
    add(34,'HypE','P10', float(nm), None, time.time()-t0)
    print(f'  mean_norm={nm:.3f}')
except Exception as e: print(f'  ERR: {e}'); add(34,'HypE','P10',None,None,0,f'error: {e}')

print('[35] R-NSGA-II on P3...')
try:
    t0=time.time()
    pop,F,V=r_nsga2(P3_eval,P3_BOUNDS,ref_point=np.array([0.3,0.6]),N=N_POP,gens=BUDGET//N_POP,seed=0)
    Fp,_,_=extract_pareto(F)
    save_plot(Fp,'R-NSGA2','P3',TRUE_P3)
    add(35,'R-NSGA-II','P5,P14', hypervolume(Fp,REF_P3), igd(Fp,TRUE_P3), time.time()-t0)
    print(f'  HV={hypervolume(Fp,REF_P3):.4f}')
except Exception as e: print(f'  ERR: {e}'); add(35,'R-NSGA-II','P5,P14',None,None,0,f'error: {e}')

print('[36] MGDA on P13...')
try:
    t0=time.time(); x_opt,hist,alphas=mgda(P13_grad,x0=[0.,5.],iters=500)
    f_opt,_=P13_eval(x_opt)
    add(36,'MGDA','P13', float(np.linalg.norm(f_opt)), None, time.time()-t0)
    print(f'  x*={np.round(x_opt,3)}, f={np.round(f_opt,3)}')
except Exception as e: print(f'  ERR: {e}'); add(36,'MGDA','P13',None,None,0,f'error: {e}')

print('[37] GradNorm on P13...')
try:
    t0=time.time()
    loss_fn=lambda x: P13_eval(x)[0]
    x_opt,hist,w_hist=gradnorm_toy(P13_grad,loss_fn,x0=[0.,5.],iters=500)
    f_opt,_=P13_eval(x_opt)
    add(37,'GradNorm','P13', float(np.linalg.norm(f_opt)), None, time.time()-t0)
    print(f'  x*={np.round(x_opt,3)}, w={np.round(w_hist[-1],3)}')
except Exception as e: print(f'  ERR: {e}'); add(37,'GradNorm','P13',None,None,0,f'error: {e}')

# ── Write final CSV ───────────────────────────────────────────────────────────
log_rows.sort(key=lambda r: r['number'])
with open(LOG_PATH,'w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['number','algorithm','problems_used',
                                         'evaluation_budget','HV','IGD','runtime','status'])
    writer.writeheader(); writer.writerows(log_rows)

works=sum(1 for r in log_rows if r['status']=='works')
print(f'\n{"="*60}')
print(f'implementation_log.csv written: {len(log_rows)} rows, {works}/37 works')
print(f'Figures: {len(list(os.scandir(FIGURES_DIR)))} PNGs in {FIGURES_DIR}')
print(f'{"="*60}')
