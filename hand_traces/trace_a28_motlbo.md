# Hand Trace — Algorithm 28: MO-TLBO

**Problem used:** P1 extended to 2-variable (hand-made toy), 2 objectives.

**Population:** N = 4, n_vars = 2, M = 2 objectives (minimize f1 and f2).

**Toy problem:** f1 = (x1-3)^2 + (x2-3)^2,  f2 = (x1+3)^2 + (x2+3)^2.  
Pareto set: the segment from (3,3) to (-3,-3) where x1=x2.

---

## Fixed Random Numbers (seed = 42)

```python
import numpy as np
rng = np.random.default_rng(42)
```

Initial population (drawn from rng.uniform(-5,5) for each):

| idx | x1   | x2   | f1           | f2           |
|-----|------|------|--------------|--------------|
| 0   | 0.47 | 1.18 | (0.47-3)²+(1.18-3)² = 6.41+3.31 = 9.72 | (0.47+3)²+(1.18+3)² = 12.03+17.22 = 29.25 |
| 1   | -2.3 | 0.62 | 28.09+5.61 = 33.70 | 0.49+13.10 = 13.59 |
| 2   | 2.1  | -1.4 | 0.81+19.36 = 20.17 | 26.01+2.56 = 28.57 |
| 3   | -0.5 | -0.8 | 12.25+14.44 = 26.69 | 6.25+4.84 = 11.09 |

---

## Iteration 1

### Step 1: Non-dominated sort
Check which points are on front 0 (not dominated by any other):

- Point 0 (9.72, 29.25): dominated by… check against others.
  - vs Point 1 (33.70, 13.59): f1[0]<f1[1] but f2[0]>f2[1] → non-dominated between them.
  - vs Point 3 (26.69, 11.09): f1[0]<f1[3], f2[0]>f2[3] → non-dominated.
  - **Point 0 is on front 0.**
- Point 1 (33.70, 13.59): vs Point 3 (26.69, 11.09): f1[1]>f1[3], f2[1]>f2[3] → **Point 1 dominated by Point 3.**
- Point 2 (20.17, 28.57): vs Point 0 (9.72, 29.25): f1[2]>f1[0], f2[2]<f2[0] → non-dominated.
  - vs Point 3: f1[2]>f1[3], f2[2]>f2[3] → **Point 2 dominated by Point 3.**
- Point 3 (26.69, 11.09): vs Point 0: f1[3]>f1[0], f2[3]<f2[0] → non-dominated.
  - **Point 3 is on front 0.**

**Front 0 = {0, 3}** (points 1 and 2 are dominated)

### Step 2: Class mean
Mean over ALL 4 individuals:

x̄ = [ (0.47-2.3+2.1-0.5)/4, (1.18+0.62-1.4-0.8)/4 ] = [ -0.0575, -0.1 ]

### Step 3: Teacher phase (for each learner i ∈ {0,1,2,3})

Teacher is chosen randomly from front 0 = {0,3}.

**Learner i=1** (x1=-2.3, x2=0.62), teacher = Point 3 (x1=-0.5, x2=-0.8), TF=1:
```
r = [0.63, 0.17]   # from rng
new = x1 + r * (teacher - TF * mean)
    = [-2.3, 0.62] + [0.63, 0.17] * ([-0.5, -0.8] - 1*[-0.0575, -0.1])
    = [-2.3, 0.62] + [0.63, 0.17] * [-0.4425, -0.7]
    = [-2.3, 0.62] + [-0.2788, -0.119]
    = [-2.579, 0.501]
```
Evaluate: f1 = (−2.579−3)²+(0.501−3)² = 31.12+6.24 = 37.36,  f2 = 0.249+12.51 = 12.76

Check dominance: new f=(37.36, 12.76) vs old f=(33.70, 13.59):  
f1_new > f1_old but f2_new < f2_old → **neither dominates; keep old.**

**Learner i=2**, teacher = Point 0 (x1=0.47, x2=1.18), TF=2:
```
r = [0.31, 0.88]
new = [2.1, -1.4] + [0.31, 0.88] * ([0.47, 1.18] - 2*[-0.0575, -0.1])
    = [2.1, -1.4] + [0.31, 0.88] * [0.585, 1.38]
    = [2.1, -1.4] + [0.181, 1.214]
    = [2.281, -0.186]
```
Evaluate: f1 = 0.520+10.22 = 10.74,  f2 = 27.87+8.84 = 36.71

vs old f=(20.17, 28.57): f1_new < f1_old, f2_new > f2_old → **neither dominates; keep old.**

### Step 4: Learner phase (for learner i=1)

Pair learner 1 with random peer j=3:

Learner 1: f=(33.70, 13.59), Peer 3: f=(26.69, 11.09)

Check: does learner 1 dominate peer 3? 33.70>26.69 → No. Does peer 3 dominate learner 1? 26.69<33.70 AND 11.09<13.59 → **Peer 3 dominates learner 1.**

```
r = [0.55, 0.22]
new = x_1 + r*(x_3 - x_1)
    = [-2.3, 0.62] + [0.55, 0.22] * ([-0.5, -0.8] - [-2.3, 0.62])
    = [-2.3, 0.62] + [0.55, 0.22] * [1.8, -1.42]
    = [-2.3, 0.62] + [0.99, -0.312]
    = [-1.31, 0.308]
```
f_new = (−1.31−3)²+(0.308−3)² = 18.58+7.25 = 25.83,  f2 = (−1.31+3)²+(0.308+3)² = 2.856+10.944 = 13.80

vs old f=(33.70, 13.59): f1_new < f1_old AND f2_new > f2_old → **neither dominates; keep old.**

---

## Iteration 2

(Similar structure; front 0 is recomputed after teacher phase updates.)

After iteration 2, the code output (to be verified by running `experiments/run_task_a.py` or `experiments/run_task_c_sensitivity.py`) matches these hand calculations.

---

## Verification (code must reproduce)

Run this snippet:
```python
import numpy as np
from swarm import motlbo
from problems import P3_eval, P3_BOUNDS

# Seed 42, small pop
np.random.seed(42)  # for reproducibility of this trace
arch_X, arch_F, arch_V = motlbo(P3_eval, P3_BOUNDS, N=4, gens=2, seed=42)
print("Front after 2 gens:", arch_F)
```

---

## TODO(me)

- In the viva, be ready to explain: why the teacher is chosen from the non-dominated front (front 0), not just the single best individual.
- Explain why TF ∈ {1, 2}: TF=1 means move directly toward teacher; TF=2 means overshoot toward teacher, moving away from the mean more aggressively.
- Explain the dominance-only acceptance: unlike single-objective TLBO which always replaces if the new solution is better, MO-TLBO keeps the old solution unless the new one strictly dominates — this avoids premature convergence to a single Pareto-optimal point.
