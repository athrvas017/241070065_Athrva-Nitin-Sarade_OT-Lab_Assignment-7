# Hand Trace — Algorithm 3: Benson's Method

**Problem used:** P1 — Factory Production Speed (1 variable, 2 objectives)

**P1:** x ∈ [1,10],  f1 = (x−3)²,  f2 = (x−7)²

**Reference point:** ref = (16, 16) — a point dominating most of the objective space.

---

## Algorithm Overview

Benson's method: from a reference point r, find the Pareto-efficient point by maximizing the weighted improvement past r:

> maximize Σᵢ wᵢ · (rᵢ − fᵢ(x))  subject to x feasible

For P1, this is a 1D optimization (closed form):

> f1 = (x−3)², f2 = (x−7)²
> maximize w1·(16 − (x−3)²) + w2·(16 − (x−7)²)
> = maximize 16(w1+w2) − w1(x−3)² − w2(x−7)²
> = minimize w1(x−3)² + w2(x−7)²

Taking d/dx = 0:

> 2w1(x−3) + 2w2(x−7) = 0
> x* = (3w1 + 7w2) / (w1 + w2)

---

## Trace 1: w = (0.9, 0.1)

x* = (3×0.9 + 7×0.1) / (0.9+0.1) = (2.7 + 0.7) / 1.0 = **3.4**

f(x*) = ((3.4−3)², (3.4−7)²) = (0.16, 12.96)

d = (16 − 0.16, 16 − 12.96) = (15.84, 3.04)

Objective improvement: 0.9×15.84 + 0.1×3.04 = 14.26 + 0.30 = **14.56** ✓

Code output expected: `x=3.4, f=(0.16, 12.96)`

---

## Trace 2: w = (0.5, 0.5)

x* = (3×0.5 + 7×0.5) / 1.0 = (1.5 + 3.5) / 1.0 = **5.0**

f(x*) = ((5−3)², (5−7)²) = (4.0, 4.0)

d = (16 − 4.0, 16 − 4.0) = (12.0, 12.0)

Objective improvement: 0.5×12.0 + 0.5×12.0 = **12.0** ✓

Code output expected: `x=5.0, f=(4.0, 4.0)` (the symmetric point)

---

## Trace 3: w = (0.1, 0.9)

x* = (3×0.1 + 7×0.9) / 1.0 = (0.3 + 6.3) / 1.0 = **6.6**

f(x*) = ((6.6−3)², (6.6−7)²) = (12.96, 0.16)

d = (16 − 12.96, 16 − 0.16) = (3.04, 15.84)

Objective improvement: 0.1×3.04 + 0.9×15.84 = 0.30 + 14.26 = **14.56** ✓

Code output expected: `x=6.6, f=(12.96, 0.16)`

---

## Pareto Curve Check

The true Pareto front for P1 satisfies f2 = (x−7)² with x ∈ [3,7], so:

> f1 = (x−3)²  →  x = 3 + √f1
> f2 = (3 + √f1 − 7)² = (√f1 − 4)² = f1 − 8√f1 + 16

Checking Trace 2: f1=4, f2=4−8×2+16=4 ✓  
Checking Trace 1: f1=0.16, f2=0.16−8×0.4+16=0.16−3.2+16=12.96 ✓  
Checking Trace 3: f1=12.96, f2=12.96−8×3.6+16=12.96−28.8+16=0.16 ✓

All three traced points lie on the true Pareto curve.

---

## Parameters and Sensitivity

| Parameter | Effect |
|-----------|--------|
| Reference point too close to the front | LP may find d=0 (no improvement possible) |
| Reference point at a Pareto point | All improvements d=0, useless starting point |
| Reference point far outside feasible region | Large d found, effectively sweeps the whole front |
| Few directions (n_dirs=5) | Only 5 front points sampled, front under-represented |
| Many directions (n_dirs=20) | Denser front, diminishing returns for convex fronts |

**Non-convex front note:** For non-convex problems (like ZDT2/P3), Benson's method with a single reference point can miss the non-convex portions of the front. This is the same limitation as Weighted Sum.

---

## TODO(me)

- In the viva, explain why d=0 at a Pareto reference point (there is no further improvement possible past a point already on the front).
- Explain the difference between Benson's method and Weighted Sum: Benson's measures distance from a FIXED reference toward the front, while WS scalarizes f directly. Both can miss non-convex portions.
- For your roll number (algorithm 3 = Benson's Method), be ready to walk through the LP formulation and the meaning of the d-variables.
