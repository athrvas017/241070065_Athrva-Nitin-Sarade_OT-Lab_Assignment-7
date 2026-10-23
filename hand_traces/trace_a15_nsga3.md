# Hand Trace — Algorithm 15: NSGA-III

**Problem:** DTLZ2 (P10), M=3 objectives, p=2 (so 6 reference points).

**Population:** 4 parents + 4 offspring (toy for tracing), M=3, p=2.

---

## Step 1: Das–Dennis Reference Points (M=3, p=2)

All points (k1, k2, k3) with k1+k2+k3=p=2, divided by p:

| Point | (k1,k2,k3) | Normalized (w1,w2,w3) |
|-------|-----------|----------------------|
| r0    | (2,0,0)   | (1.0, 0.0, 0.0)      |
| r1    | (1,1,0)   | (0.5, 0.5, 0.0)      |
| r2    | (1,0,1)   | (0.5, 0.0, 0.5)      |
| r3    | (0,2,0)   | (0.0, 1.0, 0.0)      |
| r4    | (0,1,1)   | (0.0, 0.5, 0.5)      |
| r5    | (0,0,2)   | (0.0, 0.0, 1.0)      |

---

## Step 2: Toy Population (after non-dominated sort)

Assume we have 8 combined (parent+offspring) points:

| idx | f1   | f2   | f3   | ||f||  | Notes |
|-----|------|------|------|--------|-------|
| 0   | 0.80 | 0.50 | 0.33 | 0.995  | near true front |
| 1   | 0.60 | 0.70 | 0.39 | 0.999  | near true front |
| 2   | 0.40 | 0.80 | 0.45 | 0.997  | near true front |
| 3   | 0.10 | 0.10 | 0.99 | 1.003  | near true front |
| 4   | 1.20 | 0.70 | 0.50 | 1.456  | dominated       |
| 5   | 0.70 | 0.90 | 0.60 | 1.304  | dominated       |
| 6   | 0.50 | 0.60 | 0.70 | 1.010  | borderline      |
| 7   | 0.30 | 0.40 | 0.90 | 1.037  | borderline      |

Front 0 = {0,1,2,3} (||f||≈1, near the DTLZ2 true front sphere).  
Front 1 = {6,7}  
Front 2 = {4,5}

N=4, so we take all of front 0 = {0,1,2,3}. **No truncation needed in this toy.**

---

## Step 3: Normalization

Ideal point (min per objective across all 8):
- ideal = (0.10, 0.10, 0.33)

Translate: F' = F - ideal:
- 0': (0.70, 0.40, 0.00)
- 1': (0.50, 0.60, 0.06)
- 2': (0.30, 0.70, 0.12)
- 3': (0.00, 0.00, 0.66)

Extreme points (max per objective in F'):
- scale = (0.70, 0.70, 0.66)

Normalize: F'' = F' / scale:
- 0'': (1.00, 0.57, 0.00)
- 1'': (0.71, 0.86, 0.09)
- 2'': (0.43, 1.00, 0.18)
- 3'': (0.00, 0.00, 1.00)

---

## Step 4: Associate Each Point to Nearest Reference Line

For each point, compute perpendicular distance to each unit reference vector:

**Point 0'' = (1.00, 0.57, 0.00)**:
- r0 = (1,0,0): proj = 1.00×1+0.57×0+0.00×0 = 1.00; d_perp = ||(1.00,0.57,0.00) − 1.00×(1,0,0)|| = ||(0,0.57,0)|| = **0.57**
- r1 = (0.707,0.707,0): proj = 0.707×1.00+0.707×0.57 = 1.110; d_perp = ||(1.00,0.57,0)−1.11×(0.707,0.707,0)|| = ||(0.215,−0.214,0)|| = **0.303**
- Best: **r1** (d_perp=0.303)

**Point 3'' = (0.00, 0.00, 1.00)**:
- r5 = (0,0,1): proj = 1.00; d_perp = ||(0,0,1)−1×(0,0,1)|| = **0.00**
- Best: **r5** (d_perp=0.00)

---

## Step 5: Niche Counts

After associating all front-0 members {0,1,2,3}:

| Reference | Members in selected | Niche count |
|-----------|--------------------| ------------|
| r0        | –                  | 0           |
| r1        | 0                  | 1           |
| r2        | 2                  | 1           |
| r3        | –                  | 0           |
| r4        | 1                  | 1           |
| r5        | 3                  | 1           |

(Exact assignments depend on perpendicular distances — this table is illustrative.)

---

## Step 6: Fill from Last Front (if needed)

In this toy example, front 0 already fills N=4 exactly. In a real run with a partial last front, the algorithm:

1. Finds references with niche_count = 0 (or minimum).
2. Picks the candidate in the last front associated to that reference.
   - If niche_count=0: picks the candidate CLOSEST to the reference line.
   - If niche_count>0: picks RANDOMLY among candidates.
3. Adds that candidate to the selected set; increments that reference's niche count.
4. Repeats until N survivors are chosen.

---

## Verification (code must reproduce)

```python
from elitist import nsga3, das_dennis_ref_points
ref_pts = das_dennis_ref_points(3, 2)
print("Reference points (M=3, p=2):", ref_pts)
# Expected: 6 points as listed above
```

---

## Parameters and Sensitivity

| Parameter | Effect |
|-----------|--------|
| p=6 (M=3) | 28 reference points, good spread |
| p=12 (M=3) | 91 reference points, very dense |
| p=16 (M=3) | 153 reference points, may exceed pop size |
| Too few ref points | Poor coverage of the objective space simplex |
| Too many ref points (>>N) | Many references have no members; selection degrades |

**Key insight:** NSGA-III replaces crowding distance with reference-point niching. Crowding distance is computed in objective space (a single distance) which becomes ineffective for M>=4 because most points are equally non-dominated and equidistant. Reference-point niching preserves spread along STRUCTURED directions.

---

## TODO(me)

- Explain in the viva: why crowding distance fails for M>=4 (briefly: the fraction of the population that is non-dominated approaches 100%, so almost no selection pressure remains).
- Explain the Das–Dennis structure: the number of reference points is C(M+p−1, p). For M=3, p=6 gives C(8,6)=28 points.
- Walk through at least one full association calculation (perpendicular distance = ||f_normalized − (f_normalized·r_hat)×r_hat||).
