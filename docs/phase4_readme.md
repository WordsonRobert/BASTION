# Phase 4 — Robustness Index Derivation

**Script:** `phase4_robustness.py`  
**Runtime:** ~5 seconds  
**Run first. Required before all other phases.**

---

## What it does

1. Implements the Wright-Fisher recursion with viability selection and inbreeding coefficient F.
2. Proves numerically that the Jacobian of the WF map T:(P,Q)→(P',Q') always has rank 1 and det(J)=0.
3. Verifies the exact result D' = F·p'·q' (deviation from HWE after one generation).
4. Finds the equilibrium allele frequency p_eq by iterating the recursion to convergence (|Δp| < 10⁻¹²).
5. Defines R_idx = F · p_eq · (1 − p_eq).
6. Runs 5 unit tests verifying the analytic properties of R_idx.
7. Generates heatmaps and trajectory plots.

## Key functions

| Function | What it does |
|---|---|
| `wf_step(p, F, s, h)` | One WF generation. Returns (p', D') where D' = F·p'·q'. |
| `find_equilibrium(p0, F, s, h)` | Iterates wf_step until convergence. Returns p_eq. |
| `R_idx(p0, F, s, h)` | Returns F · p_eq · (1−p_eq). |
| `numerical_jacobian(p0, F, s, h)` | 2×2 numerical Jacobian of T:(P,Q)→(P',Q'). |
| `run_unit_tests()` | Runs 5 tests. Must all pass before proceeding. |

## Outputs

```
figures/phase4/
  heatmaps_ridx.png       — R_idx over (p0, F) space, three selection regimes
  trajectories_ridx.png   — R_idx(t) for balancing vs directional selection
  Fscaling_ridx.png       — R_idx vs F at fixed p_eq, several selection types
```

## Parameters

| Parameter | Description |
|---|---|
| F | Inbreeding coefficient. P(IBD). Range [0, 1]. F=0: random mating. F=1: complete selfing. |
| s | Selection coefficient. Fitness reduction of disfavoured homozygote (aa). Range (0, 1). |
| h | Dominance coefficient. h<0: overdominance (balancing). h>0: directional. h=0: additive. |
| p | Allele frequency of A. |
| p_eq | Equilibrium p under given (s, h). |

## Fitness scheme

```
w_AA = 1
w_Aa = 1 - h*s
w_aa = 1 - s
```
