# Results

Numerical outputs only. No interpretation.

---

## Phase 4 — Analytic Results

**Exact result:**
```
D' = F · p' · q'      (no approximation)
R_idx = F · p_eq · (1 − p_eq)
```

**Jacobian rank:** 1 at all tested (p, F, s, h) combinations  
**det(J):** 0.0 (machine precision) at all tested points

**Unit test results:** 5/5 pass

**R_idx properties verified:**

| Property | Condition | Result |
|---|---|---|
| R_idx = 0 | F = 0, any (s, h, p) | Confirmed |
| R_idx = F·p₀·q₀ | s = 0 (no selection) | Confirmed |
| R_idx stable, nonzero | h < 0, balancing | Confirmed |
| R_idx → 0 | h > 0, directional, t → ∞ | Confirmed |
| R_idx linear in F | fixed (s, h, p_eq) | Confirmed |

---

## Phase 1 — Clustering Results

| Dataset | Optimal k | F_ST (prior for Phase 3) |
|---|---|---|
| HbC | 3 | 0.021 |
| CCR5-Δ32 (46 pops) | 8 | 0.031 |
| Drosophila Adh | 2 | 0.038 |
| Gadus Syp I | 2 | 0.019 |
| Zoarces Est III | 2 | 0.014 |
| Mc1r | 2 | 0.044 |
| Panaxia medionigra | 2 | 0.027 |

CLR transform: g = (P·Q·R)^(1/3), clr = (ln(P/g), ln(Q/g), ln(R/g))  
Jeffreys pseudocount: 0.5 applied to zero cells before transform  
Optimal k: selected by Calinski-Harabasz index

---

## Phase 3 — MCMC Posterior Results

**Sampler:** Metropolis-Hastings, adaptive Gaussian proposals, target acceptance 0.234  
**Chains:** 4 independent chains with jittered starts  
**Steps:** 20,000 per chain  
**Burn-in:** 5,000 discarded  
**Parameters sampled in:** logit space (p, F) or (p₀, F, s, h)

### Cross-sectional datasets (free params: p, F)

| Dataset | F (mean±sd) | p (mean±sd) | R_idx (mean) | 90% HDI | R-hat (max) |
|---|---|---|---|---|---|
| Modiano HbC | 0.012 ± 0.008 | 0.119 ± 0.009 | 0.00139 | [0.00024, 0.00320] | 1.052 |
| CCR5-Δ32 (Britain) | 0.028 ± 0.018 | 0.067 ± 0.011 | 0.00274 | [0.00060, 0.00631] | 1.001 |
| Zoarces Est III | 0.018 ± 0.012 | 0.284 ± 0.021 | 0.00371 | [0.00041, 0.00890] | 1.023 |

### Temporal datasets (free params: p₀, F, s, h)

| Dataset | p₀ (mean±sd) | F (mean±sd) | s (mean±sd) | h (mean±sd) | R_idx t=0 (mean) | 90% HDI | R-hat (max) |
|---|---|---|---|---|---|---|---|
| Drosophila Adh | 0.289 ± 0.018 | 0.040 ± 0.016 | 0.148 ± 0.008 | 0.640 ± 0.078 | 0.00834 | [0.00369, 0.01433] | 1.164 |
| Panaxia medionigra | — | — | — | — | — | — | — |

*Panaxia results: run phase3_mcmc.py with TEMPORAL_DATASET = 7 to populate.*

### Identifiability probe (synthetic data)

**Cross-sectional probe:** N=500, true p=0.40, true F=0.10  
- p recovered: 0.400 ± 0.016 (true value inside 90% HDI)  
- F recovered: 0.073 ± 0.031 (true value inside 90% HDI)

**Temporal probe:** 12 timepoints, true (p₀=0.30, F=0.20, s=0.30, h=0.50)  
- p₀ recovered: 0.302 ± 0.019 ✓  
- F recovered: 0.188 ± 0.041 ✓  
- s recovered: 0.264 ± 0.038 (partial; true=0.300)  
- h recovered: 0.668 ± 0.091 (missed; true=0.500) — likelihood surface flat once p > 0.95

---

## Phase 2 — Neural ODE Results

**Architecture:** MLP, 2 hidden layers, 32 units, tanh activation  
**Boundary enforcement:** output multiplied by p·(1−p)  
**Ensemble size:** 10 independently initialised networks  
**Uncertainty:** 90% spread across ensemble

**Physics loss anchored to Phase 3 posterior means:**  
s = 0.148, h = 0.640 (Drosophila Adh)  
*(Update with Panaxia posterior means after Phase 3 Panaxia run)*

**Trajectory agreement (Drosophila Adh):**

| Metric | Value |
|---|---|
| Mean \|p_NODE − p_WF\| across 12 timepoints | 0.0058 |
| Max \|p_NODE − p_WF\| | 0.011 |
| R_idx at generation 0 | 0.00940 |
| R_idx at generation 57 (% of peak) | ~7% |
| R_idx peak generation | ~10 |
