# Phase 3 — Bayesian MCMC Inference

**Script:** `phase3_mcmc.py`  
**Runtime:** 5–10 minutes  
**Requires:** `hwe_datasets.csv`, `phase1_fst_priors.csv`  
**Outputs:** `phase3_results.csv`, `phase3_temporal_posterior.npy` (used by Phase 2)

---

## What it does

Runs Metropolis-Hastings MCMC to infer parameters from observed genotype counts.

**Cross-sectional model** (single timepoint):  
Free parameters: `(p, F)`  
Likelihood: multinomial on `(N_AA, N_Aa, N_aa)`  
Used for: HbC (ID=1), Zoarces (ID=5), CCR5 (ID=6)

**Temporal model** (multiple timepoints):  
Free parameters: `(p0, F, s, h)`  
Likelihood: sum of multinomials across timepoints, with p(t) from WF recursion  
Used for: Panaxia (ID=7) — or Drosophila (ID=3, flagged)

## Sampler details

| Setting | Value |
|---|---|
| Chains | 4 independent, jittered starts |
| Steps per chain | 20,000 |
| Burn-in | 5,000 (discarded) |
| Proposal | Adaptive Gaussian, target acceptance 0.234 |
| Parameter space | Logit-transformed (handles [0,1] bounds) |
| Convergence | Gelman-Rubin R-hat (target < 1.1) |

## Priors

| Parameter | Prior | Range |
|---|---|---|
| p | Uniform | (0, 1) |
| F | Uniform | (0, 0.8) — informed by Phase 1 F_ST |
| s | Uniform | (0, 1) |
| h | Uniform | (−2, 2) |

## Changing the temporal dataset

At the top of `phase3_mcmc.py`:
```python
TEMPORAL_DATASET = 7   # 7 = Panaxia (recommended), 3 = Drosophila (flagged)
```

## Outputs

```
phase3_results.csv                — Posterior means, SDs, R_idx HDI, R-hat per dataset
phase3_temporal_posterior.npy     — Posterior means for Phase 2 (s, h, F, p0)
figures/phase3/
  mcmc_{dataset}.png              — Trace plots and marginal posteriors
  ridx_{dataset}.png              — Posterior R_idx distribution with 90% HDI
```

## R-hat interpretation

| R-hat | Status |
|---|---|
| < 1.05 | Good convergence |
| 1.05–1.10 | Acceptable |
| > 1.10 | Poor convergence — run longer chains or check model |
| > 1.2 | Do not use results |

## Known issue: Drosophila Adh (ID=3)

When `TEMPORAL_DATASET = 3`, the F posterior is prior-dominated. This is because the Drosophila counts were back-calculated from allele frequencies assuming HWE, so the data carry no genuine deviation signal for F. Use ID=7 (Panaxia) instead. See `datasets.md`.
