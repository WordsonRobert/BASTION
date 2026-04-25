# Phase 1 — Aitchison Geometry Clustering

**Script:** `phase1_aitchison.py`  
**Runtime:** ~30 seconds  
**Requires:** `hwe_datasets.csv`  
**Outputs:** `phase1_fst_priors.csv` (used by Phase 3)

---

## What it does

1. Loads all datasets from `hwe_datasets.csv`.
2. Applies Centred Log-Ratio (CLR) transform to (P, Q, R) genotype frequencies.
3. Runs K-means clustering on CLR-transformed data.
4. Selects optimal k using Calinski-Harabasz index.
5. Estimates F_ST per dataset from cluster allele frequency variance.
6. Saves F_ST values as priors for Phase 3 MCMC.

## Why CLR and not Euclidean

Genotype frequencies (P, Q, R) sum to 1 — they live on a 2-simplex. Euclidean distance distorts the geometry on the simplex. The CLR transform maps compositional data to an unconstrained Euclidean space where standard clustering is valid.

## CLR formula

```
g = (P * Q * R)^(1/3)                    # geometric mean
clr(P, Q, R) = (ln(P/g), ln(Q/g), ln(R/g))
```

Zero cells receive Jeffreys pseudocount 0.5 before transform.

## F_ST formula

```
F_ST = Var(p_k) / (p_bar * (1 - p_bar))
```

where p_k is the allele frequency of cluster k and p_bar is the global mean.

## Outputs

```
phase1_fst_priors.csv         — F_ST per dataset (used as MCMC priors in Phase 3)
figures/phase1/
  clusters_{dataset}.png      — CLR scatter plot coloured by cluster assignment
```

## Changing the pseudocount

Edit `PSEUDOCOUNT = 0.5` at the top of the script. 0.5 is the Jeffreys (uninformative) prior for compositional data.
