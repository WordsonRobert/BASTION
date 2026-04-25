# HWE Robustness Pipeline

**BASTIAN — Bayesian Analysis of Selection, Transmission, and Inbreeding via Observable Networks**  
IISER Kolkata · Evolutionary Biology · April 2026  
Authors: Wordson Robert, Gautam Menon, Sahil Singh, Amogh Shenoy

---

## What this does

Derives and empirically validates a scalar robustness index for Hardy-Weinberg Equilibrium:

```
R_idx = F · p_eq · (1 − p_eq)
```

where `F` is the inbreeding coefficient and `p_eq` is the equilibrium allele frequency under selection.

---

## Run order

**Must follow this order. Each phase depends on the previous.**

```
Phase 4 → Phase 1 → Phase 3 → Phase 2
```

```bash
# 1. Setup
python -m venv hwe_pipeline
source hwe_pipeline/bin/activate       # Windows: hwe_pipeline\Scripts\activate
pip install -r requirements.txt

# 2. Run phases in order
python phase4_robustness.py
python phase1_aitchison.py
python phase3_mcmc.py
python phase2_neural_ode.py
```

All scripts must be run from this directory. `hwe_datasets.csv` must be present here.

---

## Output

All figures are saved to `figures/phaseN/`. See `results.md` for numerical results.  
Add your own figure exports to `figures/` subfolders — see `figures/PLACE_FIGURES_HERE.md`.  
Add reference PDFs to `papers/` — see `papers/PLACE_PAPERS_HERE.md`.

---

## Phase summaries

| Phase | Script | Runtime | What it does |
|---|---|---|---|
| 4 | `phase4_robustness.py` | ~5 sec | Derives R_idx analytically, proves rank-1 Jacobian, runs unit tests, generates heatmaps |
| 1 | `phase1_aitchison.py` | ~30 sec | CLR clustering on genotype frequencies, F_ST priors |
| 3 | `phase3_mcmc.py` | ~5–10 min | Bayesian MCMC inference of (p, F, s, h), posterior R_idx |
| 2 | `phase2_neural_ode.py` | ~8–10 min | Physics-informed Neural ODE, continuous R_idx(t) trajectory |

Full per-phase documentation in `docs/`.

---

## Requirements

Python 3.9+. See `requirements.txt`. CUDA not required for Phase 2 (CPU sufficient).
