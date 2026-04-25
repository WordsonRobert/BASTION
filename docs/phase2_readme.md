# Phase 2 — Physics-Informed Neural ODE

**Script:** `phase2_neural_ode.py`  
**Runtime:** 8–10 minutes (CPU)  
**Requires:** `phase3_temporal_posterior.npy`  
**Run last.**

---

## What it does

Trains a neural ODE to produce a continuous R_idx(t) trajectory from discrete observed timepoints.

The MLP learns the vector field `dp/dt = f_theta(p)`. The output is multiplied by `p*(1-p)` to enforce `dp/dt = 0` at fixation boundaries (p=0 or p=1) for any parameter theta.

An ensemble of 10 independently initialised networks is trained. The 90% spread is reported as the uncertainty band.

## Loss function

```
L = L_data + lambda * L_phys

L_data  = binomial NLL at observed timepoints
L_phys  = MSE( f_theta(p), WF_increment(p; s_hat, h_hat) )
          evaluated at 150 collocation points p in (0, 1)
```

`s_hat` and `h_hat` are the posterior means from Phase 3.

## Architecture

```
Input: p (scalar)
→ Linear(1, 32) → Tanh
→ Linear(32, 32) → Tanh
→ Linear(32, 1)
× p * (1-p)          ← boundary enforcement
= dp/dt
```

## Key settings

| Setting | Value | Location in script |
|---|---|---|
| Ensemble size | 10 | `ENSEMBLE_SIZE` |
| Epochs | 800 | `N_EPOCHS` |
| Learning rate | 0.001 | `LR` |
| Physics loss weight | 0.5 | `LAMBDA_PHYS` |
| Collocation points | 150 | `N_COLLOC` |
| ODE solver | RK4 | `SOLVER` |

## Outputs

```
figures/phase2/
  neural_ode_results.png      — 3-panel: p(t) trajectory, R_idx(t), pointwise residuals
  ensemble_uncertainty.png    — All 10 network trajectories + observed data
```

## Note on the model

The NODE is a smooth interpolant of the WF dynamics, not a mechanistic model. It can only be interpreted within the range of the training data. Extrapolation beyond the observed timepoints is not reliable.
