"""
Phase 3 — Bayesian MCMC Inference
====================================
Pure NumPy Metropolis-Hastings sampler.
Infers (p, F) from cross-sectional data and (p0, F, s, h) from temporal data.
Parameters sampled in logit/log space to handle boundary constraints.
4 independent chains, 20000 steps each, adaptive Gaussian proposals.
Gelman-Rubin R-hat for convergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
import os, warnings

warnings.filterwarnings("ignore")

OUT = "figures/phase3"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# ─── Settings ───────────────────────────────────────────────────────────────

N_STEPS      = 20000
N_BURNIN     = 5000
N_CHAINS     = 4
TARGET_RATE  = 0.234   # optimal acceptance for Gaussian proposals
ADAPT_EVERY  = 100
ADAPT_SCALE  = 0.1     # initial proposal scale
PSEUDOCOUNT  = 0.5

# Select which datasets to run MCMC on:
# Cross-sectional: 1 (HbC), 5 (Zoarces), 6 (CCR5)
# Temporal: 7 (Panaxia) or 3 (Drosophila, flagged)
CROSS_SECTIONAL_IDS = [1, 5, 6]
TEMPORAL_DATASET    = 7   # set to 3 for Drosophila (flagged)


# ─── Helpers ────────────────────────────────────────────────────────────────

def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_prior_p(lp):
    """Uniform prior on p in (0,1) — flat in logit space gives logistic correction."""
    p = sigmoid(lp)
    return np.log(p) + np.log(1 - p)  # Jacobian of logit transform

def log_prior_F(lF):
    """Uniform prior on F in (0, 0.8)."""
    F = sigmoid(lF) * 0.8
    if F < 0 or F > 0.8:
        return -np.inf
    return np.log(sigmoid(lF)) + np.log(1 - sigmoid(lF)) + np.log(0.8)

def log_prior_s(ls):
    """Uniform prior on s in (0, 1)."""
    s = sigmoid(ls)
    return np.log(s) + np.log(1 - s)

def log_prior_h(lh):
    """Uniform prior on h in (-2, 2)."""
    h = sigmoid(lh) * 4.0 - 2.0
    return np.log(sigmoid(lh)) + np.log(1 - sigmoid(lh)) + np.log(4.0)

def log_multinomial(n_AA, n_Aa, n_aa, p, F):
    """Log-likelihood of genotype counts under inbreeding model."""
    q = 1.0 - p
    F = np.clip(F, 0.0, 1.0)
    prob_AA = p**2 + F * p * q
    prob_Aa = 2.0 * p * q * (1.0 - F)
    prob_aa = q**2 + F * p * q
    probs = np.array([prob_AA, prob_Aa, prob_aa])
    probs = np.clip(probs, 1e-15, 1.0)
    n = np.array([n_AA, n_Aa, n_aa])
    N = n.sum()
    ll = gammaln(N + 1) - np.sum(gammaln(n + 1)) + np.sum(n * np.log(probs))
    return ll


# ─── WF recursion for temporal likelihood ───────────────────────────────────

def wf_step(p, F, s, h):
    q = 1.0 - p
    P = p**2 + F * p * q
    Q = 2.0 * p * q * (1.0 - F)
    R = q**2 + F * p * q
    w_AA, w_Aa, w_aa = 1.0, 1.0 - h * s, 1.0 - s
    w_bar = P * w_AA + Q * w_Aa + R * w_aa
    if w_bar < 1e-10:
        return p
    p_prime = (P * w_AA + 0.5 * Q * w_Aa) / w_bar
    return float(np.clip(p_prime, 1e-6, 1 - 1e-6))

def run_wf(p0, F, s, h, generations):
    """Run WF recursion for given number of generations."""
    traj = [p0]
    p = p0
    for _ in range(int(max(generations))):
        p = wf_step(p, F, s, h)
        traj.append(p)
    return np.array(traj)


# ─── Log-likelihoods ────────────────────────────────────────────────────────

def log_likelihood_crosssectional(theta_logit, data):
    """data: list of (N_AA, N_Aa, N_aa) rows — pooled cross-sectional."""
    lp, lF = theta_logit
    p = sigmoid(lp)
    F = sigmoid(lF) * 0.8
    ll = 0.0
    for n_AA, n_Aa, n_aa in data:
        ll += log_multinomial(n_AA, n_Aa, n_aa, p, F)
    return ll

def log_posterior_crosssectional(theta_logit, data):
    lp, lF = theta_logit
    lp_prior = log_prior_p(lp) + log_prior_F(lF)
    if not np.isfinite(lp_prior):
        return -np.inf
    return log_likelihood_crosssectional(theta_logit, data) + lp_prior

def log_likelihood_temporal(theta_logit, data_generations, data_counts):
    """
    data_generations: sorted array of generation/year indices (starting from 0)
    data_counts: list of (N_AA, N_Aa, N_aa) at each timepoint
    """
    lp0, lF, ls, lh = theta_logit
    p0 = sigmoid(lp0)
    F  = sigmoid(lF) * 0.8
    s  = sigmoid(ls)
    h  = sigmoid(lh) * 4.0 - 2.0

    # Map years to generation indices (use relative spacing)
    gens = np.array(data_generations, dtype=int)
    gens = gens - gens[0]  # start from 0

    traj = run_wf(p0, F, s, h, gens)
    ll = 0.0
    for i, (n_AA, n_Aa, n_aa) in enumerate(data_counts):
        g = gens[i]
        p_t = float(np.clip(traj[g], 1e-6, 1 - 1e-6))
        ll += log_multinomial(n_AA, n_Aa, n_aa, p_t, F)
    return ll

def log_posterior_temporal(theta_logit, data_generations, data_counts):
    lp0, lF, ls, lh = theta_logit
    lprior = log_prior_p(lp0) + log_prior_F(lF) + log_prior_s(ls) + log_prior_h(lh)
    if not np.isfinite(lprior):
        return -np.inf
    return log_likelihood_temporal(theta_logit, data_generations, data_counts) + lprior


# ─── Metropolis-Hastings sampler ─────────────────────────────────────────────

def run_mcmc(log_posterior_fn, theta0, n_steps, n_burnin, proposal_scale=None, rng=None):
    """
    Adaptive Gaussian Metropolis-Hastings sampler.
    Returns: chain (n_steps x dim), acceptance rate
    """
    if rng is None:
        rng = np.random.default_rng()
    dim = len(theta0)
    if proposal_scale is None:
        proposal_scale = np.ones(dim) * ADAPT_SCALE

    theta = np.array(theta0, dtype=float)
    lp = log_posterior_fn(theta)
    chain = np.zeros((n_steps, dim))
    accepted = 0

    for i in range(n_steps + n_burnin):
        proposal = theta + rng.normal(0, proposal_scale, size=dim)
        lp_prop = log_posterior_fn(proposal)
        if np.log(rng.uniform()) < (lp_prop - lp):
            theta = proposal
            lp = lp_prop
            accepted += 1

        # Adaptive scaling during burn-in
        if i < n_burnin and (i + 1) % ADAPT_EVERY == 0:
            rate = accepted / (i + 1)
            if rate > TARGET_RATE:
                proposal_scale *= 1.1
            else:
                proposal_scale *= 0.9
            proposal_scale = np.clip(proposal_scale, 1e-4, 2.0)

        if i >= n_burnin:
            chain[i - n_burnin] = theta

    acc_rate = accepted / (n_steps + n_burnin)
    return chain, acc_rate


# ─── Gelman-Rubin R-hat ─────────────────────────────────────────────────────

def gelman_rubin(chains):
    """
    chains: list of arrays, each shape (n_steps, dim)
    Returns: R-hat per parameter
    """
    chains = np.array(chains)
    m, n, dim = chains.shape
    rhat = np.zeros(dim)
    for d in range(dim):
        psi = chains[:, :, d]
        psi_bar_j = psi.mean(axis=1)
        psi_bar   = psi_bar_j.mean()
        B = n / (m - 1) * np.sum((psi_bar_j - psi_bar)**2)
        W = psi.var(axis=1, ddof=1).mean()
        var_hat = (n - 1) / n * W + B / n
        rhat[d] = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhat


# ─── R_idx from posterior ────────────────────────────────────────────────────

def find_equilibrium(p0, F, s, h, max_iter=5000, tol=1e-10):
    p = p0
    for _ in range(max_iter):
        p_new = wf_step(p, F, s, h)
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p

def posterior_ridx(chain, n_params, param_type="cross"):
    """Compute R_idx for each posterior sample."""
    ridx_samples = []
    for theta in chain:
        if param_type == "cross":
            p = sigmoid(theta[0])
            F = sigmoid(theta[1]) * 0.8
            p_eq = p  # no temporal dynamics — equilibrium is p itself
        else:
            p0 = sigmoid(theta[0])
            F  = sigmoid(theta[1]) * 0.8
            s  = sigmoid(theta[2])
            h  = sigmoid(theta[3]) * 4.0 - 2.0
            p_eq = find_equilibrium(p0, F, s, h)
        ridx_samples.append(F * p_eq * (1.0 - p_eq))
    return np.array(ridx_samples)


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_traces_and_marginals(chains, param_names, dataset_name):
    m = len(chains)
    dim = len(param_names)
    fig, axes = plt.subplots(dim, 2, figsize=(10, 2.5 * dim))
    if dim == 1:
        axes = axes[np.newaxis, :]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for d, pname in enumerate(param_names):
        for j, chain in enumerate(chains):
            axes[d, 0].plot(chain[:400, d], color=colors[j], alpha=0.7, lw=0.8)
            axes[d, 1].hist(chain[:, d], bins=40, color=colors[j], alpha=0.5, density=True)
        axes[d, 0].set_ylabel(pname)
        axes[d, 1].axvline(np.concatenate([c[:, d] for c in chains]).mean(),
                            color="k", ls="--", lw=1.2)
        if d == 0:
            axes[d, 0].set_title("Trace (first 400)")
            axes[d, 1].set_title("Posterior marginals")

    plt.suptitle(f"Phase 3 MCMC — {dataset_name}", y=1.01)
    plt.tight_layout()
    fname = f"{OUT}/mcmc_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname

def plot_ridx_posterior(ridx_samples, dataset_name):
    mean_r = ridx_samples.mean()
    lo, hi = np.percentile(ridx_samples, [5, 95])
    plt.figure(figsize=(5, 3))
    plt.hist(ridx_samples, bins=50, density=True, color="steelblue", alpha=0.8)
    plt.axvline(lo, color="k", ls="--", lw=1.2, label=f"90% HDI [{lo:.5f}, {hi:.5f}]")
    plt.axvline(hi, color="k", ls="--", lw=1.2)
    plt.axvline(mean_r, color="red", ls="-", lw=1.5, label=f"Mean = {mean_r:.5f}")
    plt.xlabel("R_idx")
    plt.ylabel("Density")
    plt.title(f"Posterior R_idx — {dataset_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{OUT}/ridx_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 3 — Bayesian MCMC Inference")
    print("=" * 55)

    df  = pd.read_csv("hwe_datasets.csv")
    fst = pd.read_csv("phase1_fst_priors.csv").set_index("dataset_id")["F_ST"].to_dict()

    all_results = []

    # ── Cross-sectional datasets ──────────────────────────────────────────────
    print("\n--- Cross-sectional datasets ---")

    for did in CROSS_SECTIONAL_IDS:
        sub  = df[df["dataset_id"] == did]
        name = sub["dataset_name"].iloc[0]
        data = sub[["N_AA", "N_Aa", "N_aa"]].values.tolist()

        # Compute global allele frequency as starting point
        NAA = sub["N_AA"].sum(); NAa = sub["N_Aa"].sum(); Naa = sub["N_aa"].sum()
        N   = NAA + NAa + Naa
        p0  = (NAA + 0.5 * NAa) / N
        F0  = fst.get(did, 0.02)

        theta0 = [logit(p0), logit(F0 / 0.8)]
        log_post = lambda th: log_posterior_crosssectional(th, data)

        print(f"\n  {name}  (n={int(N)})")
        chains = []
        rng_list = [np.random.default_rng(42 + j) for j in range(N_CHAINS)]
        for j in range(N_CHAINS):
            # Jitter starting point
            theta_j = [t + rng_list[j].normal(0, 0.1) for t in theta0]
            chain, acc = run_mcmc(log_post, theta_j, N_STEPS, N_BURNIN, rng=rng_list[j])
            chains.append(chain)
            print(f"    Chain {j+1}: acceptance rate = {acc:.3f}")

        rhat = gelman_rubin(chains)
        print(f"    R-hat: {rhat}")

        all_chains = np.concatenate(chains, axis=0)
        ridx_s = posterior_ridx(all_chains, 2, "cross")

        p_post  = sigmoid(all_chains[:, 0])
        F_post  = sigmoid(all_chains[:, 1]) * 0.8

        print(f"    F:     {F_post.mean():.4f} ± {F_post.std():.4f}")
        print(f"    p:     {p_post.mean():.4f} ± {p_post.std():.4f}")
        print(f"    R_idx: {ridx_s.mean():.5f}  90% HDI [{np.percentile(ridx_s,5):.5f}, {np.percentile(ridx_s,95):.5f}]")

        plot_traces_and_marginals(chains, ["logit(p)", "logit(F)"], name)
        plot_ridx_posterior(ridx_s, name)

        all_results.append({
            "dataset": name,
            "type": "cross",
            "F_mean": F_post.mean(), "F_sd": F_post.std(),
            "p_mean": p_post.mean(), "p_sd": p_post.std(),
            "ridx_mean": ridx_s.mean(),
            "ridx_lo90": np.percentile(ridx_s, 5),
            "ridx_hi90": np.percentile(ridx_s, 95),
            "rhat_max": np.nanmax(rhat)
        })

    # ── Temporal dataset ──────────────────────────────────────────────────────
    print(f"\n--- Temporal dataset (ID={TEMPORAL_DATASET}) ---")

    sub  = df[df["dataset_id"] == TEMPORAL_DATASET].sort_values("generation_or_year")
    name = sub["dataset_name"].iloc[0]
    data_counts = sub[["N_AA", "N_Aa", "N_aa"]].values.tolist()

    # Normalise generations to start from 0
    years_raw = sub["generation_or_year"].values.astype(int)
    data_generations = years_raw - years_raw[0]

    # Starting point from allele frequencies at first timepoint
    row0 = data_counts[0]
    N0   = sum(row0)
    p0   = (row0[0] + 0.5 * row0[1]) / N0
    F0   = fst.get(TEMPORAL_DATASET, 0.02)

    theta0 = [logit(p0), logit(F0 / 0.8), logit(0.15), logit((0.6 + 2.0) / 4.0)]
    log_post = lambda th: log_posterior_temporal(th, data_generations, data_counts)

    print(f"\n  {name}  ({len(data_counts)} timepoints)")
    chains = []
    rng_list = [np.random.default_rng(100 + j) for j in range(N_CHAINS)]
    for j in range(N_CHAINS):
        theta_j = [t + rng_list[j].normal(0, 0.05) for t in theta0]
        chain, acc = run_mcmc(log_post, theta_j, N_STEPS, N_BURNIN, rng=rng_list[j])
        chains.append(chain)
        print(f"    Chain {j+1}: acceptance rate = {acc:.3f}")

    rhat = gelman_rubin(chains)
    print(f"    R-hat: {rhat}")

    all_chains = np.concatenate(chains, axis=0)
    ridx_s = posterior_ridx(all_chains, 4, "temporal")

    p0_post = sigmoid(all_chains[:, 0])
    F_post  = sigmoid(all_chains[:, 1]) * 0.8
    s_post  = sigmoid(all_chains[:, 2])
    h_post  = sigmoid(all_chains[:, 3]) * 4.0 - 2.0

    print(f"    p0:    {p0_post.mean():.4f} ± {p0_post.std():.4f}")
    print(f"    F:     {F_post.mean():.4f} ± {F_post.std():.4f}")
    print(f"    s:     {s_post.mean():.4f} ± {s_post.std():.4f}")
    print(f"    h:     {h_post.mean():.4f} ± {h_post.std():.4f}")
    print(f"    R_idx (t=0): {ridx_s.mean():.5f}  90% HDI [{np.percentile(ridx_s,5):.5f}, {np.percentile(ridx_s,95):.5f}]")

    plot_traces_and_marginals(chains, ["logit(p0)", "logit(F)", "logit(s)", "logit(h)"], name)
    plot_ridx_posterior(ridx_s, name)

    # Save Phase 3 posterior means for Phase 2
    np.save("phase3_temporal_posterior.npy", {
        "s_mean": s_post.mean(), "h_mean": h_post.mean(),
        "F_mean": F_post.mean(), "p0_mean": p0_post.mean(),
        "dataset_id": TEMPORAL_DATASET,
        "data_generations": data_generations,
        "data_counts": data_counts
    }, allow_pickle=True)

    all_results.append({
        "dataset": name, "type": "temporal",
        "F_mean": F_post.mean(), "F_sd": F_post.std(),
        "p_mean": p0_post.mean(), "p_sd": p0_post.std(),
        "ridx_mean": ridx_s.mean(),
        "ridx_lo90": np.percentile(ridx_s, 5),
        "ridx_hi90": np.percentile(ridx_s, 95),
        "rhat_max": np.nanmax(rhat)
    })

    pd.DataFrame(all_results).to_csv("phase3_results.csv", index=False)

    print("\nResults saved to phase3_results.csv")
    print(f"Figures saved to {OUT}/")
    print("\nPhase 3 complete.")
    print("Proceed to Phase 2: python phase2_neural_ode.py")
