"""
Phase 2 — Physics-Informed Neural ODE
========================================
MLP learns dp/dt = f_theta(p). Output multiplied by p*(1-p) to enforce
dp/dt = 0 at fixation boundaries for any parameter theta.

Loss = L_data (binomial NLL on allele counts at observed timepoints)
     + lambda * L_phys (MSE between f_theta(p) and WF selection increment
                        at 150 collocation points anchored to Phase 3 posteriors)

Ensemble of 10 independently initialised networks.
Uncertainty band = 90% spread across ensemble.

Requires: phase3_temporal_posterior.npy (from Phase 3)
          torchdiffeq installed
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint

OUT = "figures/phase2"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

ENSEMBLE_SIZE  = 10
N_EPOCHS       = 800
LR             = 1e-3
LAMBDA_PHYS    = 0.5     # weight on physics loss
N_COLLOC       = 150     # collocation points for physics loss
HIDDEN          = 32
SOLVER         = "rk4"
RTOL           = 1e-4
ATOL           = 1e-5


# ─── WF selection increment ─────────────────────────────────────────────────

def wf_increment(p, s, h):
    """
    Discrete Wright-Fisher selection increment Δp.
    Δp = p_selection(p) - p
    """
    p  = torch.clamp(p, 1e-5, 1 - 1e-5)
    q  = 1.0 - p
    w_AA = torch.ones_like(p)
    w_Aa = 1.0 - h * s
    w_aa = 1.0 - s
    # Under HWE (no inbreeding in the vector field for the ODE)
    P = p**2
    Q = 2.0 * p * q
    R = q**2
    w_bar = P * w_AA + Q * w_Aa + R * w_aa
    p_prime = (P * w_AA + 0.5 * Q * w_Aa) / w_bar
    return p_prime - p


# ─── Neural ODE ──────────────────────────────────────────────────────────────

class VectorField(nn.Module):
    """Learns the vector field dp/dt = f_theta(p) * p * (1-p)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )
        # Small initialisation to start close to neutral dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, p):
        """p: (batch,) or scalar tensor."""
        p = p.reshape(-1, 1)
        out = self.net(p).squeeze(-1)
        boundary = p.squeeze(-1) * (1.0 - p.squeeze(-1))
        return out * boundary


def solve_ode(model, p0, t_span):
    """Integrate the ODE from p0 over t_span."""
    p0_tensor = torch.tensor([p0], dtype=torch.float32)
    t_tensor  = torch.tensor(t_span, dtype=torch.float32)
    traj = odeint(model, p0_tensor, t_tensor, method=SOLVER, rtol=RTOL, atol=ATOL)
    return traj.squeeze(-1).detach().numpy()


# ─── Losses ──────────────────────────────────────────────────────────────────

def binomial_nll(p_pred, n_counts, total_n):
    """Binomial NLL on allele counts. p_pred: predicted allele frequency."""
    p_pred = torch.clamp(p_pred, 1e-6, 1 - 1e-6)
    # n_counts = number of A alleles = 2*N_AA + N_Aa
    ll = n_counts * torch.log(p_pred) + (2 * total_n - n_counts) * torch.log(1.0 - p_pred)
    return -ll.mean()

def physics_loss(model, s, h, n_colloc=N_COLLOC):
    """MSE between f_theta(p) and WF increment at collocation points."""
    p_colloc = torch.linspace(0.01, 0.99, n_colloc).unsqueeze(-1)
    f_learned = model(torch.zeros(1), p_colloc.squeeze(-1))
    with torch.no_grad():
        f_wf = wf_increment(p_colloc.squeeze(-1), s, h)
    return nn.functional.mse_loss(f_learned, f_wf)


# ─── Training ────────────────────────────────────────────────────────────────

def train_one(p0, s_anchor, h_anchor, t_obs, allele_counts, total_ns, seed):
    torch.manual_seed(seed)
    model = VectorField()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    t_obs_t = torch.tensor(t_obs, dtype=torch.float32)
    counts_t = torch.tensor(allele_counts, dtype=torch.float32)
    totals_t = torch.tensor(total_ns,      dtype=torch.float32)

    losses = []
    for epoch in range(N_EPOCHS):
        optim.zero_grad()

        # Data loss: integrate ODE, compute binomial NLL at observed timepoints
        p0_t = torch.tensor([p0], dtype=torch.float32)
        traj = odeint(model, p0_t, t_obs_t, method=SOLVER, rtol=RTOL, atol=ATOL)
        p_pred = traj.squeeze(-1)

        l_data = binomial_nll(p_pred, counts_t, totals_t)
        l_phys = physics_loss(model, s_anchor, h_anchor)
        loss   = l_data + LAMBDA_PHYS * l_phys
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(loss.item())

    return model, losses


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 2 — Physics-Informed Neural ODE")
    print("=" * 55)

    # Load Phase 3 posterior
    post = np.load("phase3_temporal_posterior.npy", allow_pickle=True).item()
    s_mean = float(post["s_mean"])
    h_mean = float(post["h_mean"])
    F_mean = float(post["F_mean"])
    p0_mean = float(post["p0_mean"])
    data_generations = np.array(post["data_generations"], dtype=float)
    data_counts_raw  = post["data_counts"]

    print(f"  Phase 3 posteriors: s={s_mean:.3f}, h={h_mean:.3f}, F={F_mean:.3f}, p0={p0_mean:.3f}")

    # Prepare observed allele counts
    allele_counts = []
    total_ns = []
    for row in data_counts_raw:
        n_AA, n_Aa, n_aa = row
        N = n_AA + n_Aa + n_aa
        allele_counts.append(2 * n_AA + n_Aa)  # count of A alleles
        total_ns.append(N)

    t_obs = data_generations.astype(float)

    # WF reference trajectory
    def wf_trajectory_np(p0, F, s, h, max_gen):
        p = p0
        traj = [p0]
        for _ in range(int(max_gen)):
            q = 1 - p
            P = p**2 + F*p*q; Q = 2*p*q*(1-F); R = q**2 + F*p*q
            w_bar = P + Q*(1-h*s) + R*(1-s)
            p = (P + 0.5*Q*(1-h*s)) / w_bar
            traj.append(p)
        return np.array(traj)

    wf_traj = wf_trajectory_np(p0_mean, F_mean, s_mean, h_mean, int(t_obs[-1]) + 1)

    # Train ensemble
    print(f"\n  Training ensemble of {ENSEMBLE_SIZE} networks ({N_EPOCHS} epochs each)...")
    all_trajs = []
    t_fine = np.linspace(t_obs[0], t_obs[-1], 200)

    for i in range(ENSEMBLE_SIZE):
        model, losses = train_one(
            p0_mean, s_mean, h_mean,
            t_obs, allele_counts, total_ns,
            seed=i * 7
        )
        model.eval()
        with torch.no_grad():
            traj = solve_ode(model, p0_mean, t_fine)
        all_trajs.append(traj)
        final_loss = losses[-1]
        print(f"    Network {i+1:2d}: final loss = {final_loss:.5f}")

    all_trajs = np.array(all_trajs)  # (ENSEMBLE_SIZE, len(t_fine))
    median_traj = np.median(all_trajs, axis=0)
    lo5  = np.percentile(all_trajs, 5,  axis=0)
    hi95 = np.percentile(all_trajs, 95, axis=0)

    # R_idx(t) from NODE median trajectory
    ridx_node = F_mean * median_traj * (1.0 - median_traj)
    # R_idx(t) from WF trajectory
    wf_t_fine = wf_trajectory_np(p0_mean, F_mean, s_mean, h_mean, int(t_fine[-1]) + 5)
    ridx_wf   = np.array([F_mean * wf_t_fine[int(t)] * (1 - wf_t_fine[int(t)]) for t in t_fine])

    # Observed allele frequencies
    p_obs = np.array(allele_counts) / (2 * np.array(total_ns))

    # ── Agreement stats ───────────────────────────────────────────────────────
    p_node_at_obs = np.array([median_traj[np.argmin(np.abs(t_fine - t))] for t in t_obs])
    p_wf_at_obs   = np.array([wf_traj[int(t)] for t in t_obs])
    node_wf_diff  = np.abs(p_node_at_obs - p_wf_at_obs)
    print(f"\n  NODE vs WF agreement:")
    print(f"    Mean |Δp| = {node_wf_diff.mean():.4f}")
    print(f"    Max  |Δp| = {node_wf_diff.max():.4f}")

    # ── Figures ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: p(t) trajectory
    ax = axes[0]
    ax.fill_between(t_fine, lo5, hi95, alpha=0.25, color="steelblue", label="90% ensemble")
    ax.plot(t_fine, median_traj, color="steelblue", lw=2, label="NODE (median)")
    ax.plot([wf_traj[int(t)] for t in t_fine if int(t) < len(wf_traj)],
            color="orange", lw=1.5, ls="--", label="WF recursion")
    ax.scatter(t_obs, p_obs, color="red", zorder=5, s=40, label="Observed")
    ax.set_xlabel("Generation / Year")
    ax.set_ylabel("Allele frequency p")
    ax.set_title("p(t) — NODE vs WF")
    ax.legend(fontsize=8)

    # Panel 2: R_idx(t)
    ax = axes[1]
    ax.plot(t_fine, ridx_node, color="steelblue", lw=2, label="R_idx (NODE)")
    ax.plot(t_fine, ridx_wf,   color="orange",    lw=1.5, ls="--", label="R_idx (WF)")
    peak_idx = np.argmax(ridx_node)
    ax.axvline(t_fine[peak_idx], color="gray", ls=":", lw=1)
    ax.annotate(f"Peak gen {t_fine[peak_idx]:.0f}\nR_idx={ridx_node[peak_idx]:.4f}",
                xy=(t_fine[peak_idx], ridx_node[peak_idx]),
                xytext=(t_fine[peak_idx] + 2, ridx_node[peak_idx] * 0.8),
                fontsize=7)
    ax.set_xlabel("Generation / Year")
    ax.set_ylabel("R_idx = F · p(t) · q(t)")
    ax.set_title("R_idx(t)")
    ax.legend(fontsize=8)

    # Panel 3: Pointwise residuals
    ax = axes[2]
    ax.bar(t_obs, node_wf_diff, color="steelblue", alpha=0.8)
    ax.set_xlabel("Generation / Year")
    ax.set_ylabel("|p_NODE − p_WF|")
    ax.set_title("Pointwise residuals")
    ax.axhline(node_wf_diff.mean(), color="red", ls="--", lw=1.2,
               label=f"Mean = {node_wf_diff.mean():.4f}")
    ax.legend(fontsize=8)

    plt.suptitle("Phase 2 — Physics-Informed Neural ODE", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/neural_ode_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT}/neural_ode_results.png")

    # Ensemble uncertainty figure
    plt.figure(figsize=(6, 4))
    for traj in all_trajs:
        plt.plot(t_fine, traj, alpha=0.3, color="steelblue", lw=0.8)
    plt.plot(t_fine, median_traj, color="steelblue", lw=2, label="Median")
    plt.scatter(t_obs, p_obs, color="red", zorder=5, s=40, label="Observed")
    plt.xlabel("Generation / Year")
    plt.ylabel("Allele frequency p")
    plt.title(f"Ensemble uncertainty ({ENSEMBLE_SIZE} networks)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/ensemble_uncertainty.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT}/ensemble_uncertainty.png")

    print("\nPhase 2 complete.")
    print(f"Figures saved to {OUT}/")
    print("All phases done. See results.md for numerical summary.")
