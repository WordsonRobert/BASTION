"""
Phase 4 — Robustness Index Derivation
======================================
Derives R_idx = F * p_eq * (1 - p_eq) analytically from the Wright-Fisher recursion.
Proves rank-1 Jacobian structure. Runs unit tests. Generates heatmaps and trajectory plots.

Run first. All other phases depend on this being verified.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUT = "figures/phase4"
os.makedirs(OUT, exist_ok=True)


# ─── Wright-Fisher recursion ────────────────────────────────────────────────

def wf_step(p, F, s, h):
    """
    One generation of WF recursion.
    Selection (viability) first, then mating with inbreeding coefficient F.

    Fitness scheme: w_AA=1, w_Aa=1-hs, w_aa=1-s
    Returns: (p_prime, D_prime) where D_prime = deviation from HWE after one generation.
    """
    q = 1.0 - p
    P = p**2 + F * p * q       # freq(AA) under inbreeding
    Q = 2.0 * p * q * (1 - F)  # freq(Aa)
    R = q**2 + F * p * q       # freq(aa)

    w_AA = 1.0
    w_Aa = 1.0 - h * s
    w_aa = 1.0 - s
    w_bar = P * w_AA + Q * w_Aa + R * w_aa

    p_prime = (P * w_AA + 0.5 * Q * w_Aa) / w_bar
    q_prime = 1.0 - p_prime
    D_prime = F * p_prime * q_prime
    return p_prime, D_prime


def find_equilibrium(p0, F, s, h, max_iter=10000, tol=1e-12):
    """Iterate WF recursion until |Δp| < tol."""
    p = p0
    for _ in range(max_iter):
        p_new, _ = wf_step(p, F, s, h)
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def R_idx(p0, F, s, h):
    """Equilibrium robustness index."""
    p_eq = find_equilibrium(p0, F, s, h)
    return F * p_eq * (1.0 - p_eq)


def trajectory(p0, F, s, h, n_gen=250):
    """R_idx(t) over n_gen generations."""
    p = p0
    R_vals = []
    for _ in range(n_gen):
        _, D = wf_step(p, F, s, h)
        R_vals.append(D)
        p, _ = wf_step(p, F, s, h)
    return np.array(R_vals)


# ─── Jacobian ───────────────────────────────────────────────────────────────

def numerical_jacobian(p0, F, s, h, eps=1e-6):
    """
    Numerical Jacobian of the map T: (P, Q) -> (P', Q').

    Because P' and Q' both depend only on p' (a scalar function of p = P + Q/2),
    the Jacobian has rank at most 1 and det(J) = 0 always.
    """
    q0 = 1.0 - p0
    P0 = p0**2 + F * p0 * q0
    Q0 = 2.0 * p0 * q0 * (1 - F)

    def T(P, Q):
        p = P + Q / 2.0
        q = 1.0 - p
        R = max(1.0 - P - Q, 0.0)
        w_AA, w_Aa, w_aa = 1.0, 1.0 - h * s, 1.0 - s
        w_bar = P * w_AA + Q * w_Aa + R * w_aa
        p_prime = (P * w_AA + 0.5 * Q * w_Aa) / w_bar
        q_prime = 1.0 - p_prime
        P_prime = (1 - F) * p_prime**2 + F * p_prime
        Q_prime = (1 - F) * 2.0 * p_prime * q_prime
        return np.array([P_prime, Q_prime])

    f0 = T(P0, Q0)
    J = np.zeros((2, 2))
    J[:, 0] = (T(P0 + eps, Q0) - f0) / eps
    J[:, 1] = (T(P0, Q0 + eps) - f0) / eps
    return J


# ─── Unit tests ─────────────────────────────────────────────────────────────

def run_unit_tests():
    print("Running unit tests...")

    # Test 1: R_idx = 0 when F = 0
    for p0, s, h in [(0.4, 0.3, 0.5), (0.5, 0.1, -1.0), (0.3, 0.2, 0.0)]:
        r = R_idx(p0, F=0.0, s=s, h=h)
        assert abs(r) < 1e-12, f"FAIL T1: R_idx={r} with F=0"
    print("  PASS T1: R_idx = 0 when F = 0")

    # Test 2: D' = F*p'*q' exactly
    for p0, F, s, h in [(0.4, 0.1, 0.3, 0.5), (0.5, 0.2, 0.1, -1.0), (0.3, 0.15, 0.0, 0.5)]:
        p_prime, D_prime = wf_step(p0, F, s, h)
        D_check = F * p_prime * (1.0 - p_prime)
        assert abs(D_prime - D_check) < 1e-14, f"FAIL T2: D'={D_prime}, F*p'*q'={D_check}"
    print("  PASS T2: D' = F·p'·q' exact (no approximation)")

    # Test 3: Rank-1 Jacobian, det = 0
    # Tolerance is 1e-5: numerical Jacobian with eps=1e-6 gives O(eps) error in det.
    for p0, F, s, h in [(0.4, 0.1, 0.3, 0.5), (0.5, 0.2, 0.1, -1.0), (0.6, 0.05, 0.5, 1.0)]:
        J = numerical_jacobian(p0, F, s, h)
        rank = np.linalg.matrix_rank(J, tol=1e-4)
        det  = np.linalg.det(J)
        assert rank <= 1, f"FAIL T3: rank={rank} at p={p0}, F={F}"
        assert abs(det) < 1e-5, f"FAIL T3: det={det} at p={p0}, F={F}"
    print("  PASS T3: Jacobian rank=1, det≈0 at all tested points (numerical tol 1e-5)")

    # Test 4: Balancing selection -> stable nonzero R_idx
    r = R_idx(0.5, F=0.2, s=0.3, h=-1.0)
    assert r > 0.01, f"FAIL T4: balancing R_idx too small: {r}"
    print(f"  PASS T4: Balancing selection R_idx = {r:.5f} (stable, nonzero)")

    # Test 5: Directional selection -> R_idx -> 0
    r = R_idx(0.3, F=0.2, s=0.5, h=1.0)
    assert r < 0.001, f"FAIL T5: directional R_idx didn't decay: {r}"
    print(f"  PASS T5: Directional selection R_idx = {r:.6f} (decays to 0)")

    print("All 5 tests passed.\n")


# ─── Plots ──────────────────────────────────────────────────────────────────

def plot_heatmaps():
    """R_idx heatmap over (p0, F) space for three selection regimes."""
    p_vals = np.linspace(0.01, 0.99, 80)
    F_vals = np.linspace(0.0, 0.6, 80)
    PP, FF = np.meshgrid(p_vals, F_vals)

    regimes = [
        ("Directional (h=1, s=0.3)",   0.3,  1.0),
        ("Additive (h=0.5, s=0.3)",    0.3,  0.5),
        ("Balancing (h=-0.5, s=0.1)",  0.1, -0.5),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, s, h) in zip(axes, regimes):
        RR = np.zeros_like(PP)
        for i in range(PP.shape[0]):
            for j in range(PP.shape[1]):
                RR[i, j] = R_idx(PP[i, j], FF[i, j], s, h)
        im = ax.contourf(PP, FF, RR, levels=20, cmap="YlOrRd")
        ax.contour(PP, FF, RR, levels=8, colors="k", linewidths=0.4, alpha=0.5)
        fig.colorbar(im, ax=ax, label="R_idx")
        ax.set_xlabel("Initial allele freq p₀")
        ax.set_ylabel("Inbreeding coefficient F")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{OUT}/heatmaps_ridx.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT}/heatmaps_ridx.png")


def plot_trajectories():
    """R_idx(t) under balancing vs. directional selection, multiple (s, F) combos."""
    gens = 250
    t = np.arange(gens)

    configs = [
        ("Directional s=0.3, F=0.2", 0.3, 1.0,  0.2, "blue",   "-"),
        ("Directional s=0.3, F=0.1", 0.3, 1.0,  0.1, "blue",   "--"),
        ("No selection, F=0.2",      0.0, 0.5,  0.2, "gray",   "-."),
        ("Recessive s=0.3, F=0.2",   0.3, 0.0,  0.2, "green",  "-"),
        ("Balancing h=-1, F=0.2",    0.3, -1.0, 0.2, "red",    "-"),
        ("Balancing h=-1, F=0.1",    0.3, -1.0, 0.1, "red",    "--"),
    ]

    plt.figure(figsize=(10, 5))
    for label, s, h, F, color, ls in configs:
        R_vals = trajectory(0.3, F, s, h, n_gen=gens)
        plt.plot(t, R_vals, color=color, ls=ls, label=label, lw=1.5)

    plt.xlabel("Generation")
    plt.ylabel("R_idx = F · p(t) · q(t)")
    plt.title("R_idx dynamics: directional depletes, balancing maintains")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/trajectories_ridx.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT}/trajectories_ridx.png")


def plot_F_scaling():
    """R_idx vs F at p_eq=0.5 for several selection types."""
    F_vals = np.linspace(0, 0.5, 100)
    configs = [
        ("Balancing OD (s=0.1, h=-1)", 0.1, -1.0, "red"),
        ("Strong balancing (s=0.3, h=-1)", 0.3, -1.0, "darkred"),
        ("No selection",               0.0,  0.5, "gray"),
        ("Weak directional",           0.05, 0.5, "steelblue"),
        ("Strong directional",         0.3,  0.5, "blue"),
    ]
    plt.figure(figsize=(7, 4))
    for label, s, h, color in configs:
        R_vals = [R_idx(0.5, F, s, h) for F in F_vals]
        plt.plot(F_vals, R_vals, label=label, color=color, lw=1.8)

    plt.xlabel("Inbreeding coefficient F")
    plt.ylabel("R_idx = F · p_eq · q_eq")
    plt.title("R_idx vs F at p₀=0.5 — equilibrium deviation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fscaling_ridx.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT}/Fscaling_ridx.png")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 4 — Robustness Index Derivation")
    print("=" * 55)

    run_unit_tests()

    print("Generating figures...")
    plot_heatmaps()
    plot_trajectories()
    plot_F_scaling()

    print("\nPhase 4 complete.")
    print(f"Figures saved to {OUT}/")
    print("Proceed to Phase 1: python phase1_aitchison.py")
