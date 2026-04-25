"""
Phase 1 — Aitchison Geometry Clustering
=========================================
Genotype frequencies (P, Q, R) are compositional data on the 2-simplex.
Euclidean distance is geometrically invalid on the simplex.

Solution: Centred Log-Ratio (CLR) transform maps to Euclidean space.
K-means is then applied on CLR-transformed data.
Calinski-Harabasz index selects optimal k.
F_ST is estimated from cluster allele frequency variance as Phase 3 prior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import os, warnings

warnings.filterwarnings("ignore")

OUT = "figures/phase1"
os.makedirs(OUT, exist_ok=True)

PSEUDOCOUNT = 0.5  # Jeffreys pseudocount for zero cells


# ─── CLR transform ──────────────────────────────────────────────────────────

def clr_transform(P, Q, R):
    """
    Centred Log-Ratio transform for compositional data (P, Q, R) on 2-simplex.
    Jeffreys pseudocount 0.5 applied to zero cells before transform.
    Returns clr vector in R^3 (sum=0 constraint gives effective R^2 embedding).
    """
    P = np.array(P, dtype=float) + PSEUDOCOUNT
    Q = np.array(Q, dtype=float) + PSEUDOCOUNT
    R = np.array(R, dtype=float) + PSEUDOCOUNT
    total = P + Q + R
    P, Q, R = P / total, Q / total, R / total
    g = (P * Q * R) ** (1.0 / 3.0)
    return np.column_stack([np.log(P / g), np.log(Q / g), np.log(R / g)])


# ─── F_ST estimation ────────────────────────────────────────────────────────

def compute_fst(cluster_labels, allele_freqs):
    """
    F_ST from Weir & Cockerham: variance of allele frequency across clusters
    divided by p_bar * (1 - p_bar).
    """
    labels = np.array(cluster_labels)
    p = np.array(allele_freqs)
    p_bar = p.mean()
    if p_bar < 1e-6 or p_bar > 1 - 1e-6:
        return 0.0
    cluster_means = [p[labels == k].mean() for k in np.unique(labels)]
    var_p = np.var(cluster_means)
    fst = var_p / (p_bar * (1.0 - p_bar))
    return float(np.clip(fst, 0.0, 1.0))


# ─── Cluster one dataset ────────────────────────────────────────────────────

def cluster_dataset(name, rows, k_range=range(2, 8)):
    """
    Given rows = list of (N_AA, N_Aa, N_aa), run CLR + K-means.
    Returns: best_k, cluster_labels, fst, clr_data
    """
    rows = np.array(rows, dtype=float)
    N_AA, N_Aa, N_aa = rows[:, 0], rows[:, 1], rows[:, 2]
    total = N_AA + N_Aa + N_aa
    P = N_AA / total
    Q = N_Aa / total
    R = N_aa / total
    p_allele = P + 0.5 * Q  # allele frequency of A

    clr = clr_transform(P, Q, R)

    if len(rows) < 3:
        # Not enough observations for clustering — assign all to one cluster
        return 1, np.zeros(len(rows), dtype=int), 0.0, clr, p_allele

    best_k, best_score, best_labels = 2, -np.inf, None
    for k in k_range:
        if k >= len(rows):
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(clr)
        if len(np.unique(labels)) < 2:
            continue
        score = calinski_harabasz_score(clr, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_labels is None:
        best_labels = np.zeros(len(rows), dtype=int)

    fst = compute_fst(best_labels, p_allele)
    return best_k, best_labels, fst, clr, p_allele


def plot_clusters(name, clr, labels, fst, best_k):
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, best_k))
    for k in range(best_k):
        mask = labels == k
        ax.scatter(clr[mask, 0], clr[mask, 1],
                   color=colors[k], label=f"Cluster {k+1}", s=50, alpha=0.8)
    ax.set_xlabel("clr(P)")
    ax.set_ylabel("clr(Q)")
    ax.set_title(f"{name} | k={best_k} | F_ST={fst:.4f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{OUT}/clusters_{name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 1 — Aitchison Geometry Clustering")
    print("=" * 55)

    df = pd.read_csv("hwe_datasets.csv")

    # Process each dataset
    fst_priors = {}
    results_rows = []

    datasets_to_cluster = df["dataset_id"].unique()

    for did in sorted(datasets_to_cluster):
        sub = df[df["dataset_id"] == did].copy()
        name = sub["dataset_name"].iloc[0]
        rows = sub[["N_AA", "N_Aa", "N_aa"]].values

        if len(rows) < 3:
            # Single-population dataset: compute F_IS directly from counts.
            # F_IS = 1 - H_obs / H_exp  (this is the appropriate prior for Phase 3)
            NAA = rows[:, 0].sum(); NAa = rows[:, 1].sum(); Naa = rows[:, 2].sum()
            N   = NAA + NAa + Naa
            p   = (NAA + 0.5 * NAa) / N
            H_obs = NAa / N
            H_exp = 2.0 * p * (1.0 - p)
            fis = float(np.clip(1.0 - H_obs / H_exp if H_exp > 1e-6 else 0.0, 0.0, 0.8))
            print(f"  {name:30s}  k=1 (single pop)  F_IS={fis:.4f}")
            fst_priors[did] = fis
            results_rows.append({"dataset_id": did, "dataset_name": name,
                                  "n_rows": len(rows), "optimal_k": 1, "F_ST": round(fis, 4)})
            continue

        best_k, labels, fst, clr, p_allele = cluster_dataset(name, rows)
        fst_priors[did] = fst
        results_rows.append({"dataset_id": did, "dataset_name": name,
                              "n_rows": len(rows), "optimal_k": best_k, "F_ST": round(fst, 4)})
        print(f"  {name:30s}  k={best_k}  F_ST={fst:.4f}")
        plot_clusters(name, clr, labels, fst, best_k)

    # Save F_ST priors for Phase 3
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("phase1_fst_priors.csv", index=False)

    print("\nF_ST priors saved to phase1_fst_priors.csv")
    print(f"Cluster plots saved to {OUT}/")
    print("\nPhase 1 complete.")
    print("Proceed to Phase 3: python phase3_mcmc.py")
