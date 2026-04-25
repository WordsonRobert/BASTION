# Datasets

## Summary

| ID | Name | Species | Locus | Reference | Stressor | Counts | Used in |
|---|---|---|---|---|---|---|---|
| 1 | HbC | Homo sapiens | HbC | Modiano 2001, Nature 414:305 | Balancing selection | Direct, Table 2 | Phase 1, 3 |
| 2 | Mc1r | Chaetodipus intermedius | Mc1r | Nachman 2003, PNAS 100:5268 | Directional selection | Direct, Table 2 | Phase 1 |
| 3 | Drosophila Adh | Drosophila melanogaster | Adh | Cavener & Clegg 1981, Evolution 35:1 | Ethanol selection | **FLAGGED** (back-calculated) | Phase 1 only |
| 4 | Gadus Syp I | Gadus morhua | Syp I | Fevolden & Pogson 1997, J Fish Biol 50:748 | Population structure | Back-calculated | Phase 1 |
| 5 | Zoarces Est III | Zoarces viviparus | Est III | Christiansen 1973, Genetics 73:291 | Zygotic selection | Direct, pooled by year | Phase 1, 3 |
| 6 | CCR5-Δ32 | Homo sapiens | CCR5 | Martinson 1997, Nat Genet 16:100 | Population structure + drift | Direct, 46 populations | Phase 1, 3 |
| 7 | Panaxia medionigra | Panaxia dominula | medionigra | Fisher & Ford 1947, Heredity 1:143 | Complex selection | Direct, Table 1 | Phase 1, 2, 3 |

---

## Dataset 1 — HbC (Modiano 2001)

- **Locus:** Haemoglobin C β-globin variant
- **Genotypes:** HbAA (N_AA), HbAC (N_Aa), HbCC (N_aa)
- **Population:** Burkina Faso, aggregate across ethnic groups
- **n:** 418
- **Counts:** N_AA=324, N_Aa=88, N_aa=6
- **Source:** Table 2, Modiano et al. 2001, Nature 414:305
- **Data quality:** Direct observed genotype counts. No reconstruction.

---

## Dataset 2 — Mc1r (Nachman 2003)

- **Locus:** Melanocortin-1 receptor
- **Genotypes:** dark/dark (N_AA), dark/light (N_Aa), light/light (N_aa)
- **Population:** Two populations — lava rock (dark substrate) and light rock
- **Source:** Table 2, Nachman et al. 2003, PNAS 100:5268
- **Data quality:** Direct observed genotype counts. No reconstruction.
- **Use in pipeline:** Phase 1 (CLR clustering) only. Not used in Phase 3 MCMC.

---

## Dataset 3 — Drosophila Adh (Cavener & Clegg 1981)  ⚠️ FLAGGED

- **Locus:** Alcohol dehydrogenase (Adh-F / Adh-S alleles)
- **Population:** Controlled ethanol-selection cage experiment
- **Timepoints:** 12 (4 cage populations × multiple generations)
- **Source:** Figure 1, Cavener & Clegg 1981, Evolution 35:1
- **⚠️ Data quality issue:** Raw genotype counts were NOT published. Counts in `hwe_datasets.csv` are back-calculated from reported allele frequencies assuming HWE at each timepoint. This means observed counts satisfy HWE exactly by construction, so F inference from Phase 3 MCMC is prior-dominated — the data contain no genuine deviation signal for F.
- **Use in pipeline:** Phase 1 only. Replaced by Dataset 7 (Panaxia) in Phase 2 and 3.

---

## Dataset 4 — Gadus morhua Syp I (Fevolden & Pogson 1997)

- **Locus:** Synaptophysin I (Syp I)
- **Population:** Norwegian spring spawning cod
- **Source:** Fevolden & Pogson 1997, J Fish Biol 50:748
- **Data quality:** Counts reconstructed from reported n, p, and H_obs. Use with caution.
- **Use in pipeline:** Phase 1 (CLR clustering) only.

---

## Dataset 5 — Zoarces Est III (Christiansen 1973)

- **Locus:** Esterase III
- **Population:** Denmark, pooled across years
- **Source:** Christiansen 1973, Genetics 73:291
- **Data quality:** Direct observed genotype counts.
- **Use in pipeline:** Phase 1, Phase 3 (cross-sectional model).

---

## Dataset 6 — CCR5-Δ32 (Martinson 1997)

- **Locus:** CCR5 chemokine receptor (32-bp deletion variant)
- **Population:** 46 populations globally. Represented in CSV as 4 regional aggregates.
- **Source:** Table 1, Martinson et al. 1997, Nat Genet 16:100
- **Data quality:** Direct observed genotype counts across 46 populations.
- **Use in pipeline:** Phase 1, Phase 3 (cross-sectional model, pooled across populations).

---

## Dataset 7 — Panaxia dominula medionigra (Fisher & Ford 1947)

- **Locus:** medionigra gene (wing pattern)
- **Genotypes:** dominula/dominula (N_AA), dominula/medionigra (N_Aa), bimacula/bimacula (N_aa)
- **Codominant:** All three genotypes directly distinguishable by phenotype. No reconstruction.
- **Population:** Isolated marsh colony near Cothill, Oxfordshire. Mark-release-recapture estimates of population size available (Table 15 of paper).
- **Timepoints:** 9 (pre-1929 pooled + 1939–1946 consecutive years)
- **Source:** Table 1, Fisher & Ford 1947, Heredity 1:143
- **Data quality:** Direct observed counts from Table 1. No HWE assumption. No back-calculation.
- **Note:** Allele frequency trajectory is non-monotone (rises 1928→1940, then declines and stabilises ~0.05). Fisher & Ford conclude fluctuating selection is responsible (Section 6, χ²=20.8, df=7, p<0.01). The fixed-(s,h) WF model fits an average trajectory; year-by-year deviations are expected and are stated as a pipeline limitation.
- **Use in pipeline:** Phase 1, Phase 2 (Neural ODE), Phase 3 (temporal MCMC). Primary temporal dataset.
