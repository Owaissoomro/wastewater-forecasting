
# Oxford Bio — Wastewater Variant Forecasting

Deterministic,  pipeline for SARS-CoV-2 wastewater variant deconvolution.
Implements closed-form Beta–Binomial priors and a robust site-level likelihood solver using NumPy only.

---

## Overview

| Stage | Purpose | Main Outputs |
|--------|----------|--------------|
| preprocessing | Clean raw SNV tables, compute AF, coverage, and bias diagnostics | `results/preprocessing/tables/feature_store_snv.csv` |
| priors | Estimate daily smoothed allele-frequency means (μ_t) and global dispersion (κ) | `results/priors/priors_hyperparams.csv` |
| likelihood | Fit site-wise lineage mixture proportions (θ_st) under Beta–Binomial IRLS + ADMM | `results/likelihood/tables/theta_estimates.csv` |

Each stage writes CSVs under `results/<stage>/tables/` and logs progress to stdout as JSON.

---

## Data Inputs

All data live under `data/` or `results/preprocessing/tables/`.

```

data/
├── jahn_like.csv        # Long SNV table (site_id, date, mutation, count, coverage)
├── signatures.csv       # Lineage → mutation signature weights
└── lineages.csv         # Optional lineage metadata

````

---

## Running the Pipeline

```bash
# run priors (global Beta–Binomial hierarchy)
python scripts/run_pipeline.py --config configs/default.yaml --stages priors --seed 12345

# run likelihood (site-level deconvolution)
python scripts/run_pipeline.py --config configs/default.yaml --stages likelihood --seed 12345

# optional preprocessing and diagnostics
python scripts/run_pipeline.py --config configs/default.yaml --stages preprocessing --seed 12345
````

---

## Stage Summaries

### Preprocessing

Cleans and validates `jahn_like.csv`, computes allele frequencies (AF), limit-of-detection thresholds, and coverage summaries.
Outputs diagnostic tables and figures:

* coverage histograms, ECDFs, violins
* bias-loci scatter plots
* site-specific variant panels
* missingness heatmaps

Main output:
`results/preprocessing/tables/feature_store_snv.csv`

---

### Priors

Constructs hierarchical priors per mutation:

1. Daily mean AF (μ_t) via penalized logistic Binomial with RW2 smoothness on logit(μ_t).
2. Global dispersion κ via Beta–Binomial MLE.
3. Empirical-Bayes Gaussian prior on (a=logit μ, b=log κ) across mutations.

Main outputs:

* `priors_full_detail.csv`
* `priors_hyperparams.csv`
* `detail_global_timeseries.csv`
* `eb_population_prior.csv`

---

### Likelihood

Fits lineage mixture weights θ_st for each site and day.

1. Uses priors from the previous stage (μ, κ).
2. Uses `signatures.csv` as the lineage × mutation weight matrix.
3. Solves per-site optimization via IRLS + ADMM with simplex constraints.
4. Exports mixture estimates, residuals, and diagnostics.

Main outputs:

* `theta_estimates.csv`
* `residuals.csv`
* `objective_trace.csv`
* `zscore_diagnostics.csv`
* `mutation_leverage.csv`

---

## Notes

* All computations are deterministic and use only NumPy/SciPy.
* All paths are relative and writable (`results/...`, `data/...`).
* Figures are saved under `results/<stage>/figures/`.
* The pipeline can be run stage-by-stage or end-to-end through the `scripts/run_pipeline.py` driver.

```

---
```
