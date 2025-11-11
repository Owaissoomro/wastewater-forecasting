#  Wastewater Variant Forecasting

Pipeline for SARS-CoV-2 wastewater variant deconvolution.
Implements closed-form Beta–Binomial priors and a robust site-level likelihood solver using NumPy only.

---

## Overview

| Stage | Purpose | Main Outputs |
|--------|----------|--------------|
| baseline | End-to-end convenience run that orchestrates `priors` → `likelihood` with defaults | Same as **Priors** + **Likelihood** (see below) |
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
```

---

## Running the Pipeline

```bash
# end-to-end baseline (runs priors → likelihood with defaults)
python scripts/run_pipeline.py --config configs/default.yaml --stages baseline --seed 12345

# run priors (global Beta–Binomial hierarchy)
python scripts/run_pipeline.py --config configs/default.yaml --stages priors --seed 12345

# run likelihood (site-level deconvolution)
python scripts/run_pipeline.py --config configs/default.yaml --stages likelihood --seed 12345

# optional preprocessing and diagnostics
python scripts/run_pipeline.py --config configs/default.yaml --stages preprocessing --seed 12345
```

---

## Stage Summaries

### Baseline

Runs the canonical end-to-end workflow in one command using the defaults from `configs/default.yaml`.

1. Validates file paths and configuration.
2. Uses the precomputed features at `results/preprocessing/tables/feature_store_snv.csv` (generate with **Preprocessing** if missing).
3. Executes **Priors** to estimate daily AF means (μ_t) and global dispersion (κ).
4. Executes **Likelihood** to fit site/day lineage mixtures (θ_st) with simplex constraints.
5. Streams JSON progress logs to stdout and preserves all per-stage artifacts.

**Main outputs** (written to the same per-stage locations as below unless overridden by config):
- From **Priors**: `results/priors/priors_hyperparams.csv`, `results/priors/priors_full_detail.csv`, `results/priors/detail_global_timeseries.csv`, `results/priors/eb_population_prior.csv`
- From **Likelihood**: `results/likelihood/tables/theta_estimates.csv`, `results/likelihood/tables/residuals.csv`, `results/likelihood/tables/objective_trace.csv`, `results/likelihood/tables/zscore_diagnostics.csv`, `results/likelihood/tables/mutation_leverage.csv`

> Note: The `baseline` stage is an orchestrator; it does not introduce a new results directory by default. Override output roots in the config if you prefer a dedicated `results/baseline/` folder.

---

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
3. Empirical-Bayes Gaussian prior on (a = logit μ, b = log κ) across mutations.

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
