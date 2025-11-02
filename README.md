# 🧬 OXBio Variant Forecasting (LAMIP → IFFL)

**Goal.** End‑to‑end, reproducible pipeline for forecasting SARS‑CoV‑2 lineages from wastewater **SNV counts** using
a Beta–Binomial observation model, lineage mixtures, and particle filtering/smoothing.  
Outputs are **CSV tables** and **PNG/PDF figures** only — **no Parquet, no LaTeX**.

---

## Contents

- [What the pipeline does](#what-the-pipeline-does)
- [Data you provide](#data-you-provide)
- [Quickstart](#quickstart)
- [Pipeline stages](#pipeline-stages)
- [Mathematical sketch](#mathematical-sketch)
- [Configuration reference](#configuration-reference)
- [Outputs & file naming](#outputs--file-naming)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [License & citation](#license--citation)

---

## What the pipeline does

1. **Preprocess** raw counts into a tidy feature store; collect a **signature matrix** (mutation ↔ lineage weights).
2. Build **mutation‑level priors** (dispersion / effective sample size).
3. **Forecast lineage proportions** over time per site with an **Auxiliary Particle Filter** and a **Backward Simulation Smoother**.
4. Run **posterior predictive diagnostics** (coverage, PIT, residuals, calibration).
5. *(Optional)* **Detection** of emerging lineages with anytime‑valid **e‑values**, online e‑BH FDR, and **Shiryaev–Roberts** statistics.

All intermediate artifacts are written under `results/<stage>/{tables,figures,logs}` in **CSV + PNG/PDF** only.

---

## Data you provide

Place inputs under `data/` (or point config paths to your files). Minimal schemas:

### 1) SNV counts (long format)
`data/snv_counts.csv`
```text
sample_id,site_id,date,mutation,count,coverage
S1,WWTP-001,2024-05-01,A23403G,  12,  1540
S2,WWTP-001,2024-05-08,A23403G,  25,  1802
...
```
- `date` parseable by pandas; `coverage >= 0`.

### 2) Lineage mutation signatures
`data/signatures.csv`
```text
mutation,lineage,weight
A23403G,BA.5,1.0
C241T,GLOBAL,1.0
...
```
- `weight ∈ [0,1]` (probability a lineage expresses the mutation). A special `GLOBAL` column is allowed; any missing or all‑zero mutation rows are routed to `GLOBAL=1` automatically during forecasting.

### 3) Priors (hyperparameters per mutation)
`results/priors/tables/priors_hyperparams.csv`
```text
mutation,phi
A23403G,  220.0
C241T,     90.0
```
- You may alternatively provide `alpha,beta` with `phi = alpha+beta` inferred automatically.

> Tip: You can start without an explicit **priors** stage by crafting this CSV manually; the pipeline will use the columns it finds.

---

## Quickstart

> Requires Python ≥3.9 and packages listed in `requirements.txt` / `environment.yml`.

### 1) Install
```bash
# conda (recommended)
conda env create -f environment.yml
conda activate oxfv

# OR pip
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Minimal config
Save as `configs/example.yml`:
```yaml
seed: 1234

forecast:
  inputs:
    snv_counts_path: data/snv_counts.csv
    signatures_path: data/signatures.csv
    priors_hyperparams_path: results/priors/tables/priors_hyperparams.csv
    # baseline_theta_path: results/likelihood/tables/theta_long.csv  # optional

  pf:
    num_particles: 256
    kappa: 200.0        # Dirichlet concentration driving day-to-day variability
    init_kappa: 200.0

  smoothing:
    method: bss         # bss | map | none  (Backward Simulation Smoother or MAP)
    lambda_kl: 10.0     # used for MAP
    max_iter: 200

  selection:
    default: 0.0        # per-lineage selection coefficient s_ℓ (multiplicative 1+s)
    by_lineage: {}      # e.g., {BA.5: 0.03}

  posterior_predictive:
    pit_bins: 20
    num_draws: 200      # for t+1 counts

  filter:
    sites: null         # e.g., ["WWTP-001", "WWTP-010"]
    lineages: null

diagnostics:
  coverage_levels: [0.5, 0.8, 0.9, 0.95]
  pit_bins: 20
  kappa_default: 200.0  # used if priors lack phi/alpha/beta

detection:              # Optional
  alpha: 0.10
  min_prop: 0.01
  e_value_method: power
  e_value_power: 0.5
  sr_block_size: 7
  sr_nperm: 200
  focus_site: WWTP-001
```

### 3) Run per stage (simple driver)
Create `run_stage.py`:
```python
import yaml
from utils.run import RunContext
from stages.forecast import run_forecast
from diagnostics import run_diagnostics
from detection import run_detection

cfg = yaml.safe_load(open("configs/example.yml"))
run = RunContext.start(cfg)                # creates results/runs/<run_id> and sets seeds

# Forecast
ctx = run.stage("forecast")
run_forecast(cfg, ctx)
ctx.close()

# Diagnostics
ctx = run.stage("diagnostics")
run_diagnostics(cfg, ctx)
ctx.close()

# Detection (optional)
ctx = run.stage("detection")
run_detection(cfg, ctx)
ctx.close()
```

Run it:
```bash
python run_stage.py
```

---

## Pipeline stages

### 1) **Preprocessing** (expected inputs)
Produce the SNV long table and a signature matrix. Typical outputs:
- `results/preprocessing/tables/feature_store_snv.csv`
- `results/preprocessing/tables/signatures.csv`

### 2) **Priors**
Compute mutation‑wise hyperparameters (e.g., `phi = alpha+beta ≈ ESS`). Output:
- `results/priors/tables/priors_hyperparams.csv`

### 3) **Forecast**
Time‑series inference of lineage proportions per site using:
- **Auxiliary Particle Filter** (look‑ahead importance) with **Beta–Binomial** emissions.
- **Dirichlet** transitions with selection vector \( s \) (per‑lineage multiplicative growth \( \propto 1+s_\ell \)).
- **Backward Simulation Smoother** for a coherent trajectory sample.
- **Posterior predictive** summaries, **WAIC**, optional baseline ΔELPD, and **t+1** predictions for lineages **and** SNV counts.

The stage guarantees the signature matrix **covers all mutations present in priors**, auto‑adding/using `GLOBAL` when needed.
Outputs include per‑site figures (2×2 panels, mutation heatmaps, turnover) and tidy CSV tables for summaries.

### 4) **Diagnostics**
Analytic posterior predictive checks under the **Beta–Binomial** model:
- **Coverage** of central credible intervals,
- **PIT** histograms and KS tests,
- **Standardized residuals** and QQ plots,
- **Calibration** curves (empirical ≤ nominal q).

Tables/figures are written under `results/diagnostics/` for immediate inspection.

### 5) **Detection** *(optional)*
Anytime‑valid **e‑values** from posterior tail probabilities \( \mathbb{P}(\theta_{\ell,t} \le \text{min\_prop}) \) via a calibrated power family, online **e‑BH** for FDR control, and **Shiryaev–Roberts** (log‑space) with **block‑permutation** threshold calibration. Also writes per‑site timelines and lineage heatmaps.

---

## Mathematical sketch

### Observation model (SNV counts)
For mutation \( m \) at site \( s \), date \( t \):
\[
Y_{m,s,t} \sim \text{BetaBinomial}\big(n_{m,s,t},\, \mu_{m,s,t},\, \phi_m\big), \qquad 
\mu_{m,s,t} = \sum_{\ell} \theta_{\ell,s,t}\, S_{m,\ell},
\]
with \( \phi_m \) the **dispersion / effective sample size** per mutation and \( S \) the signature matrix.

### State evolution (lineages)
\[
\theta_{t+1} \sim \text{Dirichlet}\big(\kappa \cdot \pi(\theta_t, s)\big), 
\quad\text{where}\quad \pi_\ell(\theta_t,s) \propto \theta_{\ell,t}\,(1+s_\ell)^+.
\]
APF uses look‑ahead weights; smoothing is via **Godsill–Doucet–West** backward simulation.

### Diagnostics
Exact Beta–Binomial **PIT** and **coverage** are computed analytically (no Monte Carlo), with standardized residuals and calibration summaries.

### Detection (optional)
Construct **Beta** approximations from posterior summaries to tail probabilities, transform p→e via a power family ensuring \( \mathbb{E}[e]=1 \) under the null, then apply **online e‑BH** and **SR** with permutation‑calibrated thresholds.

---

## Configuration reference

### `forecast`

- `inputs.snv_counts_path` – long SNV table (see schema above).  
- `inputs.signatures_path` – mutation↔lineage weights (see schema above).  
- `inputs.priors_hyperparams_path` – priors per mutation (`phi` or `alpha,beta`).  
- `inputs.baseline_theta_path` – optional baseline deconvolution for ΔELPD and initialization.

- `pf.num_particles` – particle count (≥128 recommended).  
- `pf.kappa` / `pf.init_kappa` – Dirichlet concentration (smoothness).  
- `smoothing.method` – `bss` (default), `map`, or `none`.  
- `selection.default` / `selection.by_lineage` – selection coefficients \( s_\ell \).  
- `posterior_predictive.pit_bins`, `posterior_predictive.num_draws` – diagnostics & t+1 draws.  
- `filter.sites`, `filter.lineages` – optional include lists.  
- `tplus1.use_last_coverage` – reuse last coverages for one‑day‑ahead SNV count predictions.

### `diagnostics`

- `coverage_levels` – e.g., `[0.5, 0.8, 0.9, 0.95]`.  
- `pit_bins` – histogram bins.  
- `kappa_default` – fallback if priors lacked `phi/alpha/beta`.  
- `max_obs`, `chunk_rows` – performance caps for large datasets.  
- `sample_pit_per_site` – PIT down‑sample size per site.  
- `write_pred_obs_sample` – include a small observed vs predicted join.

### `detection` (optional)

- `alpha` – FDR target and e‑value line \( 1/\alpha \).  
- `min_prop` – null threshold \( H_0:\theta_{\ell,t} \le \text{min\_prop} \).  
- `e_value_method`, `e_value_power`, `p_floor`.  
- `sr_block_size`, `sr_nperm`, `sr_alpha` – SR calibration.  
- `ebh_alpha` – e‑BH level.  
- `focus_site` – which site to highlight in figures.  
- `reference_dates` / `reference_dates_file` – optional CSV or inline mapping for delay analysis.

---

## Outputs & file naming

All outputs live under `results/<stage>/`:

- **Forecast / figures** (per site):  
  `site_panel_2x2_<site>.png`, `site_mutations_2x2_<site>.png`, `site_mutation_heatmap_<site>.png`, `site_turnover_<site>.png`

- **Forecast / tables (examples)**:  
  `forecast_smoothed_props.csv` (posterior summaries by site/date/lineage),  
  `tplus1_predictions.csv` (if enabled), `waic.csv`, `delta_elpd.csv`, etc.

- **Diagnostics / tables**:  
  `ppc_coverage.csv`, `ppc_pit_hist.csv`, `ppc_pit_ks.csv`, `ppc_residuals_summary.csv`, `ppc_calibration_curve.csv`

- **Diagnostics / figures**:  
  `ppc_panel.png`, `ppc_coverage.png`, `ppc_pit_rank.png`

- **Detection / tables**:  
  `e_values.csv`, `q_values.csv`, `sr_statistics.csv`, `sr_thresholds.csv`, `detection_calls.csv`, `detection_delays.csv`

- **Detection / figures**:  
  `detection_timelines_<site>.png`, `site_heatmaps_<site>.png`

> **Note**: This repository saves **CSV** and **PNG/PDF** only (no Parquet, no LaTeX).

---

## Reproducibility

- Global and per‑stage seeds are set via config; deterministic NumPy/`random` paths are used.
- Each run is placed under `results/runs/<run_id>`; `results/runs/latest` points to the latest run.
- Structured JSONL logs live under `results/<stage>/logs/` and include payload context for later audit.

---

## Troubleshooting

- **“Counts table missing required columns”** → verify headers exactly: `sample_id, site_id, date, mutation, count, coverage`.
- **“Cannot identify theta value column”** (detection/diagnostics) → ensure smoothed forecast tables include one of: `theta | median | mean | value`, or a quantile near 0.5 (e.g., `q50`).
- **Signature coverage**: if a mutation is absent or has all‑zero weights across lineages, it is routed to `GLOBAL=1.0` automatically.
- **Runtime**: reduce `pf.num_particles` or restrict `filter.sites` / `filter.lineages` for quick tests.

---

## License & citation

MIT License — see `LICENSE`.

If you use this framework, please cite:
```bibtex
@misc{oxbio_forecasting,
  author       = {Owais Soomro},
  title        = {OXBio Variant Forecasting (LAMIP → IFFL)},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-repo>}},
  note         = {Auxiliary Particle Filter + Beta--Binomial + IFFL}
}
```
