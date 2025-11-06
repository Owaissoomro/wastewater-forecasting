# Priors stage report

- Mutations fitted: 10
- Total rows used: 10428
- Median κ (ESS): 0.35
- ΔLL(BB - Binomial): median=1376355.566 (mean=12860159.855)
- |residual| > 2 proportion: 0.009

Figures generated:
- priors_diagnostics
- timeseries_all_mutations
- timeseries_<SITE>_page_XX
- pmf_overlay (if enough mutations)
# Priors stage report

- Mutations fitted: 10
- Total rows used: 10428
- Median κ (ESS): 0.35
- ΔLL(BB - Binomial): median=1376355.566 (mean=12860159.855)
- |residual| > 2 proportion: 0.009

Figures generated:
- priors_diagnostics
- timeseries_all_mutations
- timeseries_<SITE>_page_XX
- pmf_overlay (if enough mutations)
# Priors stage report

- Mutations fitted: 62
- Total rows used: 60251
- Median κ (ESS): 0.44
- ΔLL(BB - Binomial): median=932891.012 (mean=10480180.242)
- |residual| > 2 proportion: 0.010
- PIT KS p-value: 0
- Interval coverage (equal-tailed Beta–Binomial):
  - Nominal 50% → empirical 0.636

## Time-local priors
- window_days = 28, per_site = True, smooth = ema (α=0.3)
- Used μₜ, κₜ in ribbons/points where available; fell back to global priors otherwise.
# Priors stage report

- Mutations fitted: 62
- Total rows used: 60251
- Median κ (ESS): 0.44
- ΔLL(BB - Binomial): median=932891.012 (mean=10480180.242)
- |residual| > 2 proportion: 0.010
- PIT KS p-value: 0
- Interval coverage (equal-tailed Beta–Binomial):
  - Nominal 50% → empirical 0.636
  - Nominal 80% → empirical 0.921
  - Nominal 95% → empirical 0.953

## Time-local priors
- window_days = 28, per_site = True, smooth = ema (α=0.3)
- Used μₜ, κₜ in ribbons/points where available; fell back to global priors otherwise.
# Priors stage report

- Mutations fitted: 62
- Total rows used: 82538
- Median κ (ESS): 1.29
- ΔLL(BB − Binomial): median=942650.319
- |residual| > 2 proportion: 0.030
- PIT KS p-value: 0

## Time-local priors (per-site, closed-form κ_t)
- window_days = 28, min_rows = 16, smooth = ema (α=0.3)
# Priors stage report

- Mutations fitted: 62
- Total rows used: 82538
- Median κ (ESS): 1.29
- ΔLL(BB − Binomial): median=942650.319
- |residual| > 2 proportion: 0.030
- PIT KS p-value: 0

## Time-local priors (per-site, closed-form κ_t)
- window_days = 28, min_rows = 16, smooth = ema (α=0.3)
# Priors stage report

- Mutations fitted: 62
- Total rows used: 82538
- Median κ (ESS): 3.23
- ΔLL(BB − Binomial, weighted): median=3136.893
- |residual| > 2 proportion: 0.137
- PIT KS p-value: 0

## Time-local priors (per-site, closed-form κ_t)
- window_days = 28, min_rows = 16, smooth = ema (α=0.3)
# Priors stage report (EB-MAP)

- Mutations fitted: 62
- Total rows used: 82538
- Median κ(MAP): 3.19
- ΔLL(BB − Binomial, weighted): median=3136.835
- |residual| > 2 proportion: 0.234
- EB hyperparams: m_a=-3.541, τ_a=1.267, m_b=1.518, τ_b=1.091
- PIT KS p-value: 0

## Time-local priors (per-site)
- window_days = 28, min_rows = 16, smooth = ema (α=0.3)
# Priors stage report (rigorous EB-MAP)

- Mutations fitted: 62
- Total rows used: 82538
- Median κ(MAP): 2.99
- ΔLL(BB − Binomial, weighted): median=3136.878
- |residual| > 2 proportion: 0.156
- EB hyperparams: m_a=-3.804, τ_a=3.953, m_b=2.034, τ_b=4.060
- PIT KS p-value: 0

## Time-local priors (per-site)
- window_days = 28, min_rows = 16, smooth = ema (α=0.3)
# Priors report (EB–MAP, minimal)
- Mutations: 62
- Rows used:  82538
- Median κ:   2.95
- ΔLL median: 3136.863
- PIT KS p:   0

Likelihood-ready files:
  • results/priors/tables/priors_hyperparams.csv  (mutation, mu, kappa, alpha, beta)
  • results/priors/tables/priors_time_local.csv   (site_id, date, mutation, mu_t, kappa_t, n_rows_window, sum_coverage_window)  [if enabled]
# Refined Priors report
- Mutations: 62
- Rows used:  82538
- Median κ (final): 0.76
- ΔLL median (BB − Binom): 3135.150
- PIT KS p (hybrid): 0

Likelihood-ready file:
  • results/priors/tables/priors_hyperparams.csv
Diagnostics:
  • results/priors/tables/fit_ll.csv
  • results/priors/tables/residuals.csv
  • results/priors/tables/prior_predictive_pit.csv
Optional:
  • results/priors/tables/priors_time_local.csv
# Refined Priors report
- Mutations: 62
- Rows used:  82538
- Median κ (final): 0.76
- ΔLL median (BB-Bin, refined): 3135.150
- PIT KS p (hybrid): 0

Likelihood-ready file:
  • results/priors/tables/priors_hyperparams.csv  (mutation, mu, kappa, alpha, beta)
Diagnostics:
  • results/priors/tables/fit_ll.csv
  • results/priors/tables/residuals.csv
  • results/priors/tables/prior_predictive_pit.csv
Optional:
  • results/priors/tables/priors_time_local.csv
# Refined Priors report (variance-correct)
- Mutations: 62
- Rows used:  82538
- Median κ (final): 0.64
- ΔLL median (BB-Bin, refined): 3135.420
- PIT KS p (hybrid): 0
Files: priors_hyperparams.csv, fit_ll.csv, residuals.csv, prior_predictive_pit.csv
# Priors report (exact, variance-correct)
- Mutations: 62
- Rows used:  82538
- Median κ (final): 0.50
- ΔLL median (BB-Bin, refined): 3135.202
- PIT KS p (exact mid-P): 0
Files: priors_hyperparams.csv, fit_ll.csv, residuals.csv, prior_predictive_pit.csv, priors_time_local.csv
# Dynamic hierarchical priors report (exact BB, Laplace–Kalman)
- Mutations: 62
- Rows used: 82538
- Median κ (final): 1.13
- ΔLL median (BB-dynamic minus Binomial): 2162.999
- PIT KS p (exact mid-P): 0
Files: priors_hyperparams.csv, fit_ll.csv, residuals.csv, prior_predictive_pit.csv, priors_time_local.csv
