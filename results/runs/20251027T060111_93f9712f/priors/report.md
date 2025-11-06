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
