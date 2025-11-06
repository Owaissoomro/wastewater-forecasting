# Baseline evidence: poisson

Artifacts created by `stages/baseline.py`.

- meta.json — run metadata and environment
- resolved_config.yaml — exact model/prior settings
- run_summary.csv/jsonl — one row per baseline (log-evidence + posterior summary)
- <model_id>*/posterior_params.json — conjugate posterior hyperparameters (MAP/means)
- <model_id>*/evidence.json — log marginal likelihood
- <model_id>*/grid_check.json — 1D numerical validation (if enabled)

Notes:
- Bernoulli–Beta: optional Binomial coefficient via include_binomial_coefficient.
- Poisson–Gamma (RATE β): evidence includes ∑log(x_i!).
- Gaussian–NIG: evidence includes (2π)^(-n/2).
