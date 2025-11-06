# Likelihood v3 report (JAX CPU)

- Samples fitted: 1575 across 6 sites; lineages: 5
- Priors mutations (authoritative): 62
- λ_ridge=0.0003, λ_L2(base)=0.003, λ_overlap=0, λ_unknown=0
- IRLS iters≤10, ADMM iters≤500, tol=5e-07, adaptive_rho=True
- κ scale=1, κ cap=inf
- Simplex ok fraction: 1.000
- Residual MAE=0.2980, RMSE=0.3612
- Uncertainty table: yes, Leverage: yes

Notes:
• IRLS curvature/targets are computed in JAX with exact Beta–Binomial + Beta prior pseudocounts.
• Each slice solves a dense simplex QP via JAX‑ADMM (one Cholesky per ρ).
• Temporal L2 enters both H and f exactly; overlap adds O penalty; ridge stabilizes.
• Prior‑only mode sets y=n=0 internally, using μ,κ; still writes full tables/figures.
