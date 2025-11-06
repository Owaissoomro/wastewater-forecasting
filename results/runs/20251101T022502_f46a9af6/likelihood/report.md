# Likelihood stage report

- Samples fitted: 1550 across 6 sites, lineages: 2
- Ridge λ: 1e-06 (chosen by κ(H(λ)) profile unless provided)
- Objective monotone (PGD): 1.000
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.4689, RMSE: 0.5410
- Unknown mass (median over samples): 0.0000

Identifiability hints:
- Top overlaps (≥0.90) listed in merge_suggestions table.
# Likelihood stage report

- Samples fitted: 1550 across 6 sites, lineages: 2
- Ridge λ: 1e-06 (chosen by κ(H(λ)) profile unless provided)
- Objective monotone (PGD): 1.000
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.4689, RMSE: 0.5410
- Unknown mass (median over samples): 0.0000

Identifiability hints:
- Top overlaps (≥0.90) listed in merge_suggestions table.
# Likelihood stage report

- Samples fitted: 1550 across 6 sites; lineages: 3
- Ridge λ: 10 (κ-profile unless provided)
- PGD objective monotone: 0.859
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1308, RMSE: 0.3121
- Unknown mass (median over samples): 0.0013
# Likelihood stage report

- Samples fitted: 1550 across 6 sites; lineages: 3
- Ridge λ: 10 (κ-profile unless provided)
- PGD objective monotone: 0.859
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1308, RMSE: 0.3121
- Unknown mass (median over samples): 0.0013
# Likelihood stage report

- Samples fitted: 1550 across 6 sites; lineages: 3
- Ridge λ: 10 (κ-profile unless provided)
- PGD objective monotone: 0.859
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1308, RMSE: 0.3121
- Unknown mass (median over samples): 0.0013
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Ridge λ: 2.31013 (κ-profile unless provided)
- PGD objective monotone: 0.703
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1881, RMSE: 0.2767
- Unknown mass (median over samples): 0.0049
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Ridge λ: 2.31013 (κ-profile unless provided)
- PGD objective monotone: 0.673
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1920, RMSE: 0.2858
- Unknown mass (median over samples): 0.0043
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Ridge λ: 2.31013 (κ-profile unless provided)
- PGD monotone: 1.000, simplex ok: 1.000
- Residual MAE: 0.2955, RMSE: 0.5288
- Unknown mass (median over samples): 0.0000
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Ridge λ: 0.00404962 (κ-profile unless provided)
- PGD objective monotone: 0.046
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1252, RMSE: 0.2477
- Unknown mass (median over samples): 0.0708
# Likelihood stage report

- Samples fitted (joint temporal): 1575 across 6 sites; lineages: 5
- Ridge λ: 0.00404962 (κ-profile unless provided)
- Temporal smoothing λ_ts: 1
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.1950, RMSE: 0.2965
- Unknown mass (median over samples): 0.0036
# Likelihood stage report

- Samples fitted (joint temporal): 1575 across 6 sites; lineages: 5
- Temporal smoothing λ_ts: 3, Ridge λ: 0.01, Overlap λ: 0
- Unknown extra ridge: 10, Dirichlet α0: 0.3, μ_t pull λ: 0
- Residual MAE: 0.1517, RMSE: 0.3538
- Unknown mass (median over samples): 0.0011
# Likelihood stage report (logistic‑normal, Beta–Binomial)\n\n- Samples fitted: 1575 across 6 sites; lineages: 5\n- λ_temporal (φ): 2, ridge φ: 0.0001, gauge: 0.001\n- Residual MAE: 0.1743, RMSE: 0.2710\n\nTips to tighten accuracy further:\n1) Increase temporal_smooth_lambda modestly to denoise more if trends are jagged.\n2) Ensure signatures.csv matches your assay; add UNKNOWN column (kept here) if not already present.\n3) Provide time‑local kappa in results/priors/tables/priors_time_local.csv;\n   the code uses those concentrations directly, improving calibration.\n4) If some mutations are systematically biased, consider re‑estimating signatures or\n   adding site‑specific calibration upstream (preprocessing).
# Likelihood stage report\n\n- Samples fitted (joint temporal): 1575 across 6 sites; lineages: 5\n- Ridge λ: 0.0001 (κ-profile unless provided); ridge_mode=diag_hessian\n- Temporal λ_l2: 0.6; λ_tv: 0.25; auto_tune=False\n- Preselection: True (top_k=14, min_k=6, score=snr)\n- Support refinement: True (thr=0.01, max_add=4)\n- Simplex constraints satisfied (fraction): 1.000\n- Residual MAE: 0.1967, RMSE: 0.3029\n- Unknown mass (median over samples): 0.0031
# Likelihood stage report

- Samples fitted (PDHG convex): 1575 across 6 sites; lineages: 5
- TV λ: 0.25; L2 λ: 0.6; ridge λ: 0.001
- Overlap penalty λ: 0
- Simplex constraints satisfied (fraction): 1.000
- Residual MAE: 0.2272, RMSE: 0.2506
- Unknown mass (median over samples): 0.0070
# Likelihood stage report\n\n- Samples fitted (joint temporal): 1575 across 6 sites; lineages: 5\n- Ridge λ: 0.001 (κ-profile merged with config)\n- Temporal smoothing: λ_L2=0.6, λ_TV=0.25\n- PDHG: iters≤2000, tol=1e-06, τ=4.16e-15, σ=5.41e+13\n- Simplex constraints satisfied (fraction): 1.000\n- Residual MAE: 0.3013, RMSE: 0.3689\n- Unknown mass (median over samples): 0.1395
# Likelihood stage report\n\n- Samples fitted (joint temporal): 1575 across 6 sites; lineages: 5\n- Ridge λ: 0.001 (κ-profile merged with config)\n- Temporal smoothing: λ_L2=0.6, λ_TV=0.25\n- PDHG: iters≤2000, tol=1e-06, τ=4.16e-15, σ=5.41e+13\n- Simplex constraints satisfied (fraction): 1.000\n- Residual MAE: 0.3013, RMSE: 0.3689\n- Unknown mass (median over samples): 0.1395
# Likelihood stage report

- Samples fitted: 1574 across 6 sites; lineages: 5
- Priors: time-local yes, global yes
- λ_ridge=0.001, λ_L2=0.6, λ_overlap=0, unknown_extra=12
- Simplex ok fraction: 1.000
- Residual MAE=0.2886, RMSE=0.3543
# Likelihood stage report

- Samples fitted: 1574 across 6 sites; lineages: 5
- Priors: time-local yes, global yes
- λ_ridge=0.001, λ_L2=0.6, λ_overlap=0, unknown_extra=12
- Simplex ok fraction: 1.000
- Residual MAE=0.2886, RMSE=0.3543
# Likelihood stage report

- Samples fitted: 1574 across 6 sites; lineages: 5
- Priors: time-local yes, global yes
- λ_ridge=0.001, λ_L2=0.6, λ_overlap=0, unknown_extra=12
- Simplex ok fraction: 1.000
- Residual MAE=0.2886, RMSE=0.3543
# Likelihood stage report

- Samples fitted: 1574 across 6 sites; lineages: 5
- Priors: time-local yes, global yes
- λ_ridge=0.001, λ_L2=0.6, λ_overlap=0, unknown_extra=12
- Simplex ok fraction: 1.000
- Residual MAE=0.2887, RMSE=0.3543
# Likelihood stage report

- Samples fitted: 1574 across 6 sites; lineages: 5
- Priors: time-local yes, global yes
- λ_ridge=0.0005, λ_L2=0.02, λ_overlap=0, unknown_extra=2
- Simplex ok fraction: 1.000
- Residual MAE=0.2886, RMSE=0.3543
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Priors mutations (authoritative): 62
- λ_ridge=0.0005, λ_L2(base)=0.02, λ_overlap=0, λ_unknown=0
- Simplex ok fraction: 1.000
- Residual MAE=0.2928, RMSE=0.3621

Notes:
• Mutation universe is taken strictly from priors_hyperparams.csv (no fallbacks).
• Signatures are aligned to priors; GLOBAL = 1 − row_sum(non-GLOBAL), rows rescaled if they summed > 1.
• Exact Binomial likelihood + Beta pseudo-count priors; convex MAP solved with Fisher-MD + Armijo.
• Temporal coupling is coverage-aware per site×date for trend stability (λ_t = base / sqrt(1 + cov/scale)).
# Likelihood stage report (logistic‑normal, Beta–Binomial)\n\n- Samples fitted: 1575 across 6 sites; lineages: 5\n- λ_temporal (φ): 2, ridge φ: 0.0001, gauge: 0.001\n- Residual MAE: 0.1690, RMSE: 0.2676\n\nTips to tighten accuracy further:\n1) Increase temporal_smooth_lambda modestly to denoise more if trends are jagged.\n2) Ensure signatures.csv matches your assay; add UNKNOWN column (kept here) if not already present.\n3) Provide time‑local kappa in results/priors/tables/priors_time_local.csv;\n   the code uses those concentrations directly, improving calibration.\n4) If some mutations are systematically biased, consider re‑estimating signatures or\n   adding site‑specific calibration upstream (preprocessing).
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Priors mutations (authoritative): 62
- λ_ridge=0.0003, λ_L2(base)=0.003, λ_overlap=0, λ_unknown=0
- κ scale=1, κ cap=inf
- Simplex ok fraction: 1.000
- Residual MAE=0.2856, RMSE=0.3565

Notes:
• IRLS builds a local WLS target from the exact Beta–Binomial curvature; each slice solves a convex QP on the simplex.
• Temporal coupling shrinks when coverage is high (λ_t = base / sqrt(1 + cov/scale)).
• Overlap penalty (θ^T O θ) and ridge are incorporated in the Hessian for stability.
• Timeseries plotting is downsampled to avoid RendererAgg OOM.
# Likelihood stage report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Priors mutations (authoritative): 62
- λ_ridge=0.0003, λ_L2(base)=0.003, λ_overlap=0, λ_unknown=0
- κ scale=1, κ cap=inf
- Simplex ok fraction: 1.000
- Residual MAE=0.2854, RMSE=0.3565

Notes:
• IRLS builds a local WLS target from the exact Beta–Binomial curvature; each slice solves a convex QP on the simplex.
• Prior-only mode uses y=n=0 internally; the objective becomes pure prior (a=κμ, b=κ(1-μ)).
• Temporal coupling shrinks when precision proxy (n or κ) is high: λ_t = base / sqrt(1 + proxy/scale).
• Overlap penalty (θ^T O θ) and ridge are incorporated in the Hessian for stability.
• Timeseries plotting is downsampled to avoid RendererAgg OOM.
# Likelihood v2 report

- Samples fitted: 1575 across 6 sites; lineages: 5
- Priors mutations (authoritative): 62
- λ_ridge=0.0003, λ_L2(base)=0.003, λ_overlap=0, λ_unknown=0
- IRLS iters≤10, ADMM iters≤500, tol=5e-07, adaptive_rho=True
- κ scale=1, κ cap=inf
- Simplex ok fraction: 1.000
- Residual MAE=0.2855, RMSE=0.3565
- Uncertainty table: yes, Leverage: yes
- Kalman smoother: off (q=0.0001, r=0.001)

Notes:
• IRLS builds a local WLS target from the exact Beta–Binomial curvature; each slice solves a convex QP on the simplex.
• Prior-only mode uses y=n=0 internally; the objective becomes pure prior (a=κμ, b=κ(1-μ)).
• Adaptive ADMM ρ ≈ sqrt(λ_max / mean_eig) accelerates convergence when H is ill-conditioned.
• θ uncertainty from diag(H^{-1}) (delta-method); mutation leverage from hat-matrix diag.
• Optional Kalman smoother (RW1) post-fit; re-projects each θ_t onto the simplex.
• Timeseries plotting is downsampled to avoid RendererAgg OOM; bands require theta_uncertainty.
