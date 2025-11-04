# -*- coding: utf-8 -*-
"""

=========================================================

Purpose
-------
Deterministic construction of hierarchical priors for wastewater SNV
allele-frequency (AF) time series 

For each mutation:
1) Estimate a **global day-wise mean AF μ_t** (aggregated across sites) by solving a
   penalized logistic Binomial problem with an RW2 smoothness penalty on the logit
   scale (Newton steps with PD Hessian).
2) Estimate a **single Beta–Binomial dispersion κ** for that mutation by MLE on the
   log-scale, holding μ_t fixed (all rows, robust bounds).

Outputs (flat CSVs; identical schema to your Stan stage)
--------------------------------------------------------
1) results/priors/priors_full_detail.csv
   Row-level table with per-row μ_t (mapped from day) and constant κ̂ for the mutation.
2) results/priors/priors_hyperparams.csv
   Per-mutation empirical-Bayes (EB) hyperparameters (μ, κ, α=μκ, β=(1−μ)κ).
3) results/priors/detail_global_timeseries.csv
   Per-mutation daily summary of μ_t and κ̂ on the (a=logit μ, b=log κ) scale.
4) results/priors/eb_population_prior.csv
   EB Gaussian on (a, b) across mutations: mean (m_a, m_b) and covariance (Saa, Sab, Sbb).

Data expectations
-----------------
Prefers your pipeline paths; falls back to `data/`. Required columns:
    site_id, date, mutation, count, coverage
Dates must parse to pandas datetimes. Coverage is auto-capped (optional quantile).

Why this design?
----------------
- **Stability / speed**: convex data term + quadratic penalty → well-conditioned Newton.
- **Interpretability**: RW2 on logit(μ_t) is a standard smooth trend prior; κ MLE is
  scalar and robustly bounded.
- **Compatibility**: Emits the exact CSVs  downstream expects.

CLI
---
    python scripts/run_pipeline.py --config configs/default.yaml --stages priors --seed 12345




Config keys (all optional; shown with defaults)
-----------------------------------------------
priors:
  data_root: "data"
  results_root: "results"
  cap_coverage_quantile: 0.99     # clip coverage to this quantile for stability (0.5–<1)
  rw2_penalty: 10.0               # λ for RW2 smoothness on a_t = logit(μ_t)
  kappa_lo: 0.05                  # lower bound for κ MLE (strictly > 0)
  kappa_hi: 100.0                 # upper bound for κ MLE
  min_dates: 2                    # skip mutations with fewer distinct days

Notes
-----
- This stage is deliberately minimal and side-effect free except for writing CSVs.
- All numeric operations are guarded with small epsilons to avoid boundary issues.
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.special import expit, logit
from scipy.stats import betabinom
from scipy.optimize import minimize_scalar

# ------------------------ constants ------------------------
EPS: float = 1e-9
GLOBAL_SITE_ID: str = "__GLOBAL__"


def _clip01(x: np.ndarray | float, eps: float = 1e-9) -> np.ndarray:
    """Clamp to the open unit interval (eps, 1-eps) to avoid boundary issues.

    Parameters
    ----------
    x : array-like or float
        Values to clip.
    eps : float
        Small positive tolerance; default 1e-9.

    Returns
    -------
    np.ndarray
        Clipped array (float dtype).
    """
    return np.clip(np.asarray(x, float), eps, 1.0 - eps)


def _ensure_dir(p: str) -> None:
    """Create directory `p` if missing (idempotent)."""
    os.makedirs(p, exist_ok=True)


def _atomic_append_csv(path: str, df: Optional[pd.DataFrame]) -> None:
    """Append a DataFrame to CSV, writing a header only if the file is new.

    Parameters
    ----------
    path : str
        Destination CSV path.
    df : pd.DataFrame or None
        Frame to append. If None or empty, only ensure header exists.
    """
    header = not os.path.exists(path)
    if df is None or df.shape[0] == 0:
        # Create just the header if needed.
        if header:
            pd.DataFrame(columns=getattr(df, "columns", [])).to_csv(path, index=False)
        return
    df.to_csv(path, mode="a", header=header, index=False)


def _dt_days(dates_like: List[pd.Timestamp]) -> np.ndarray:
    """Compute day deltas for a list of dates, flooring any <1 gaps to 1.

    Notes
    -----
    Currently unused; kept for parity with other stages and potential extensions.
    """
    if len(dates_like) == 0:
        return np.array([], float)
    d = pd.to_datetime(pd.Series(dates_like)).values.astype("datetime64[D]").astype("int64")
    dt = np.diff(np.insert(d, 0, d[0])).astype(float)
    dt[dt < 1] = 1.0
    return dt


# ------------------------ IO ------------------------
def _read_feature_table(ctx, pri: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Load the SNV feature store from canonical pipeline locations.

    Search order (first existing is used):
    1) results/preprocessing/tables/feature_store_snv.csv
    2) results/preprocessing/tables/feature_store_snv.cleaned.csv
    3) results/preprocessing/tables/snv_long.csv
    4) results/preprocessing/tables/feature_store_snv_long.csv
    5) {data_root}/jahn_like.csv

    Ensures required columns and basic hygiene on types/ranges.

    Returns
    -------
    df : pd.DataFrame
        Cleaned, sorted table with columns
        [site_id, date, mutation, count, coverage, af].
    all_mutations : List[str]
        Sorted unique mutation identifiers.
    """
    data_root = pri.get("data_root", "data")
    results_root = pri.get("results_root", "results")
    preproc_tables = os.path.join(results_root, "preprocessing", "tables")

    candidates = [
        os.path.join(preproc_tables, "feature_store_snv.csv"),
        os.path.join(preproc_tables, "feature_store_snv.cleaned.csv"),
        os.path.join(preproc_tables, "snv_long.csv"),
        os.path.join(preproc_tables, "feature_store_snv_long.csv"),
        os.path.join(data_root, "jahn_like.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "SNV table not found; expected results/preprocessing/tables/feature_store_snv*.csv "
            "or data/jahn_like.csv"
        )

    ctx.log(level="INFO", message="Loaded SNV feature store", context={"path": path})
    df = pd.read_csv(path)

    required = ["site_id", "date", "mutation", "count", "coverage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Types / ordering / minimal hygiene
    df["site_id"] = df["site_id"].astype(str)
    df["mutation"] = df["mutation"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0).astype(int)

    # Coverage is at least 1 to avoid division-by-zero; count ≤ coverage.
    df.loc[df["coverage"] < 1, "coverage"] = 1
    df.loc[df["count"] > df["coverage"], "count"] = df["coverage"]

    # Raw allele frequency (diagnostic only; model fits on counts)
    df["af"] = (df["count"] / df["coverage"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    df = df.sort_values(["site_id", "mutation", "date"]).reset_index(drop=True)
    all_mutations = sorted(df["mutation"].unique())
    return df, all_mutations


def _prep_daily_for_mutation(df_m: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Aggregate counts across sites by day for a single mutation.

    Parameters
    ----------
    df_m : pd.DataFrame
        Rows for one mutation across sites and dates.

    Returns
    -------
    Y : np.ndarray
        Daily total successes (mutant counts).
    N : np.ndarray
        Daily total trials (coverage).
    dates : list[pd.Timestamp]
        The day labels (sorted).
    """
    g = (
        df_m.groupby("date", as_index=False)[["count", "coverage"]]
        .sum()
        .sort_values("date")
    )
    Y = g["count"].to_numpy(int)
    N = g["coverage"].to_numpy(int)
    dates = g["date"].to_list()
    return Y, N, dates


# ------------------------ Penalized logistic Binomial (RW2) ------------------------
def _make_rw2_penalty(T: int) -> np.ndarray:
    """Construct the RW2 penalty R = DᵀD for a T-length sequence a[0..T-1].

    The second-difference operator D maps a → d where
        d_i = a_i - 2 a_{i+1} + a_{i+2},  i=0..T-3

    Returns
    -------
    R : (T,T) ndarray
        Positive semidefinite penalty matrix (zeros if T ≤ 2).
    """
    if T <= 2:
        return np.zeros((T, T), float)
    D = np.zeros((T - 2, T), float)
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D


def _fit_logistic_rw2(
    Y: np.ndarray,
    N: np.ndarray,
    lam: float = 10.0,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve for a_t = logit(μ_t) with RW2 penalty via damped Newton.

    Objective (convex):
        min_a  Σ_t [ N_t log(1 + exp(a_t)) - Y_t a_t ] + (λ/2) aᵀ R a

    Parameters
    ----------
    Y, N : arrays
        Day-wise successes and trials (N is clipped to ≥ 1).
    lam : float
        RW2 regularization strength λ.
    max_iter : int
        Newton iterations.
    tol : float
        Infinity-norm step tolerance for early stop.

    Returns
    -------
    a : np.ndarray
        MAP on logit scale.
    mu : np.ndarray
        MAP on probability scale: expit(a).
    lo, hi : np.ndarray
        Approximate 95% pointwise intervals for μ via a diagonal Hessian inverse
        approximation and delta method (conservative when λR dominates).

    Notes
    -----
    - Hessian: diag(N μ (1-μ)) + λR, positive semidefinite; we add εI to guarantee PD.
    - If linear solve fails (rare), we fall back to a safe coordinate step.
    """
    T = len(Y)
    Y = np.asarray(Y, float)
    N = np.maximum(np.asarray(N, float), 1.0)
    R = _make_rw2_penalty(T)

    # Initialize from smoothed proportions (add-0.5/1.0)
    p0 = _clip01((Y + 0.5) / (N + 1.0), 1e-6)
    a = logit(p0)

    for _ in range(max_iter):
        mu = expit(a)
        # Gradient/Hessian of data term
        g_data = N * mu - Y
        H_diag = N * mu * (1.0 - mu)  # ≥ 0

        # Add penalty
        g = g_data + lam * (R @ a)
        H = np.diag(H_diag) + lam * R

        # Small jitter on the diagonal to keep PD and help conditioning
        H[np.diag_indices_from(H)] += 1e-8

        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Robust fallback (coordinate-like step)
            step = -g / np.maximum(H_diag + lam * 1e-6, 1e-6)

        a_new = a + step
        if np.max(np.abs(step)) < tol:
            a = a_new
            break
        a = a_new

    mu = _clip01(expit(a), 1e-6)

    # Diagonal approximation for Var[a]: 1 / diag(H)
    H_diag_total = np.diag(H)
    se_a = 1.0 / np.sqrt(np.maximum(H_diag_total, 1e-8))
    z = 1.96
    lo = expit(a - z * se_a)
    hi = expit(a + z * se_a)
    return a, mu, lo, hi


# ------------------------ κ MLE on log-scale ------------------------
def _kappa_mle_global(
    Y_rows: np.ndarray,
    N_rows: np.ndarray,
    t_idx: np.ndarray,
    mu_day: np.ndarray,
    kappa_lo: float = 0.05,
    kappa_hi: float = 100.0,
) -> Tuple[float, Tuple[float, float]]:
    """MLE for a single κ per mutation, given day-wise μ_t.

    We maximize Σ_i log BetaBinom(y_i | n_i, α=μ_{t(i)} κ, β=(1-μ_{t(i)}) κ)
    over log κ ∈ [log kappa_lo, log kappa_hi] using bounded scalar search.

    Parameters
    ----------
    Y_rows, N_rows : arrays
        Row-level counts and coverages.
    t_idx : array[int]
        Day index (0-based) per row mapping into mu_day.
    mu_day : array
        Day-wise μ_t (probabilities) for this mutation.
    kappa_lo, kappa_hi : float
        Strictly positive bounds for κ.

    Returns
    -------
    k_hat : float
        MLE of κ.
    (lo, hi) : Tuple[float, float]
        Approximate 95% CI from curvature of the 1D objective at the optimum.

    Notes
    -----
    - The curvature-based CI is robust in practice; a conservative fallback is used
      if curvature is ill-estimated.
    """
    mu_day = _clip01(mu_day, 1e-6)
    Y = np.asarray(Y_rows, int)
    N = np.asarray(N_rows, int)
    tg = np.asarray(t_idx, int)

    def neg_ll(u: float) -> float:
        k = float(np.exp(u))
        mu = mu_day[tg]
        a = np.maximum(mu * k, 1e-8)
        b = np.maximum((1.0 - mu) * k, 1e-8)
        val = -np.sum(betabinom.logpmf(Y, N, a, b))
        if not np.isfinite(val):
            return 1e50
        return val

    res = minimize_scalar(neg_ll, bounds=(np.log(kappa_lo), np.log(kappa_hi)), method="bounded")
    u_hat = float(res.x) if res.success else np.log(np.sqrt(kappa_lo * kappa_hi))
    k_hat = float(np.exp(u_hat))

    # Numerical curvature for CI on log κ
    h = 1e-3
    fpp = (neg_ll(u_hat + h) - 2.0 * neg_ll(u_hat) + neg_ll(u_hat - h)) / (h * h)
    if np.isfinite(fpp) and fpp > 1e-8:
        se_u = 1.0 / np.sqrt(fpp)
        z = 1.96
        lo = float(np.exp(u_hat - z * se_u))
        hi = float(np.exp(u_hat + z * se_u))
    else:
        lo, hi = max(kappa_lo, k_hat / 3.0), min(kappa_hi, k_hat * 3.0)

    return k_hat, (lo, hi)


# ------------------------ summarize & write ------------------------
def _summarize_and_write(
    ctx,
    mutation: str,
    df_m: pd.DataFrame,
    lam_rw2: float,
    kappa_bounds: Tuple[float, float],
    outdir: str,
) -> Optional[Tuple[float, float]]:
    """Fit μ_t (daily) and κ̂ (global) for one mutation, then write all tables.

    Side effects
    ------------
    Appends to:
      - priors_full_detail.csv (row-level)
      - priors_hyperparams.csv (per-mutation)
      - detail_global_timeseries.csv (per-day)

    Returns
    -------
    (a_med, b_med) : Optional[Tuple[float, float]]
        Median a=logit(μ_t) and b=log κ across the mutation's days, for EB pooling.
        Returns None if the mutation has zero daily rows.
    """
    # 1) Daily global μ_t by aggregated Binomial with RW2 smoothing
    Yd, Nd, dates = _prep_daily_for_mutation(df_m)
    if len(Yd) == 0:
        return None

    a_t, mu_t, mu_lo_t, mu_hi_t = _fit_logistic_rw2(Yd, Nd, lam=lam_rw2)

    # 2) Global κ̂ via MLE across all rows for this mutation
    day_index = {d: i for i, d in enumerate(dates)}
    tg = df_m["date"].map(day_index).to_numpy(int)
    k_lo, k_hi = kappa_bounds
    k_hat, (k_lo_hat, k_hi_hat) = _kappa_mle_global(
        df_m["count"].to_numpy(int),
        df_m["coverage"].to_numpy(int),
        tg,
        mu_t,
        kappa_lo=k_lo,
        kappa_hi=k_hi,
    )

    # ---------- Row-level table (maps each row to its day's μ_t; κ is constant) ----------
    mu_map = mu_t[tg]
    mu_lo_map = mu_lo_t[tg]
    mu_hi_map = mu_hi_t[tg]
    out_detail = pd.DataFrame(
        {
            "site_id": df_m["site_id"].astype(str).values,
            "date": pd.to_datetime(df_m["date"].values),
            "mutation": mutation,
            "count": df_m["count"].astype(int).values,
            "coverage": df_m["coverage"].astype(int).values,
            "af": (df_m["count"] / np.maximum(df_m["coverage"], 1)).astype(float).values,
            "mu_t": mu_map.astype(float),
            "kappa_t": np.full(len(df_m), float(k_hat)),
            "mu_lo": mu_lo_map.astype(float),
            "mu_hi": mu_hi_map.astype(float),
            "kappa_lo": np.full(len(df_m), float(k_lo_hat)),
            "kappa_hi": np.full(len(df_m), float(k_hi_hat)),
        }
    )
    _atomic_append_csv(os.path.join(outdir, "priors_full_detail.csv"), out_detail)

    # ---------- Per-mutation hyperparams (EB on Beta, via μ̂ median and κ̂) ----------
    mu0 = float(np.median(mu_map)) if len(mu_map) else 0.5
    k0 = float(k_hat)
    _atomic_append_csv(
        os.path.join(outdir, "priors_hyperparams.csv"),
        pd.DataFrame(
            [
                {
                    "mutation": mutation,
                    "mu": mu0,
                    "kappa": k0,
                    "alpha": mu0 * k0,
                    "beta": (1.0 - mu0) * k0,
                }
            ]
        ),
    )

    # ---------- Per-day global summary (μ_t, κ̂) on (a=logit μ, b=log κ) ----------
    gday = (
        out_detail.groupby("date", as_index=False)
        .agg(mu_t=("mu_t", "mean"), kappa_t=("kappa_t", "mean"), n_sites=("site_id", "count"))
        .sort_values("date")
    )
    gday["site_id"] = GLOBAL_SITE_ID
    gday["mutation"] = mutation
    gday["a_t"] = logit(_clip01(gday["mu_t"], 1e-9)).astype(float)
    gday["b_t"] = np.log(np.maximum(gday["kappa_t"], EPS)).astype(float)

    # Columns expected by downstream; intervals (mu_lo/hi, kappa_lo/hi) not repeated here.
    gday[["var_a", "var_b", "cov_ab", "mu_lo", "mu_hi", "kappa_lo", "kappa_hi", "q_LL_used", "q_b_used"]] = np.nan
    _atomic_append_csv(
        os.path.join(outdir, "detail_global_timeseries.csv"),
        gday[
            [
                "site_id",
                "date",
                "mutation",
                "mu_t",
                "kappa_t",
                "a_t",
                "b_t",
                "var_a",
                "var_b",
                "cov_ab",
                "n_sites",
                "q_LL_used",
                "q_b_used",
                "mu_lo",
                "mu_hi",
                "kappa_lo",
                "kappa_hi",
            ]
        ],
    )

    # Provide medians on (a, b) for EB pooling across mutations.
    return float(np.median(gday["a_t"])), float(np.median(gday["b_t"]))


# ------------------------ pipeline ------------------------
@dataclass
class SimpleCtx:
    """Minimal logging shim used by the pipeline runner.

    Methods
    -------
    log(level="INFO", message="", context=None, stage="priors", site_id=None, lineage=None)
        Prints a single JSON record (one line), suitable for ingestion by your
        existing log readers.
    """

    def log(
        self,
        level: str = "INFO",
        message: str = "",
        context: Optional[dict] = None,
        stage: str = "priors",
        site_id: Optional[str] = None,
        lineage: Optional[str] = None,
    ) -> None:
        rec = {
            "time": pd.Timestamp.utcnow().isoformat(),
            "level": level,
            "stage": stage,
            "site_id": site_id,
            "lineage": lineage,
            "message": message,
            "context": context or {},
        }
        print(json.dumps(rec))


def run_priors(cfg: Dict, ctx) -> Dict:
    """End-to-end prior construction (deterministic)

    Steps
    -----
    1) Load counts table and (optionally) cap coverage at a high quantile.
    2) For each mutation with ≥ `min_dates` unique days:
         a) Fit daily μ_t with RW2 logistic smoothing.
         b) Fit a single κ̂ by MLE given μ_t.
         c) Write row-level detail, per-mutation hyperparams, and per-day summary.
    3) Fit an EB Gaussian prior over (a=logit μ, b=log κ) across mutations.

    Parameters
    ----------
    cfg : dict
        Full config; either contains a top-level "priors" mapping or is itself
        the priors mapping.
    ctx : object
        Must implement `ctx.log(...)` for structured JSON logging.

    Returns
    -------
    dict
        {"tables": ["priors_full_detail", "priors_hyperparams",
                    "detail_global_timeseries", "eb_population_prior"]}
    """
    pri = cfg.get("priors", cfg)
    results_root = pri.get("results_root", "results")
    outdir = os.path.join(results_root, "priors")
    _ensure_dir(outdir)

    # Pre-create empty CSVs with headers to make downstream robust to empty stages.
    for name, cols in [
        (
            "priors_full_detail.csv",
            ["site_id", "date", "mutation", "count", "coverage", "af", "mu_t", "kappa_t", "mu_lo", "mu_hi", "kappa_lo", "kappa_hi"],
        ),
        ("priors_hyperparams.csv", ["mutation", "mu", "kappa", "alpha", "beta"]),
        (
            "detail_global_timeseries.csv",
            [
                "site_id",
                "date",
                "mutation",
                "mu_t",
                "kappa_t",
                "a_t",
                "b_t",
                "var_a",
                "var_b",
                "cov_ab",
                "n_sites",
                "q_LL_used",
                "q_b_used",
                "mu_lo",
                "mu_hi",
                "kappa_lo",
                "kappa_hi",
            ],
        ),
        ("eb_population_prior.csv", ["m_a", "m_b", "Saa", "Sab", "Sbb"]),
    ]:
        p = os.path.join(outdir, name)
        if not os.path.exists(p):
            pd.DataFrame(columns=cols).to_csv(p, index=False)

    # ---- Load data
    ctx.log(level="INFO", message="Loading counts")
    df, all_mutations = _read_feature_table(ctx, pri)

    # Optional coverage cap to reduce leverage of extreme depths
    cap_q = float(pri.get("cap_coverage_quantile", 0.99))
    if 0.5 <= cap_q < 1.0 and not df.empty:
        cap_val = float(df["coverage"].quantile(cap_q))
        if np.isfinite(cap_val) and cap_val > 0:
            df["coverage"] = np.minimum(df["coverage"], int(cap_val))

    # Hyperparameters
    lam_rw2 = float(pri.get("rw2_penalty", 10.0))
    kappa_lo = float(pri.get("kappa_lo", 0.05))
    kappa_hi = float(pri.get("kappa_hi", 100.0))
    min_days = int(pri.get("min_dates", 2))

    # ---- Per-mutation loop
    eb_rows: List[Dict[str, float]] = []
    for mut in all_mutations:
        df_m = df[df["mutation"] == mut].sort_values(["date", "site_id"]).reset_index(drop=True)
        days = df_m["date"].nunique()
        if days < min_days:
            ctx.log(level="WARN", message=f"Skipping {mut} (insufficient days)", context={"days": int(days)})
            continue

        ctx.log(level="INFO", message=f"Fitting closed-form priors for mutation {mut}")
        out = _summarize_and_write(ctx, mut, df_m, lam_rw2, (kappa_lo, kappa_hi), outdir)
        if out is not None:
            a_med, b_med = out
            eb_rows.append({"mutation": mut, "a_map": a_med, "b_map": b_med})

    # ---- EB Gaussian across mutations on (a, b)
    if eb_rows:
        eb = pd.DataFrame(eb_rows).sort_values("mutation")
        a = eb["a_map"].to_numpy(float)
        b = eb["b_map"].to_numpy(float)
        m_a, m_b = float(np.mean(a)), float(np.mean(b))
        S = np.cov(np.vstack([a, b]))
        Saa = float(S[0, 0]) if np.isfinite(S[0, 0]) else 1.0
        Sab = float(S[0, 1]) if np.isfinite(S[0, 1]) else 0.0
        Sbb = float(S[1, 1]) if np.isfinite(S[1, 1]) else 1.0
        _atomic_append_csv(
            os.path.join(outdir, "eb_population_prior.csv"),
            pd.DataFrame([{"m_a": m_a, "m_b": m_b, "Saa": Saa, "Sab": Sab, "Sbb": Sbb}]),
        )

    ctx.log(level="INFO", message="Done", context={"outdir": outdir})
    return {
        "tables": [
            "priors_full_detail",
            "priors_hyperparams",
            "detail_global_timeseries",
            "eb_population_prior",
        ]
    }


# ------------------------ CLI ------------------------
def _load_config(path: Optional[str]) -> Dict:
    """Load YAML/JSON config from `path`. If PyYAML is missing, fall back to JSON.

    Returns an empty dict if `path` is None.
    """
    if path is None:
        return {}
    txt = open(path, "r", encoding="utf-8").read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        try:
            return json.loads(txt)
        except Exception:
            raise RuntimeError("Provide YAML/JSON config or install PyYAML.")


def main() -> None:
    """CLI entry point. Example:

        python -m priors_closed_form --config config.yaml

    Where config.yaml may contain either just priors keys or a top-level
    mapping with a 'priors' block.
    """
    ap = argparse.ArgumentParser(description="Oxford Bio Priors (closed-form, stable)")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config (uses cfg['priors'] if present).",
    )
    args = ap.parse_args()

    cfg: Dict = _load_config(args.config) if args.config else {}
    ctx = SimpleCtx()
    out = run_priors(cfg, ctx)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
