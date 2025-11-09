#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrated Priors (Swiss wastewater) — ZIBB + RW2

From first principles:
  • Zero-Inflated Beta–Binomial (ZIBB) observation model per row:
       With prob π_b (bin by μ), Y = 0 (structural zero), else Y ~ Beta-Binomial(n, α=μ κ, β=(1-μ) κ).
  • μ_t (per mutation/day) by penalized logistic RW2 with correct uncertainty (diag(inv(H))).
  • EM algorithm:
      E-step: responsibilities r_i = P(struct-zero | data).
      M-step: weighted RW2 for μ_t using weights (1 - r_i); weighted κ(μ) via moment match; update π per μ-decile.
  • κ(μ): per-μ-decile moment match (weighted) → log-space interpolation.
  • Per-μ-decile predictive coverage calibration τ_b (mixture) to make 50% / 90% intervals nominal (±1%) within each decile.

Outputs (results/priors/):
  - priors_full_detail.csv           (row-level μ̃, μ̃_lo/hi, κ̃, τ, π, mixture PIs: y_lo/hi_50/90, p_lo/hi_50/90)
  - priors_hyperparams.csv           (per-mutation μ, κ, alpha, beta)
  - detail_global_timeseries.csv     (GLOBAL per-day μ̃_t, κ̃_t, a_t, b_t, μ-bounds)
  - eb_population_prior.csv          (EB Gaussian on (a=logit μ̃, b=log κ̃))

Config (priors: ...), tuned defaults:
  results_root: "results"
  data_root: "data"
  cap_coverage_quantile: 0.99
  rw2_penalty: 10.0
  kappa_lo: 1e-6
  kappa_hi: 150.0
  min_dates: 2
  em_iters: 6
  use_mu_calibration: true
  calibration_bins: 40
  mu_floor_c: 5.0
  target_coverages: [0.50, 0.90]
  coverage_tolerance: 0.01
  tau_bounds: [0.1, 60.0]
  tau_bisect_max_iter: 18
  min_rows_per_bin: 200
"""
# add this with your other scipy.special imports
from __future__ import annotations
import os, glob, json, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import betabinom
from scipy.special import logsumexp


# -------------------- Constants --------------------
GLOBAL_SITE_ID = "GLOBAL"
EPS = 1e-9


# -------------------- Utilities --------------------
def _clip01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(np.asarray(x, float), eps, 1.0 - eps)

def _read_csv_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -------------------- Logger --------------------
@dataclass
class SimpleCtx:
    def log(self, level: str="INFO", message: str="", context: dict|None=None) -> None:
        print(json.dumps({
            "time": pd.Timestamp.utcnow().isoformat(),
            "level": level, "stage": "priors",
            "message": message, "context": context or {}
        }))


# -------------------- IO --------------------
def _read_feature_table(ctx: SimpleCtx, pri: Dict) -> tuple[pd.DataFrame, List[str]]:
    results_root = pri.get("results_root", "results")
    data_root    = pri.get("data_root", "data")
    candidates = sorted(glob.glob(os.path.join(results_root, "preprocessing", "tables", "feature_store_snv*.csv")))
    if candidates:
        path = candidates[0]
    else:
        path = os.path.join(data_root, "jahn_like.csv")
        if not os.path.exists(path):
            raise FileNotFoundError("SNV table not found; expected results/.../feature_store_snv*.csv or data/jahn_like.csv")
    ctx.log("INFO", "Loaded SNV feature store", {"path": path})
    df = _read_csv_any(path)

    req = ["site_id", "date", "mutation", "count", "coverage"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    df["site_id"]  = df["site_id"].astype(str)
    df["mutation"] = df["mutation"].astype(str)
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"]     = df["date"].dt.normalize()
    df["count"]    = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0).astype(int)

    df.loc[df["coverage"] < 1, "coverage"] = 1
    df.loc[df["count"] > df["coverage"], "count"] = df["coverage"]
    df["af"] = (df["count"] / df["coverage"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    return df.sort_values(["site_id","mutation","date"]).reset_index(drop=True), sorted(df["mutation"].unique())


# -------------------- RW2 μ_t with correct uncertainty --------------------
def _make_rw2_penalty(T: int) -> np.ndarray:
    if T <= 2: return np.zeros((T,T))
    D = np.zeros((T-2, T))
    for i in range(T-2):
        D[i, i]   = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    return D.T @ D
def _fit_logistic_rw2(
    Y: np.ndarray,
    N: np.ndarray,
    lam: float = 10.0,
    max_iter: int = 60,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Penalized logistic RW2:
      minimizes   L(a) = sum_i [ -Y_i*a_i + N_i*softplus(a_i) ] + 0.5*lam_eff * a^T R a
      with softplus(x) = log(1 + exp(x)) in a numerically stable form.

    Returns:
      a       : logit(μ_t) at optimum
      mu      : μ_t
      mu_lo   : μ_t 95% lower (via diag(inv(H)) on a, mapped through logistic)
      mu_hi   : μ_t 95% upper
      var_a   : diag(inv(H)) for a_t
    """
    Y = np.asarray(Y, float)
    N = np.maximum(np.asarray(N, float), 1.0)
    T = len(Y)

    R = _make_rw2_penalty(T)
    # Mild scaling by coverage so extremely deep days do not dominate curvature
    lam_eff = float(lam / (1.0 + (np.mean(N) / 5e5)))

    # Stable initial values (add 0.5 / 1.0 to reduce separation)
    def _clip01s(x): return np.clip(x, 1e-8, 1 - 1e-8)
    mu0 = _clip01s((Y + 0.5) / (N + 1.0))
    a = logit(mu0)

    # Stable softplus and objective
    def _softplus(x):
        # log(1+exp(x)) computed stably
        # = max(x,0) + log1p(exp(-|x|))
        m = np.maximum(x, 0.0)
        return m + np.log1p(np.exp(-np.abs(x)))

    def _obj(a_):
        return float(np.sum(N * _softplus(a_) - Y * a_) + 0.5 * lam_eff * (a_ @ (R @ a_)))

    # Newton with backtracking line search (Armijo)
    f_prev = _obj(a)
    for _ in range(max_iter):
        mu = expit(a)
        g = N * mu - Y + lam_eff * (R @ a)
        Hd = N * mu * (1.0 - mu)
        H = np.diag(Hd) + lam_eff * R
        # diagonal inflation to guarantee SPD
        H[np.diag_indices_from(H)] += 1e-8

        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Fallback: diagonally preconditioned gradient step
            step = -g / np.maximum(np.diag(H), 1e-8)

        # Armijo backtracking
        t = 1.0
        c1 = 1e-4
        gdotp = float(g @ step)
        a_new = a + t * step
        f_new = _obj(a_new)
        while f_new > f_prev + c1 * t * gdotp and t > 1e-6:
            t *= 0.5
            a_new = a + t * step
            f_new = _obj(a_new)

        a = a_new
        if np.max(np.abs(t * step)) < tol:
            f_prev = f_new
            break
        f_prev = f_new

    # Final quantities
    mu = _clip01(expit(a), 1e-12)

    # Correct uncertainty: diag(inv(H))
    try:
        H_inv = np.linalg.inv(H + 1e-12 * np.eye(T))
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H + 1e-12 * np.eye(T))
    var_a = np.clip(np.diag(H_inv), 1e-12, np.inf)
    se_a = np.sqrt(var_a)

    z = 1.96
    mu_lo = _clip01(expit(a - z * se_a), 1e-12)
    mu_hi = _clip01(expit(a + z * se_a), 1e-12)

    return a, mu, mu_lo, mu_hi, var_a


def _prep_daily_for_mutation(df_m: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,List[pd.Timestamp]]:
    g = df_m.groupby("date", as_index=False)[["count","coverage"]].sum().sort_values("date")
    return g["count"].to_numpy(int), g["coverage"].to_numpy(int), g["date"].to_list()


# -------------------- μ calibration (isotonic) --------------------
def _bin_by_mu(mu: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, float)
    if mu.size == 0: return np.zeros(0,int), np.array([0.0,1.0])
    qs    = np.linspace(0.0, 1.0, K+1)
    edges = np.quantile(mu, qs)
    if np.unique(edges).size < 3:
        edges = np.linspace(0.0, 1.0, K+1)
    idx = np.digitize(mu, edges[1:-1], right=True)
    return np.clip(idx, 0, len(edges)-2), edges

def _pav_isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    order = np.argsort(x); x, y, w = x[order], y[order], w[order]
    y_hat, w_hat = y.copy(), w.copy()
    blocks = [[i] for i in range(len(y_hat))]
    i=0
    while i < len(y_hat)-1:
        if y_hat[i] <= y_hat[i+1] + 1e-15:
            i += 1; continue
        new_w = w_hat[i] + w_hat[i+1]
        new_y = (y_hat[i]*w_hat[i] + y_hat[i+1]*w_hat[i+1]) / max(new_w,1e-12)
        y_hat[i], w_hat[i] = new_y, new_w
        blocks[i] += blocks[i+1]
        y_hat = np.delete(y_hat, i+1)
        w_hat = np.delete(w_hat, i+1)
        del blocks[i+1]
        if i>0: i -= 1
    x_rep = np.array([x[b].mean() for b in blocks], float)
    return x_rep, y_hat

def _build_calibrator(mu_pred: np.ndarray, y: np.ndarray, n: np.ndarray, K: int):
    if len(mu_pred) < 50: return None
    bins,_ = _bin_by_mu(mu_pred, K)
    df = pd.DataFrame({"bin": bins, "mu": mu_pred, "y": y, "n": n})
    g = df.groupby("bin", as_index=False).apply(
        lambda d: pd.Series({
            "pred_mu_mean": float(np.average(d["mu"], weights=np.maximum(d["n"],1))) if d["n"].sum()>0 else float(d["mu"].mean()),
            "obs_rate": float(d["y"].sum() / max(d["n"].sum(), 1)),
            "sum_n": int(d["n"].sum())
        })
    ).dropna()
    if g.shape[0] < 3: return None
    x_iso, y_iso = _pav_isotonic(g["pred_mu_mean"].to_numpy(float),
                                 g["obs_rate"].to_numpy(float),
                                 w=np.maximum(g["sum_n"].to_numpy(float),1.0))
    xk, yk = [], []
    for xi, yi in zip(x_iso, y_iso):
        if len(xk) and abs(xk[-1]-xi) < 1e-12: yk[-1] = yi
        else: xk.append(float(np.clip(xi,0,1))); yk.append(float(np.clip(yi,1e-12,1-1e-12)))
    if xk[0] > 0: xk.insert(0,0.0); yk.insert(0,yk[0])
    if xk[-1] < 1: xk.append(1.0); yk.append(yk[-1])
    return np.array(xk), np.array(yk)

def _apply_calibrator(mu: np.ndarray, knots) -> np.ndarray:
    if knots is None: return _clip01(mu)
    xk, yk = knots
    return _clip01(np.interp(mu, xk, yk))


# -------------------- κ(μ): weighted moment match + log-space interpolation --------------------


def _weighted_var(values: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, float); v = np.asarray(values, float)
    w = np.clip(w, 0.0, np.inf)
    if w.sum() <= 0: return 0.0
    m  = np.sum(w*v) / w.sum()
    return float(np.sum(w*(v-m)**2) / w.sum())

def _kappa_by_decile_moments_weighted(mu: np.ndarray, y: np.ndarray, n: np.ndarray,
                                      w: np.ndarray, K: int, k_lo: float, k_hi: float) -> tuple[np.ndarray,np.ndarray]:
    mu = _clip01(mu, 1e-12)
    p  = y / np.maximum(n, 1)
    idx, edges = _bin_by_mu(mu, K)
    kappas = np.full(edges.size-1, k_hi, float)
    for b in range(edges.size-1):
        m  = (idx == b)
        if np.sum(m) < 5: continue
        wb = w[m]
        if wb.sum() <= 0: continue
        mu_b = mu[m]; n_b = np.maximum(n[m], 1); p_b = p[m]
        var_emp = _weighted_var(p_b, wb)
        W_bar   = float(np.sum(wb * (mu_b*(1-mu_b)/n_b)) / wb.sum())
        C_bar   = float(np.sum(wb * (mu_b*(1-mu_b)*(n_b-1)/n_b)) / wb.sum())
        denom = var_emp - W_bar
        k_est = C_bar / denom - 1.0 if denom > 1e-12 else k_hi
        kappas[b] = float(np.clip(k_est, k_lo, k_hi))
    return kappas, edges

def _interp_log_kappa(mu: np.ndarray, edges: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    centers = 0.5*(edges[:-1] + edges[1:])
    logk = np.log(np.clip(kappas, 1e-12, None))
    return np.exp(np.interp(mu, centers, logk, left=logk[0], right=logk[-1]))


# -------------------- Mixture (ZIBB) quantiles & coverage --------------------
def _mix_logprob_y(y: np.ndarray, n: np.ndarray, mu: np.ndarray, kappa: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Log probability for ZIBB mixture, numerically stable."""
    a = _clip01(mu, 1e-12) * np.clip(kappa, 1e-12, 1e12)
    b = (1.0 - _clip01(mu, 1e-12)) * np.clip(kappa, 1e-12, 1e12)
    log_bb = betabinom.logpmf(y, n, a, b)
    log1m_pi = np.log1p(-np.clip(pi, 0.0, 1.0 - 1e-12))
    log_pi   = np.log(np.clip(pi, 1e-12, 1.0 - 1e-12))

    # start with nonzero case: log((1-π)*BB)
    logp = log1m_pi + log_bb

    # y==0: log(π + (1-π)*BB(0)) via log-sum-exp
    z = (y == 0)
    if np.any(z):
        log_bb0 = betabinom.logpmf(np.zeros_like(y[z]), n[z], a[z], b[z])
        comp = np.vstack([log_pi[z], log1m_pi[z] + log_bb0])
        logp[z] = logsumexp(comp, axis=0)
    return logp

def _mixture_pmf_row(n: int, a: float, b: float, pi: float) -> np.ndarray:
    """PMF of mixture: P(Y=0)=pi + (1-pi)*BB(0); P(Y=y>0)=(1-pi)*BB(y)."""
    y = np.arange(n+1, dtype=int)
    bb = betabinom.pmf(y, n, a, b)
    pmf = (1.0 - pi) * bb
    pmf[0] += pi
    return pmf

def _mixture_ppf_row(q: float, n: int, a: float, b: float, pi: float) -> int:
    """
    Exact quantile for ZIBB via a closed-form mapping to the Beta-Binomial PPF:
      F_mix(y) = π + (1-π) * F_BB(y)
      => F_BB(y) >= (q - π) / (1 - π)
    This is both exact and far faster than enumerating PMFs up to n.
    """
    pi = float(np.clip(pi, 0.0, 1.0 - 1e-12))
    if q <= 0.0: 
        return 0
    if q >= 1.0:
        return int(n)

    denom = max(1.0 - pi, 1e-12)
    q_bb = np.clip((q - pi) / denom, 0.0, 1.0)
    # scipy.stats.betabinom.ppf is vectorized; here used scalar
    y = betabinom.ppf(q_bb, int(n), float(a), float(b))
    return int(y)


def _mixture_coverage(y: np.ndarray, n: np.ndarray, mu: np.ndarray, kappa: np.ndarray,
                      pi: np.ndarray, qlo: float, qhi: float) -> float:
    covered = []
    for yi, ni, mui, ki, pii in zip(y, n, mu, kappa, pi):
        a = _clip01(mui, 1e-12) * np.clip(ki, 1e-12, 1e12)
        b = (1.0 - _clip01(mui, 1e-12)) * np.clip(ki, 1e-12, 1e12)
        lo = _mixture_ppf_row(qlo, int(ni), a, b, float(pii))
        hi = _mixture_ppf_row(qhi, int(ni), a, b, float(pii))
        covered.append(1.0 if (yi >= lo and yi <= hi) else 0.0)
    return float(np.mean(covered))


# -------------------- Per-μ-bin τ calibration (mixture) --------------------
def _calibrate_tau_per_bin_mixture(
    y: np.ndarray, n: np.ndarray, mu: np.ndarray, kappa: np.ndarray, pi: np.ndarray,
    edges: np.ndarray, targets: List[float],
    bounds: tuple[float,float]=(1.0,200.0), max_iter: int=18, tol: float=0.01, min_rows_per_bin: int=200
) -> np.ndarray:
    """
    Per-μ-bin τ calibration for ZIBB mixture that ONLY WIDENS (τ >= 1).
    This fixes 90% under-coverage without creating razor-thin tails.
    """
    K = edges.size - 1
    idx,_ = _bin_by_mu(mu, K)
    tau = np.ones(K, float)

    lo = max(1.0, float(bounds[0]))     # enforce widen-only
    hi = max(lo * 1.0001, float(bounds[1]))

    def worst_gap_subset(mask, tau_):
        kb = kappa[mask] / max(tau_, 1.0)  # τ widens → κ' = κ/τ ≤ κ
        gaps=[]
        for t in targets:
            qlo = (1.0 - t)/2.0; qhi = (1.0 + t)/2.0
            gaps.append(_mixture_coverage(y[mask], n[mask], mu[mask], kb, pi[mask], qlo, qhi) - t)
        return min(gaps)  # the worst target gap (negative means under-coverage)

    for b in range(K):
        m = (idx==b)
        if int(np.sum(m)) < min_rows_per_bin:
            tau[b]=1.0
            continue

        g_lo = worst_gap_subset(m, lo)
        g_hi = worst_gap_subset(m, hi)

        # If already over-covered at lo, keep it small
        if g_lo >= -tol:
            tau[b] = lo
            continue
        # If still under-covered at hi, push to hi
        if g_hi <= -tol:
            tau[b] = hi
            continue

        left,right = lo,hi
        for _ in range(max_iter):
            mid = float(np.sqrt(left*right))
            g_mid = worst_gap_subset(m, mid)
            if abs(g_mid) <= tol:
                left = right = mid
                break
            if g_mid < 0:   # under-covered → widen more
                left = mid
            else:
                right = mid
        tau[b] = float(np.sqrt(left*right))
    return np.clip(tau, lo, hi)
def _em_zibb_fit_per_mutation(
    ctx: SimpleCtx, mutation: str, df_m: pd.DataFrame,
    lam_rw2: float, kappa_bounds: tuple[float,float], K_bins: int, mu_floor_c: float,
    use_mu_calibration: bool, em_iters: int, min_rows_per_bin: int
) -> dict:
    # Aggregate daily
    Yd, Nd, dates = _prep_daily_for_mutation(df_m)
    a_t, mu_t, mu_t_lo, mu_t_hi, var_a_t = _fit_logistic_rw2(Yd, Nd, lam=lam_rw2)

    # Map rows to days
    day_index = {d:i for i,d in enumerate(dates)}
    tg = df_m["date"].map(day_index).to_numpy(int)
    y  = df_m["count"].to_numpy(int)
    n  = df_m["coverage"].to_numpy(int)
    y  = np.clip(y, 0, None)  # safety

    # Initial μ rows
    mu_rows = mu_t[tg].copy()

    # Gentle μ floor (cap at 5%)
    if mu_floor_c > 0:
        floor_rows = np.minimum(mu_floor_c / np.maximum(n, 1), 0.05)
        mu_rows = np.maximum(mu_rows, floor_rows)
        mu_t    = np.maximum(mu_t, np.minimum(mu_floor_c / np.maximum(Nd,1), 0.05))

    # μ calibration (optional)
    mu_knots = _build_calibrator(mu_rows, y, n, K_bins) if use_mu_calibration else None
    def _cal(v): return _apply_calibrator(v, mu_knots)
    mu_rows = _cal(mu_rows)
    mu_t    = _cal(mu_t)

    # κ(μ) init (weighted moments)
    k_lo, k_hi = kappa_bounds
    kappas_bin, edges = _kappa_by_decile_moments_weighted(mu_rows, y, n, np.ones_like(y), K_bins, k_lo, k_hi)
    kappa_rows = _interp_log_kappa(mu_rows, edges, kappas_bin)

    # π init by zero rate, then gate & shrink
    idx,_ = _bin_by_mu(mu_rows, K_bins)
    pi_bins = np.zeros(K_bins, float)
    centers = 0.5*(edges[:-1] + edges[1:])
    for b in range(K_bins):
        m = (idx==b)
        if np.sum(m) < min_rows_per_bin:
            pi_bins[b] = 0.0
            continue
        if centers[b] > 0.15:
            pi_bins[b] = 0.0
        else:
            zr = float(np.mean(y[m] == 0))
            m_sum = int(np.sum(m))
            pi_bins[b] = float(np.clip((zr*m_sum + 0.5) / (m_sum + 0.5 + 5.0), 0.0, 0.99))

    # EM iterations
    r = np.zeros_like(y, float)  # responsibilities for structural zero
    for _ in range(max(1, em_iters)):
        # --- E-step (log-space) ---
        pi_rows = np.clip(pi_bins[idx], 0.0, 0.99)
        r[:] = 0.0
        z = (y == 0)
        if np.any(z):
            a = _clip01(mu_rows[z], 1e-12) * np.clip(kappa_rows[z], 1e-12, 1e12)
            b = (1.0 - _clip01(mu_rows[z], 1e-12)) * np.clip(kappa_rows[z], 1e-12, 1e12)
            log_pi   = np.log(np.clip(pi_rows[z], 1e-12, 1.0 - 1e-12))
            log1m_pi = np.log1p(-pi_rows[z])
            log_bb0  = betabinom.logpmf(0, n[z], a, b)
            log_den  = logsumexp(np.vstack([log_pi, log1m_pi + log_bb0]), axis=0)
            r[z] = np.exp(log_pi - log_den)

        # --- M-step: weighted RW2 for μ_t ---
        w_eff = (1.0 - r)
        Yd_w = np.bincount(tg, weights=w_eff*y, minlength=len(dates))
        Nd_w = np.bincount(tg, weights=w_eff*n, minlength=len(dates))
        a_t, mu_t, mu_t_lo, mu_t_hi, var_a_t = _fit_logistic_rw2(Yd_w, np.maximum(Nd_w, 1), lam=lam_rw2)

        # Map back to rows + gentle floor + calibration
        mu_rows = mu_t[tg].copy()
        if mu_floor_c > 0:
            floor_rows = np.minimum(mu_floor_c / np.maximum(n, 1), 0.05)
            mu_rows = np.maximum(mu_rows, floor_rows)
            mu_t    = np.maximum(mu_t, np.minimum(mu_floor_c / np.maximum(Nd,1), 0.05))
        mu_rows = _cal(mu_rows)
        mu_t    = _cal(mu_t)

        # Update κ(μ) via weighted moments
        kappas_bin, edges = _kappa_by_decile_moments_weighted(mu_rows, y, n, w_eff, K_bins, k_lo, k_hi)
        kappa_rows = _interp_log_kappa(mu_rows, edges, kappas_bin)

        # Update π per μ-bin with gating + shrinkage
        idx,_ = _bin_by_mu(mu_rows, K_bins)
        centers = 0.5*(edges[:-1] + edges[1:])
        for b in range(K_bins):
            m = (idx==b)
            if np.sum(m) < min_rows_per_bin:
                continue
            if centers[b] > 0.15:
                pi_bins[b] = 0.0
            else:
                r_sum = float(np.sum(r[m]))
                m_sum = int(np.sum(m))
                pi_bins[b] = float(np.clip((r_sum + 0.5) / (m_sum + 0.5 + 5.0), 0.0, 0.99))

    return {
        "mu_rows": mu_rows,
        "mu_t": mu_t, "mu_t_lo": mu_t_lo, "mu_t_hi": mu_t_hi,
        "var_a_t": var_a_t,
        "kappa_rows": kappa_rows, "edges": edges, "pi_bins": pi_bins, "idx": idx, "dates": dates
    }

# -------------------- Per-mutation export (with τ calibration and mixture PIs) --------------------
def _export_mutation(
    ctx: SimpleCtx, mutation: str, df_m: pd.DataFrame, em_fit: dict,
    outdir: str, targets: List[float], tau_bounds: Tuple[float,float],
    tau_bisect_max_iter: int, cov_tol: float, min_rows_per_bin: int
) -> tuple[float, float] | None:
    # --- Inputs & safety ---
    y = df_m["count"].to_numpy(int)
    n = df_m["coverage"].to_numpy(int)
    y = np.clip(y, 0, n)  # enforce invariants (no af>1)
    af = (y / np.maximum(n, 1)).astype(float)

    mu_rows = em_fit["mu_rows"]
    mu_t    = em_fit["mu_t"]
    mu_t_lo = em_fit["mu_t_lo"]
    mu_t_hi = em_fit["mu_t_hi"]
    var_a_t = em_fit.get("var_a_t", np.full_like(mu_t, np.nan, dtype=float))  # support older fits gracefully
    kappa_rows = em_fit["kappa_rows"]
    edges  = em_fit["edges"]
    idx    = em_fit["idx"]
    dates  = em_fit["dates"]

    # --- per-row π from μ-bin assignment ---
    pi_rows = np.zeros_like(mu_rows, dtype=float)
    for b in range(edges.size - 1):
        m = (idx == b)
        if np.any(m):
            pi_rows[m] = float(em_fit["pi_bins"][b])

    # --- τ calibration (widen-only; τ ≥ 1) ---
    tau_lo = max(1.0, float(tau_bounds[0]))
    tau_hi = max(tau_lo * 1.0001, float(tau_bounds[1]))
    tau_bins = _calibrate_tau_per_bin_mixture(
        y=y, n=n, mu=mu_rows, kappa=kappa_rows, pi=pi_rows, edges=edges,
        targets=targets, bounds=(tau_lo, tau_hi), max_iter=tau_bisect_max_iter,
        tol=cov_tol, min_rows_per_bin=min_rows_per_bin
    )
    tau_rows = tau_bins[idx]
    kappa_rows_cal = np.clip(kappa_rows / np.maximum(tau_rows, 1.0), 1e-12, 1e12)

    # --- Map μ credible bands to rows' days ---
    day_index = {d: i for i, d in enumerate(dates)}
    tg = df_m["date"].map(day_index).to_numpy(int)

    # --- Row-level export (μ_t, μ_lo/μ_hi, κ_t (τ-calibrated), π, τ, and mixture PIs) ---
    out = pd.DataFrame({
        "site_id": df_m["site_id"].astype(str).values,
        "date":    pd.to_datetime(df_m["date"].values),
        "mutation": mutation,
        "count":   y,
        "coverage": n,
        "af": af,
        "mu_t":    mu_rows.astype(float),
        "mu_lo":   mu_t_lo[tg].astype(float),
        "mu_hi":   mu_t_hi[tg].astype(float),
        "kappa_t": kappa_rows_cal.astype(float),
        "pi":      np.clip(pi_rows, 0.0, 0.99).astype(float),
        "tau":     tau_rows.astype(float),
    })

    # Mixture predictive bands (50% & 90%) via exact mapping to BB ppf
    for qlo, qhi, tag in [(0.25, 0.75, "50"), (0.05, 0.95, "90")]:
        ylo, yhi, plo, phi = [], [], [], []
        for yi, ni, mui, ki, pii in zip(y, n, mu_rows, kappa_rows_cal, pi_rows):
            a = _clip01(mui, 1e-12) * np.clip(ki, 1e-12, 1e12)
            b = (1.0 - _clip01(mui, 1e-12)) * np.clip(ki, 1e-12, 1e12)
            lo = _mixture_ppf_row(qlo, int(ni), a, b, float(pii))
            hi = _mixture_ppf_row(qhi, int(ni), a, b, float(pii))
            ylo.append(lo); yhi.append(hi)
            denom = max(int(ni), 1)
            plo.append(lo / denom); phi.append(hi / denom)
        out[f"y_lo_{tag}"] = np.asarray(ylo, dtype=int)
        out[f"y_hi_{tag}"] = np.asarray(yhi, dtype=int)
        out[f"p_lo_{tag}"] = np.asarray(plo, dtype=float)
        out[f"p_hi_{tag}"] = np.asarray(phi, dtype=float)

    p_detail = os.path.join(outdir, "priors_full_detail.csv")
    out.to_csv(p_detail, mode="a", header=not os.path.exists(p_detail), index=False)

    # --- Per-mutation hyperparams (medians for robustness) ---
    mu0 = float(np.median(out["mu_t"]))
    k0  = float(np.median(out["kappa_t"]))
    p_hyper = os.path.join(outdir, "priors_hyperparams.csv")
    pd.DataFrame([{
        "mutation": mutation, "mu": mu0, "kappa": k0,
        "alpha": mu0 * k0, "beta": (1.0 - mu0) * k0,
        "mu_shrunk": mu0, "kappa_shrunk": k0
    }]).to_csv(p_hyper, mode="a", header=not os.path.exists(p_hyper), index=False)

    # --- Per-day GLOBAL summary (τ-calibrated κ) ---
    mu_day_bins, _ = _bin_by_mu(mu_t, edges.size - 1)
    tau_days  = tau_bins[mu_day_bins]

    # Smooth per-day κ by interpolating log κ across μ bins using row weights
    # (same approach you had, just kept explicit)
    bin_weights = np.bincount(idx, weights=kappa_rows, minlength=edges.size - 1)
    bin_counts  = np.bincount(idx, minlength=edges.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa_bin_mean = bin_weights / (bin_counts + 1e-9)
    kappa_days = _interp_log_kappa(mu_t, edges, kappa_bin_mean)

    kappa_days_cal = np.clip(kappa_days / np.maximum(tau_days, 1.0), 1e-12, 1e12)
    n_sites_by_day = (
        df_m.groupby("date")["site_id"].nunique()
        .reindex(pd.Index(dates), fill_value=0)
        .to_numpy()
    )

    gday = pd.DataFrame({
        "site_id": GLOBAL_SITE_ID,
        "date": dates,
        "mutation": mutation,
        "mu_t": mu_t.astype(float),
        "kappa_t": kappa_days_cal.astype(float),
        "a_t":  logit(_clip01(mu_t)).astype(float),
        "b_t":  np.log(np.maximum(kappa_days_cal, EPS)).astype(float),
        "var_a": np.asarray(var_a_t, float),   # <-- correct uncertainty (diag(inv(H))) per day
        "var_b": np.nan,
        "cov_ab": np.nan,
        "n_sites": n_sites_by_day,
        "q_LL_used": np.nan,
        "q_b_used": np.nan,
        "mu_lo": mu_t_lo.astype(float),
        "mu_hi": mu_t_hi.astype(float),
        "kappa_lo": np.nan,
        "kappa_hi": np.nan
    }).sort_values("date")

    p_ts = os.path.join(outdir, "detail_global_timeseries.csv")
    gday.to_csv(p_ts, mode="a", header=not os.path.exists(p_ts), index=False)

    return float(np.median(gday["a_t"])), float(np.median(gday["b_t"]))



# -------------------- Driver --------------------
def run_priors(cfg: Dict[str,Any], ctx: SimpleCtx|None=None) -> Dict[str,Any]:
    ctx = ctx or SimpleCtx()
    pri = cfg.get("priors", cfg)
    results_root = pri.get("results_root", "results")
    outdir = os.path.join(results_root, "priors")
    _ensure_dir(outdir)

    # Pre-create CSVs
    for name, cols in [
        ("priors_full_detail.csv",
         ["site_id","date","mutation","count","coverage","af",
          "mu_t","mu_lo","mu_hi","kappa_t","pi","tau",
          "y_lo_50","y_hi_50","p_lo_50","p_hi_50","y_lo_90","y_hi_90","p_lo_90","p_hi_90"]),
        ("priors_hyperparams.csv",
         ["mutation","mu","kappa","alpha","beta","mu_shrunk","kappa_shrunk"]),
        ("detail_global_timeseries.csv",
         ["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab",
          "n_sites","q_LL_used","q_b_used","mu_lo","mu_hi","kappa_lo","kappa_hi"]),
        ("eb_population_prior.csv", ["m_a","m_b","Saa","Sab","Sbb"]),
    ]:
        p = os.path.join(outdir, name)
        if not os.path.exists(p): pd.DataFrame(columns=cols).to_csv(p, index=False)

    # Data
    df, mutations = _read_feature_table(ctx, pri)

    # Cap extreme coverage
    cap_q = float(pri.get("cap_coverage_quantile", 0.99))
    if 0.5 <= cap_q < 1.0 and not df.empty:
        cap_val = float(df["coverage"].quantile(cap_q))
        if np.isfinite(cap_val) and cap_val > 0:
            df["coverage"] = np.minimum(df["coverage"], int(cap_val))

    # Hyperparameters
    lam_rw2   = float(pri.get("rw2_penalty", 10.0))
    k_lo      = float(pri.get("kappa_lo", 1e-6))
    k_hi      = float(pri.get("kappa_hi", 150.0))
    min_days  = int(pri.get("min_dates", 2))
    em_iters  = int(pri.get("em_iters", 6))
    K_bins    = int(pri.get("calibration_bins", 40))
    mu_floor  = float(pri.get("mu_floor_c", 5.0))
    use_mu_cal= bool(pri.get("use_mu_calibration", True))
    targets   = list(pri.get("target_coverages", [0.50, 0.90]))
    cov_tol   = float(pri.get("coverage_tolerance", 0.01))
    tau_bounds= tuple(pri.get("tau_bounds", [0.1, 60.0]))
    tau_iter  = int(pri.get("tau_bisect_max_iter", 18))
    min_rows_per_bin = int(pri.get("min_rows_per_bin", 200))

    eb_rows=[]
    for m in mutations:
        df_m = df[df["mutation"]==m].sort_values(["date","site_id"]).reset_index(drop=True)
        if df_m["date"].nunique() < min_days:
            ctx.log("WARN", f"Skipping {m} (insufficient days)", {"days": int(df_m["date"].nunique())})
            continue

        ctx.log("INFO", f"EM ZIBB + RW2 for {m}")
        fit = _em_zibb_fit_per_mutation(
            ctx, m, df_m, lam_rw2, (k_lo,k_hi), K_bins, mu_floor, use_mu_calibration=use_mu_cal,
            em_iters=em_iters, min_rows_per_bin=min_rows_per_bin
        )

        ctx.log("INFO", f"Export + τ calibration for {m}")
        a_med, b_med = _export_mutation(
            ctx, m, df_m, fit, outdir, targets, tau_bounds, tau_iter, cov_tol, min_rows_per_bin
        )
        eb_rows.append({"mutation": m, "a_map": a_med, "b_map": b_med})

    if eb_rows:
        eb_df = pd.DataFrame(eb_rows).sort_values("mutation")
        a_vals = eb_df["a_map"].to_numpy(float)
        b_vals = eb_df["b_map"].to_numpy(float)
        m_a, m_b = float(np.mean(a_vals)), float(np.mean(b_vals))
        cov_ab = np.cov(np.vstack([a_vals, b_vals]))
        Saa = float(cov_ab[0,0]) if np.isfinite(cov_ab[0,0]) else 1.0
        Sab = float(cov_ab[0,1]) if np.isfinite(cov_ab[0,1]) else 0.0
        Sbb = float(cov_ab[1,1]) if np.isfinite(cov_ab[1,1]) else 1.0
        p_eb = os.path.join(outdir,"eb_population_prior.csv")
        pd.DataFrame([{"m_a": m_a, "m_b": m_b, "Saa": Saa, "Sab": Sab, "Sbb": Sbb}]).to_csv(
            p_eb, mode="a", header=not os.path.exists(p_eb), index=False
        )

    ctx.log("INFO","Done", {"outdir": outdir})
    return {"tables":["priors_full_detail","priors_hyperparams","detail_global_timeseries","eb_population_prior"]}


# -------------------- CLI --------------------
def _load_config(path: str|None) -> Dict[str,Any]:
    if not path: return {}
    txt = open(path,"r",encoding="utf-8").read()
    try:
        import yaml
        return yaml.safe_load(txt)
    except Exception:
        try:
            return json.loads(txt)
        except Exception:
            raise RuntimeError("Provide YAML/JSON config or install PyYAML.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZIBB-Calibrated Priors (Swiss wastewater)")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config.")
    args = ap.parse_args()
    cfg = _load_config(args.config)
    run_priors(cfg, SimpleCtx())
