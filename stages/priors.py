# -*- coding: utf-8 -*-
"""
priors.py — hierarchical global trend (multi‑site fusion) + exact BB, tuned & aligned (WW‑safe)
==============================================================================================

Key features (WW-optimized):
  • GLOBAL = multi‑site IEKS fusion per date (population × flow × QC weights), not pooled counts.
  • Site time‑shift δ (bounded integer days) aligned to the global a(t) curve.
  • Per‑mutation process‑noise (q_LL, q_b) tuned by predictive Beta–Binomial likelihood.
  • Coverage hygiene for pseudo‑obs (min coverage → inflate R; cap extreme coverage).
  • Exact BB mid‑P PIT & exact coverage calibration.
  • JAX exact CDF with OOM‑safe SciPy fallback for large K/batches.

I/O contracts/filenames unchanged; drop into your pipeline as-is.

Entry point: run_priors(cfg, ctx)

cfg["priors"] extra (all optional; safe defaults):
  - served_population_column: str|None  (e.g., "served_population")
  - flow_column:              str|None  (e.g., "flow_mgd")
  - qc_pass_column:           str|None  (e.g., "qc_pass" 0/1)
  - min_coverage_for_update:  float     (default 10)
  - cap_coverage_quantile:    float|None (default 0.999, set None to disable)
  - max_shift_days:           int       (default 28)
  - tune_qLL_grid, tune_qb_grid: lists for grid tuning
  - save_detail:              bool      (default True) — write exhaustive detail tables only (no plot CSVs)
"""

from __future__ import annotations

# ---- JAX must be configured BEFORE importing jax ----
import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "True")
# Optional: avoid weird GPU device grabs / keep CPU stable
_os.environ.setdefault("JAX_PLATFORMS", "cpu")  # newer jax
_os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # older jax env
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import os, math, gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import betabinom, kstest
from scipy.special import gammaln, expit, logit

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, hessian, vmap
from jax.nn import sigmoid
from jaxopt import LBFGS

# ---------------- numeric guards / bounds ----------------
_A_MAX, _B_MAX = 15.0, 50.0              # a=logit(μ) clamp; b=log κ clamp
_KAPPA_MIN, _KAPPA_MAX = 1e-6, 1e12       # κ clamps
_EPS_DEFAULT = 1e-9
_JITTER = 1e-9
_SEED = 123
_GLOBAL_SITE_ID = "__GLOBAL__"           # global fused trend pseudo-site

# XLA/JAX memory safety thresholds (control fallback to SciPy)
_MAX_K_FOR_JAX = 4096                    # max recurrence length per call
_MAX_ELEMS_FOR_JAX = 8_000_000           # B*(K+1) guard per cdf block
_BIG_R = 1e12
_MIN_W = 1e-6

# ---------------- Minimal RunContext fallback ----------------
try:
    from utils.run import RunContext  # project-specific
except Exception:
    class RunContext:
        def log(self, **kw): print("[LOG]", kw)
        def write_table(self, key, df):
            out = os.path.join("results", "priors", "tables"); os.makedirs(out, exist_ok=True)
            df.to_csv(os.path.join(out, f"{key}.csv"), index=False)
        def write_report(self, text):
            out = os.path.join("results", "priors"); os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "report.md"), "a", encoding="utf-8") as f: f.write(text + "\n")

# ---------------- small helpers ----------------
def _clip01(x, eps):
    return np.clip(x, eps, 1.0 - eps)

def _safe_logit(p: float, eps: float) -> float:
    return float(logit(float(np.clip(p, eps, 1 - eps))))

def _inv_logit(a: float, eps: float) -> float:
    a = float(np.clip(a, -_A_MAX, _A_MAX))
    res = float(expit(a))
    return float(np.clip(res, eps, 1 - eps))

def _safe_kappa_from_b(b: float) -> float:
    """κ = exp(b) with b clamped; then κ clamped."""
    b = float(np.clip(b, -_B_MAX, _B_MAX))
    k = float(np.exp(b))
    return float(np.clip(k, _KAPPA_MIN, _KAPPA_MAX))

def _safe_b_from_kappa(kappa: float) -> float:
    """b = log(κ) with κ clamped; then b also clamped to [-B_MAX, B_MAX]."""
    k = float(np.clip(kappa, _KAPPA_MIN, _KAPPA_MAX))
    b = float(np.log(k))
    return float(np.clip(b, -_B_MAX, _B_MAX))

def _as_1d_f64(x) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=np.float64)))

def _ensure_1d(x) -> np.ndarray:
    a = np.asarray(x)
    return a.reshape(1,) if a.ndim == 0 else a.reshape(-1)

def _make_w(n: np.ndarray) -> np.ndarray:
    n = _as_1d_f64(n); s = float(np.sum(n))
    return (n / s) if s > 0.0 else np.zeros(n.shape[0], dtype=np.float64)

def var_Y_beta_binom(n, mu, kappa):
    n = np.asarray(n, float)
    mu = _clip01(np.asarray(mu, float), 1e-12)
    kappa = np.clip(np.asarray(kappa, float), _KAPPA_MIN, _KAPPA_MAX)
    rho = 1.0/(kappa + 1.0)
    return n * mu * (1.0 - mu) * (1.0 + (n - 1.0) * rho)

# ---------------- Structures ----------------
@dataclass
class PreparedMut:
    mutation: str
    y: np.ndarray
    n: np.ndarray

@dataclass
class PreparedSeries:
    site_id: str
    mutation: str
    dates: List[pd.Timestamp]
    y: np.ndarray
    n: np.ndarray

def _prep_group_arrays(g: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_1d_f64(g["count"].to_numpy(copy=False))
    n = _as_1d_f64(g["coverage"].to_numpy(copy=False))
    m = np.isfinite(y) & np.isfinite(n) & (n > 0.0)
    if not np.any(m):
        return np.zeros(1, dtype=np.float64), np.ones(1, dtype=np.float64)
    y = np.minimum(np.maximum(y[m], 0.0), n[m])
    n = n[m]
    return y, n

def _prepare_arrays_for_priors(df: pd.DataFrame) -> Tuple[List[PreparedMut], List[PreparedSeries], pd.DataFrame]:
    muts: List[PreparedMut] = []
    for mut, g in df.groupby("mutation", sort=False):
        y, n = _prep_group_arrays(g)
        muts.append(PreparedMut(mutation=str(mut), y=y, n=n))
    daily = (df.groupby(["site_id","date","mutation"], as_index=False)[["count","coverage"]]
               .sum().sort_values(["site_id","mutation","date"]))
    series: List[PreparedSeries] = []
    for (site, mut), s in daily.groupby(["site_id","mutation"], sort=False):
        y = _as_1d_f64(s["count"].to_numpy(copy=False))
        n = np.maximum(_as_1d_f64(s["coverage"].to_numpy(copy=False)), 1.0)
        dates = pd.to_datetime(s["date"]).tolist()
        series.append(PreparedSeries(site_id=str(site), mutation=str(mut), dates=dates, y=y, n=n))
    return muts, series, daily

# ---------------- JAX: loglik in (a,b) ----------------
def _bb_logpmf_sum_ab(a: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, n: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    mu = sigmoid(a)
    kap = jnp.clip(jnp.exp(b), _KAPPA_MIN, _KAPPA_MAX)
    alpha, beta = mu * kap, (1.0 - mu) * kap
    lg = jax.scipy.special.gammaln
    logpmf = (
        lg(n + 1.0) - lg(y + 1.0) - lg(n - y + 1.0)
        + lg(y + alpha) + lg(n - y + beta) - lg(n + alpha + beta)
        + lg(alpha + beta) - lg(alpha) - lg(beta)
    )
    invalid = (y < 0) | (n < 0) | (y > n)
    logpmf = jnp.where(invalid, -jnp.inf, logpmf)
    return jnp.sum(w * logpmf)

@jit
def _obj_ab(ab: jnp.ndarray, y: jnp.ndarray, n: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    a = jnp.clip(ab[0], -_A_MAX, _A_MAX)
    b = jnp.clip(ab[1], -_B_MAX, _B_MAX)
    return -_bb_logpmf_sum_ab(a, b, y, n, w)

_grad_obj = jit(grad(_obj_ab))
_hess_obj = jit(hessian(_obj_ab))

@jit
def _bb_logpmf_ab_single(a: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    mu = sigmoid(a)
    kap = jnp.clip(jnp.exp(b), _KAPPA_MIN, _KAPPA_MAX)
    alpha, beta = mu * kap, (1.0 - mu) * kap
    lg = jax.scipy.special.gammaln
    return (lg(n + 1.0) - lg(y + 1.0) - lg(n - y + 1.0)
            + lg(y + alpha) + lg(n - y + beta) - lg(n + alpha + beta)
            + lg(alpha + beta) - lg(alpha) - lg(beta))

_grad_single = jit(grad(lambda a_b, y, n: _bb_logpmf_ab_single(a_b[0], a_b[1], y, n)))
_hess_single = jit(hessian(lambda a_b, y, n: _bb_logpmf_ab_single(a_b[0], a_b[1], y, n)))

@jit
def _pseudo_obs_for_row(a_t: jnp.ndarray, b_t: jnp.ndarray, y: jnp.ndarray, n: jnp.ndarray):
    theta = jnp.array([a_t, b_t])
    g  = _grad_single(theta, y, n)
    Hn = -_hess_single(theta, y, n) + jnp.eye(2)*1e-8
    R  = jnp.linalg.inv(Hn)
    z  = theta - (R @ g)
    return z, R

_vmap_pseudo = jit(vmap(_pseudo_obs_for_row, in_axes=(0,0,0,0), out_axes=(0,0)))

# ---------------- Robust MLE in (a,b) ----------------
@dataclass
class MLEConfig:
    max_iter: int = 1000
    tol: float = 1e-8
    eps: float = _EPS_DEFAULT

def _moments_start(y: np.ndarray, n: np.ndarray, eps: float) -> Tuple[float, float]:
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    tot_n = float(np.sum(n)); tot_y = float(np.sum(y))
    mu0 = float(np.clip(tot_y / max(tot_n, 1.0), eps, 1 - eps)) if tot_n > 0 else 0.5
    return mu0, 1e3

def _mle_bb_jaxopt(y: np.ndarray, n: np.ndarray, w: np.ndarray, mu0: float, k0: float, cfg: MLEConfig):
    yj, nj, wj = jnp.array(_as_1d_f64(y)), jnp.array(_as_1d_f64(n)), jnp.array(_as_1d_f64(w))
    a0 = _safe_logit(mu0, cfg.eps)
    b0 = _safe_b_from_kappa(k0)
    solver = LBFGS(fun=_obj_ab, value_and_grad=True, maxiter=int(cfg.max_iter), tol=float(cfg.tol))
    res = solver.run(jnp.array([a0, b0], dtype=jnp.float64), yj, nj, wj)
    a_hat, b_hat = float(res.params[0]), float(res.params[1])
    mu_hat = _inv_logit(a_hat, cfg.eps)
    kap_hat = _safe_kappa_from_b(b_hat)
    H = np.array(_hess_obj(res.params, yj, nj, wj), dtype=float)
    H[0,0] = max(H[0,0], 1e-10); H[1,1] = max(H[1,1], 1e-10)
    if not np.all(np.isfinite(H)): H = np.diag([1.0, 1.0])
    return mu_hat, kap_hat, float(res.state.value), H

def _mle_beta_binom_robust(y, n, mu0, kappa0, cfg: MLEConfig):
    w_base = _make_w(n)
    mu_hat, kap_hat, _, H = _mle_bb_jaxopt(y, n, w_base, mu0, kappa0, cfg)
    for _ in range(2):
        var_y = var_Y_beta_binom(n, mu_hat, kap_hat)
        z = (_as_1d_f64(y) - _as_1d_f64(n)*mu_hat) / np.sqrt(np.maximum(var_y, cfg.eps))
        u = np.clip(z/4.685, -1, 1)
        w = w_base * np.where(np.abs(z) < 4.685, (1 - u*u)**2, 0.0)
        if np.sum(w) <= 0: break
        mu_hat, kap_hat, _, H = _mle_bb_jaxopt(y, n, w, mu_hat, kap_hat, cfg)
    return mu_hat, kap_hat, H

# ---------------- EB pooling in (a,b) ----------------
def _fit_re_normal_fullcov(mle_rows: List[Dict], min_tau: float = 1e-4):
    if not mle_rows:
        mu_emp = np.array([0.0, math.log(1e3)], float)
        S_pop = np.diag([1.0, 1.0])
        return mu_emp, S_pop, np.linalg.inv(S_pop)
    A=[]; B=[]; Sig_sam=[]
    for r in mle_rows:
        A.append(float(r["a_hat"])); B.append(float(r["b_hat"]))
        H = np.array(r["H"], float)
        try: Sig = np.linalg.inv(H + 1e-9*np.eye(2))
        except Exception: Sig = np.diag([1.0,1.0])
        Sig_sam.append(Sig)
    A=np.asarray(A,float); B=np.asarray(B,float)
    mu_emp = np.array([np.mean(A), np.mean(B)], float)
    S_emp  = np.cov(np.vstack([A,B]))
    if not np.all(np.isfinite(S_emp)) or S_emp.shape!=(2,2): S_emp = np.diag([1.0,1.0])
    Sig_med = np.median(np.stack(Sig_sam,axis=0), axis=0)
    S_pop = S_emp - Sig_med
    w, V = np.linalg.eigh(S_pop); w = np.clip(w, min_tau**2, None)
    S_pop = (V * w) @ V.T
    return mu_emp, S_pop, np.linalg.inv(S_pop)

def _map_fullcov_for_mut(y, n, a0, b0, mu_emp, Si, eps, max_iter=400, ftol=1e-8):
    from scipy.optimize import minimize
    w_cov = _make_w(n)
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    def nll_pen(ab):
        a = float(np.clip(ab[0], -_A_MAX, _A_MAX))
        b = float(np.clip(ab[1], -_B_MAX, _B_MAX))
        mu = _inv_logit(a, eps); kap = _safe_kappa_from_b(b)
        ll  = betabinom.logpmf(y.astype(int), n.astype(int), mu*kap, (1.0-mu)*kap)
        base = -float(np.sum(w_cov[np.isfinite(ll)] * ll[np.isfinite(ll)])) if ll.size else 1e100
        d = np.array([a, b]) - mu_emp
        return base + 0.5 * float(d @ Si @ d)
    res = minimize(nll_pen, x0=np.array([a0,b0],float), method="L-BFGS-B",
                   bounds=[(-_A_MAX,_A_MAX),(-_B_MAX,_B_MAX)],
                   options={"maxiter": int(max_iter), "ftol": float(ftol)})
    a_map, b_map = (res.x if (res.success and np.all(np.isfinite(res.x))) else np.array([a0,b0]))
    mu_map = _inv_logit(float(a_map), eps)
    kap_map = _safe_kappa_from_b(float(b_map))
    return mu_map, kap_map, float(a_map), float(b_map)

# ---------------- Hygiene helpers ----------------
def _apply_hygiene_for_pseudo(y, n, min_cov: Optional[float], cap_quantile: Optional[float]):
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    mask_low = np.zeros_like(n, dtype=bool)
    if min_cov is not None:
        mask_low = n < float(min_cov)
    y_eff, n_eff = y.copy(), n.copy()
    if cap_quantile is not None and n.size:
        cap = float(np.quantile(n, float(cap_quantile)))
        n_eff = np.minimum(n_eff, cap)
        af = np.clip(y / np.maximum(n, 1.0), 0.0, 1.0)
        y_eff = np.round(af * n_eff).astype(np.float64)
        y_eff = np.minimum(np.maximum(y_eff, 0.0), n_eff)
    return y_eff, n_eff, mask_low

# ---------------- RW2(a,v)+RW1(b) smoother (safe) ----------------
def _compute_dt_days(dates_like: List[pd.Timestamp]) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates_like)).values.astype('datetime64[D]').astype('int64')
    dt = np.diff(np.insert(d, 0, d[0])).astype(float); dt[dt < 1] = 1.0
    return dt

def _rw2_kalman_smoother_vec_ct(z_list, R_list, q_LL, q_b, dt, x0, P0):
    T = len(z_list)
    x_pred = np.zeros((T,3)); P_pred = np.zeros((T,3,3))
    x_filt = np.zeros((T,3)); P_filt = np.zeros((T,3,3))
    I3 = np.eye(3); C = np.array([[1.,0.,0.],[0.,0.,1.]], dtype=float)

    x_prev = x0.copy()
    P_prev = P0.copy()
    for t in range(T):
        dt_t = float(max(dt[t], 1.0))
        F = np.array([[1., dt_t, 0.],
                      [0., 1.,   0.],
                      [0., 0.,   1.]], dtype=float)
        Q = np.array([[q_LL*dt_t**3/3., q_LL*dt_t**2/2., 0.],
                      [q_LL*dt_t**2/2., q_LL*dt_t,       0.],
                      [0.,              0.,              q_b*dt_t]], dtype=float)

        x_pr = F @ x_prev
        P_pr = F @ P_prev @ F.T + Q

        R = np.array(R_list[t], dtype=float)
        if not np.all(np.isfinite(R)): R = np.diag([1e6, 1e6])
        R = R + _JITTER*np.eye(2)

        S = C @ P_pr @ C.T + R
        S = S + _JITTER*np.eye(2)
        try:
            S_inv = np.linalg.inv(S)
        except Exception:
            S_inv = np.linalg.pinv(S)
        K = P_pr @ C.T @ S_inv
        z = np.array(z_list[t], dtype=float)

        innov = z - (C @ x_pr)
        x_upd = x_pr + K @ innov
        x_upd[0] = float(np.clip(x_upd[0], -_A_MAX, _A_MAX))
        x_upd[2] = float(np.clip(x_upd[2], -_B_MAX, _B_MAX))
        P_upd = (I3 - K @ C) @ P_pr

        x_pred[t], P_pred[t] = x_pr, P_pr
        x_filt[t], P_filt[t] = x_upd, P_upd
        x_prev, P_prev = x_upd, P_upd

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T-2, -1, -1):
        dt_t = float(max(dt[t+1], 1.0))
        F = np.array([[1., dt_t, 0.],
                      [0., 1.,   0.],
                      [0., 0.,   1.]], dtype=float)
        Ppr_next = P_pred[t+1]
        try:
            invPpr = np.linalg.inv(Ppr_next + _JITTER*np.eye(3))
        except Exception:
            invPpr = np.linalg.pinv(Ppr_next + _JITTER*np.eye(3))
        A = P_filt[t] @ F.T @ invPpr
        x_smooth[t] += A @ (x_smooth[t+1] - (F @ x_filt[t]))
        x_smooth[t,0] = float(np.clip(x_smooth[t,0], -_A_MAX, _A_MAX))
        x_smooth[t,2] = float(np.clip(x_smooth[t,2], -_B_MAX, _B_MAX))
        P_smooth[t] += A @ (P_smooth[t+1] - Ppr_next) @ A.T

    return x_smooth, P_smooth

def _dynamic_bb_series_ct_rw2(y, n, dates, kappa_init, mu_init,
                              q_LL=0.05, q_b=1e-3, iters=3, eps=1e-9,
                              min_cov: Optional[float]=None, cap_quantile: Optional[float]=None):
    T = len(y)
    if T == 0: return (np.zeros((0,3)), np.zeros((0,3,3)), np.zeros((0,)))
    mu0 = float(np.clip(mu_init, eps, 1-eps)); k0 = float(np.clip(kappa_init, _KAPPA_MIN, _KAPPA_MAX))
    a0, b0 = _safe_logit(mu0, eps), _safe_b_from_kappa(k0)
    dts = _compute_dt_days(dates)

    x = np.tile(np.array([a0, 0.0, b0], float), (T,1))
    P = np.tile(np.diag([10.0, 5.0, 5.0]), (T,1,1))

    y_eff, n_eff, mask_low = _apply_hygiene_for_pseudo(y, n, min_cov, cap_quantile)

    for _ in range(max(1, int(iters))):
        a_arr = jnp.array(np.clip(x[:,0], -_A_MAX, _A_MAX))
        b_arr = jnp.array(np.clip(x[:,2], -_B_MAX, _B_MAX))
        y_arr = jnp.array(_as_1d_f64(y_eff), dtype=jnp.float64)
        n_arr = jnp.array(_as_1d_f64(n_eff), dtype=jnp.float64)
        Z, R = _vmap_pseudo(a_arr, b_arr, y_arr, n_arr)
        Z = np.array(Z); R = np.array(R)
        if mask_low is not None and np.any(mask_low):
            R[mask_low] = R[mask_low] + np.diag([_BIG_R, _BIG_R])
        x, P = _rw2_kalman_smoother_vec_ct(Z, R,
                                           q_LL=float(q_LL), q_b=float(q_b),
                                           dt=dts,
                                           x0=np.array([a0, 0.0, b0], float),
                                           P0=np.diag([10.0, 5.0, 5.0]))
    return x, P, dts

def _dynamic_bb_series_ct_rw2_with_offset(y, n, dates, g_a_vec, kappa_init, mu_init,
                                          q_LL=0.05, q_b=1e-3, iters=3, eps=1e-9,
                                          min_cov: Optional[float]=None, cap_quantile: Optional[float]=None):
    T = len(y)
    if T == 0: return (np.zeros((0,3)), np.zeros((0,3,3)), np.zeros((0,)))
    mu0 = float(np.clip(mu_init, eps, 1-eps))
    k0 = float(np.clip(kappa_init, _KAPPA_MIN, _KAPPA_MAX))
    a0_local = _safe_logit(mu0, eps)
    g0 = float(g_a_vec[0]) if T > 0 else 0.0
    d0 = float(np.clip(a0_local - g0, -_A_MAX, _A_MAX))
    b0 = _safe_b_from_kappa(k0)
    dts = _compute_dt_days(dates)

    x = np.tile(np.array([d0, 0.0, b0], float), (T,1))
    P = np.tile(np.diag([10.0, 5.0, 5.0]), (T,1,1))

    y_eff, n_eff, mask_low = _apply_hygiene_for_pseudo(y, n, min_cov, cap_quantile)

    for _ in range(max(1, int(iters))):
        a_arr = jnp.array(np.clip(g_a_vec + x[:,0], -_A_MAX, _A_MAX))
        b_arr = jnp.array(np.clip(x[:,2], -_B_MAX, _B_MAX))
        y_arr = jnp.array(_as_1d_f64(y_eff), dtype=jnp.float64)
        n_arr = jnp.array(_as_1d_f64(n_eff), dtype=jnp.float64)
        Z, R = _vmap_pseudo(a_arr, b_arr, y_arr, n_arr)
        Z = np.array(Z); R = np.array(R)
        Z[:,0] = Z[:,0] - g_a_vec
        if mask_low is not None and np.any(mask_low):
            R[mask_low] = R[mask_low] + np.diag([_BIG_R, _BIG_R])
        x, P = _rw2_kalman_smoother_vec_ct(Z, R,
                                           q_LL=float(q_LL), q_b=float(q_b),
                                           dt=dts,
                                           x0=np.array([d0, 0.0, b0], float),
                                           P0=np.diag([10.0, 5.0, 5.0]))
    return x, P, dts

# ---------------- FAST exact mid‑P & coverage via JAX recurrence (with SciPy fallback) ----------------
def _bb_pmf0(alpha: jnp.ndarray, beta: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    lg = jax.scipy.special.gammaln
    return jnp.exp(lg(n + beta) + lg(alpha + beta) - lg(n + alpha + beta) - lg(beta))

def _cdf_forward_batch_factory(K: int):
    @jit
    def _fn(alpha: jnp.ndarray, beta: jnp.ndarray, n: jnp.ndarray):
        pmf0 = _bb_pmf0(alpha, beta, n)
        ks = jnp.arange(K, dtype=alpha.dtype)
        num1 = (n[:, None] - ks[None, :])
        den1 = (ks[None, :] + 1.0)
        num2 = (ks[None, :] + alpha[:, None])
        den2 = (n[:, None] - ks[None, :] - 1.0 + beta[:, None])
        ratio = (num1/den1) * (num2/den2)
        pmf_tail = jnp.cumprod(ratio, axis=1)
        pmf = jnp.concatenate([pmf0[:,None], pmf0[:,None]*pmf_tail], axis=1)
        cdf = jnp.cumsum(pmf, axis=1)
        return pmf, cdf
    return _fn

_CDF_FN_CACHE: Dict[int, callable] = {}
def _get_cdf_forward_fn(K: int):
    fn = _CDF_FN_CACHE.get(K)
    if fn is None:
        fn = _cdf_forward_batch_factory(K)
        _CDF_FN_CACHE[K] = fn
    return fn

def _midp_exact_scipy(y: np.ndarray, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Fy = betabinom.cdf(y, n, a, b)
    pmf_y = betabinom.pmf(y, n, a, b)
    out = np.clip(Fy - 0.5*pmf_y, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.5
    return out

def _coverage_exact_scipy(mu: np.ndarray, k: np.ndarray, y: np.ndarray, n: np.ndarray, q: float) -> float:
    a = mu*k; b = (1.0-mu)*k
    Fy = betabinom.cdf(y, n, a, b)
    pmf_y = betabinom.pmf(y, n, a, b)
    Fy = np.clip(Fy, 0.0, 1.0)
    Fym1 = np.clip(Fy - pmf_y, 0.0, 1.0)
    a_tail = (1.0 - q)/2.0
    covered = (Fym1 <= a_tail) & ((1.0 - Fy) <= a_tail)
    return float(np.mean(covered)) if covered.size else 0.0

def _midp_from_exact_vec_fast(y_np: np.ndarray, n_np: np.ndarray, mu_np: np.ndarray, kappa_np: np.ndarray,
                              chunk_rows: int = 4096) -> np.ndarray:
    y_np = _ensure_1d(y_np).astype(int)
    n_np = _ensure_1d(n_np).astype(int)
    mu_np = _clip01(_ensure_1d(mu_np).astype(float), 1e-12)
    k_np  = np.clip(_ensure_1d(kappa_np).astype(float), _KAPPA_MIN, _KAPPA_MAX)
    a_np, b_np = mu_np * k_np, (1.0 - mu_np) * k_np

    out = np.empty_like(mu_np, dtype=float)
    m = len(y_np)
    for start in range(0, m, chunk_rows):
        stop = min(m, start + chunk_rows)
        y, n, a, b = y_np[start:stop], n_np[start:stop], a_np[start:stop], b_np[start:stop]

        left_mask = (y <= (n//2)); right_mask = ~left_mask
        K_L = int(np.max(y[left_mask])) if np.any(left_mask) else 0
        yr_cdf = (n[right_mask] - y[right_mask] - 1).astype(int) if np.any(right_mask) else np.array([], int)
        yr_pmf = (n[right_mask] - y[right_mask]).astype(int) if np.any(right_mask) else np.array([], int)
        K_R = int(np.max(np.maximum(yr_cdf, yr_pmf))) if np.any(right_mask) else 0
        B = y.shape[0]
        use_jax = (max(K_L, K_R) <= _MAX_K_FOR_JAX) and (B*(max(K_L,K_R)+1) <= _MAX_ELEMS_FOR_JAX)

        if not use_jax:
            mid_chunk = _midp_exact_scipy(y, n, a, b)
            out[start:stop] = mid_chunk
            continue

        mid_chunk = np.empty_like(y, dtype=float)

        if np.any(left_mask):
            yL = y[left_mask]; nL = n[left_mask]; aL = a[left_mask]; bL = b[left_mask]
            K = int(np.max(yL)); fn = _get_cdf_forward_fn(K)
            pmfL, cdfL = fn(jnp.array(aL), jnp.array(bL), jnp.array(nL))
            Fy   = np.asarray(cdfL[np.arange(yL.size), yL], dtype=float)
            Fym1 = np.zeros_like(Fy)
            pos = yL > 0
            Fym1[pos] = np.asarray(cdfL[np.arange(yL.size)[pos], yL[pos]-1], dtype=float)
            pmf_y = Fy - Fym1
            out[left_mask] = np.clip(Fym1 + 0.5*pmf_y, 0.0, 1.0)

        if np.any(right_mask):
            yR = y[right_mask]; nR = n[right_mask]; aR = a[right_mask]; bR = b[right_mask]
            yr_cdf = (nR - yR - 1).astype(int)
            yr_pmf = (nR - yR).astype(int)
            K = int(np.max(np.maximum(yr_cdf, yr_pmf)))
            fn = _get_cdf_forward_fn(K)
            pmfR, cdfR = fn(jnp.array(bR), jnp.array(aR), jnp.array(nR))
            U = np.zeros(yr_cdf.shape[0], dtype=float); posU = yr_cdf >= 0
            U[posU] = np.asarray(cdfR[np.arange(yr_cdf.size)[posU], yr_cdf[posU]], dtype=float)
            Fy = 1.0 - U
            pmf_y = np.asarray(pmfR[np.arange(yr_pmf.size), yr_pmf], dtype=float)
            Fym1 = Fy - pmf_y
            out[right_mask] = np.clip(Fym1 + 0.5*pmf_y, 0.0, 1.0)
    return out

def _coverage_exact_fast(mu_vec: np.ndarray, kappa_vec: np.ndarray, y: np.ndarray, n: np.ndarray, q: float) -> float:
    y = _ensure_1d(y).astype(int)
    n = _ensure_1d(n).astype(int)
    mu = _clip01(_ensure_1d(mu_vec).astype(float), 1e-12)
    k  = np.clip(_ensure_1d(kappa_vec).astype(float), _KAPPA_MIN, _KAPPA_MAX)
    a, b = mu*k, (1.0-mu)*k

    K_eff = int(np.max(np.minimum(y, n-y))) if y.size else 0
    B = y.size
    use_jax = (K_eff <= _MAX_K_FOR_JAX) and (B*(K_eff+1) <= _MAX_ELEMS_FOR_JAX)
    if not use_jax:
        return _coverage_exact_scipy(mu, k, y, n, q)

    covered = np.empty_like(mu, dtype=bool)
    chunk_rows = max(256, int(_MAX_ELEMS_FOR_JAX // (max(1, K_eff+1))))
    for start in range(0, len(mu), chunk_rows):
        stop = min(len(mu), start + chunk_rows)
        yC, nC, aC, bC = y[start:stop], n[start:stop], a[start:stop], b[start:stop]
        left_mask = (yC <= (nC//2))
        right_mask = ~left_mask
        cov_local = np.zeros_like(yC, dtype=bool)

        if np.any(left_mask):
            yL = yC[left_mask]; nL = nC[left_mask]; aL = aC[left_mask]; bL = bC[left_mask]
            K = int(np.max(yL)); fn = _get_cdf_forward_fn(K)
            pmfL, cdfL = fn(jnp.array(aL), jnp.array(bL), jnp.array(nL))
            Fy   = np.asarray(cdfL[np.arange(yL.size), yL], dtype=float)
            Fym1 = np.zeros_like(Fy)
            pos = yL > 0
            Fym1[pos] = np.asarray(cdfL[np.arange(yL.size)[pos], yL[pos]-1], dtype=float)
            a_tail = (1.0 - q)/2.0
            cov_local[left_mask] = (Fym1 <= a_tail) & ((1.0 - Fy) <= a_tail)

        if np.any(right_mask):
            yR = yC[right_mask]; nR = nC[right_mask]; aR = aC[right_mask]; bR = bC[right_mask]
            yr_cdf = (nR - yR - 1).astype(int)
            yr_pmf = (nR - yR).astype(int)
            K = int(np.max(np.maximum(yr_cdf, yr_pmf))); fn = _get_cdf_forward_fn(K)
            pmfR, cdfR = fn(jnp.array(bR), jnp.array(aR), jnp.array(nR))
            U = np.zeros(yr_cdf.shape[0], dtype=float); posU = yr_cdf >= 0
            U[posU] = np.asarray(cdfR[np.arange(yr_cdf.size)[posU], yr_cdf[posU]], dtype=float)
            Fy = 1.0 - U
            pmf_y = np.asarray(pmfR[np.arange(yr_pmf.size), yr_pmf], dtype=float)
            Fym1 = Fy - pmf_y
            a_tail = (1.0 - q)/2.0
            cov_local[right_mask] = (Fym1 <= a_tail) & ((1.0 - Fy) <= a_tail)

        covered[start:stop] = cov_local
    return float(np.mean(covered)) if covered.size else 0.0

# ---------------- Calibration ----------------
def _calibrate_kappa_global_exact_dynamic(rows: pd.DataFrame, target_q: float = 0.80,
                                          iters: int = 6, s_bounds=(1/8, 8), return_trace: bool=False,
                                          max_rows: Optional[int]=200_000, seed: int=123):
    use = rows[["count","coverage","mu_t","kappa"]].copy()
    if max_rows is not None and len(use) > max_rows:
        use["cov_bin"] = pd.qcut(use["coverage"], q=min(32, max(2, use["coverage"].nunique())), duplicates="drop", labels=False)
        use["mu_bin"]  = pd.qcut(use["mu_t"],     q=min(32, max(2, use["mu_t"].nunique())),     duplicates="drop", labels=False)
        frac = max_rows / len(use)
        use = (use.groupby(["cov_bin","mu_bin"], group_keys=False)
                 .apply(lambda g: g.sample(max(1, int(round(frac*len(g)))), random_state=seed))
                 .reset_index(drop=True))
    y  = use["count"].to_numpy(int)
    nn = use["coverage"].to_numpy(int)
    mu = use["mu_t"].to_numpy(float)
    k0 = use["kappa"].to_numpy(float)

    s_lo, s_hi = float(s_bounds[0]), float(s_bounds[1])
    trace = []
    for it in range(iters):
        s_mid = math.sqrt(s_lo*s_hi)
        cov_mid = _coverage_exact_fast(mu, k0*s_mid, y, nn, target_q)
        trace.append({"iter": it, "s_lo": s_lo, "s_hi": s_hi, "s_mid": s_mid, "coverage_emp": cov_mid})
        if cov_mid > target_q: s_lo = s_mid
        else: s_hi = s_mid
    s_final = float(math.sqrt(s_lo*s_hi))
    return (s_final, pd.DataFrame(trace)[["iter","s_lo","s_hi","s_mid","coverage_emp"]]) if return_trace else s_final

# ---------------- I/O ----------------
def _read_and_prepare_feature_table(ctx: RunContext, cfg: Dict) -> pd.DataFrame:
    data_root = cfg.get("data_root", "data")
    results_root = cfg.get("results_root", "results")
    preproc_tables = os.path.join(results_root, "preprocessing", "tables")
    candidates = [ "feature_store_snv.csv", "feature_store_snv.cleaned.csv",
                   "snv_long.csv", "feature_store_snv_long.csv" ]
    path = None
    for name in candidates:
        p = os.path.join(preproc_tables, name)
        if os.path.exists(p): path = p; break
    if path is None:
        p = os.path.join(data_root, "jahn_like.csv")
        if not os.path.exists(p):
            raise FileNotFoundError("SNV table not found in preprocessing or data/")
        path = p
    ctx.log(level="INFO", message="Loaded SNV feature store", context={"path": path})
    df = pd.read_csv(path)

    required = ["site_id","date","mutation","count","coverage"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    df["site_id"] = df["site_id"].astype(str)
    df["mutation"] = df["mutation"].astype(str)
    if "sample_id" in df.columns: df["sample_id"] = df["sample_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["count","coverage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df[df["coverage"] > 0]
    df["count"] = df["count"].clip(lower=0)
    df.loc[df["count"] > df["coverage"], "count"] = df["coverage"]
    df["af"] = (df["count"] / df["coverage"]).clip(0.0, 1.0)
    df = df.sort_values(["site_id","mutation","date"]).reset_index(drop=True)
    return df

# ---------------- Diagnostics & gates ----------------
def _stratify_by_coverage_and_time(df: pd.DataFrame, n_bins_cov=10, n_bins_time=4):
    out=[]
    if df.empty: return pd.DataFrame(columns=["cov_bin","time_bin","coverage_p50","start_date","end_date"])
    cov = df["coverage"].to_numpy(float)
    cov_q = np.quantile(cov, np.linspace(0,1,n_bins_cov+1))
    cov_q[0] = np.floor(cov_q[0]); cov_q[-1] = np.ceil(cov_q[-1])
    dates = pd.to_datetime(df["date"]).sort_values()
    if dates.nunique() < n_bins_time:
        tbins = np.linspace(0, 1, n_bins_time+1)
        time_edges = pd.to_datetime(np.quantile(dates.values.astype('int64'), tbins)).values
    else:
        time_edges = np.quantile(dates.values.astype('int64'), np.linspace(0,1,n_bins_time+1)).astype('int64')
    for i in range(n_bins_cov):
        c_lo, c_hi = cov_q[i], cov_q[i+1]
        sel_c = (df["coverage"] >= c_lo) & (df["coverage"] <= c_hi if i==n_bins_cov-1 else df["coverage"] < c_hi)
        for j in range(n_bins_time):
            t_lo, t_hi = time_edges[j], time_edges[j+1]
            tvals = df["date"].values.astype('int64')
            sel_t = (tvals >= t_lo) & (tvals <= t_hi if j==n_bins_time-1 else tvals < t_hi)
            s = df[sel_c & sel_t]
            if s.empty: continue
            out.append({
                "cov_bin": i, "time_bin": j,
                "coverage_p50": float(np.median(s["coverage"])),
                "start_date": pd.to_datetime(t_lo),
                "end_date":   pd.to_datetime(t_hi),
                "n_rows": int(s.shape[0]),
            })
    return pd.DataFrame(out)

def _compute_stratified_stats(rows: pd.DataFrame, target_q: float):
    base = rows.copy()
    base["pit_mid"] = _midp_from_exact_vec_fast(base["count"].to_numpy(int),
                                                base["coverage"].to_numpy(int),
                                                base["mu_t"].to_numpy(float),
                                                base["kappa_use"].to_numpy(float))
    base["z"] = (base["count"] - base["coverage"]*base["mu_t"]) / np.sqrt(np.maximum(
        var_Y_beta_binom(base["coverage"], base["mu_t"], base["kappa_use"]), _EPS_DEFAULT))

    try: ks_p = kstest(base["pit_mid"].to_numpy(float), "uniform")[1]
    except Exception: ks_p = np.nan

    cov_rate = _coverage_exact_fast(base["mu_t"].to_numpy(float), base["kappa_use"].to_numpy(float),
                                    base["count"].to_numpy(int), base["coverage"].to_numpy(int), target_q)

    strata = _stratify_by_coverage_and_time(base)
    cov_rows=[]; pit_rows=[]
    for _, st in strata.iterrows():
        cov_median = st["coverage_p50"]; lo, hi = 0.8*cov_median, 1.2*cov_median
        s = base[(base["coverage"]>=lo) & (base["coverage"]<=hi) & (base["date"].between(st["start_date"], st["end_date"]))]
        if s.empty: 
            cov_rows.append({"cov_bin": int(st["cov_bin"]), "time_bin": int(st["time_bin"]),
                             "coverage_target": float(target_q), "coverage_emp": float(np.nan),
                             "n_rows": 0})
            pit_rows.append({"cov_bin": int(st["cov_bin"]), "time_bin": int(st["time_bin"]), "ks_p": float(np.nan), "n_rows": 0})
            continue
        cov_rate_s = _coverage_exact_fast(s["mu_t"].to_numpy(float), s["kappa_use"].to_numpy(float),
                                          s["count"].to_numpy(int), s["coverage"].to_numpy(int), target_q)
        cov_rows.append({"cov_bin": int(st["cov_bin"]), "time_bin": int(st["time_bin"]),
                         "coverage_target": float(target_q), "coverage_emp": float(cov_rate_s),
                         "n_rows": int(s.shape[0])})
        try: p = kstest(s["pit_mid"].to_numpy(float), "uniform")[1]
        except Exception: p = np.nan
        pit_rows.append({"cov_bin": int(st["cov_bin"]), "time_bin": int(st["time_bin"]), "ks_p": float(p), "n_rows": int(s.shape[0])})

    cov_df = pd.DataFrame(cov_rows).sort_values(["cov_bin","time_bin"]).reset_index(drop=True)
    pit_df = pd.DataFrame(pit_rows).sort_values(["cov_bin","time_bin"]).reset_index(drop=True)

    # enrich with coverage/time metadata
    if not cov_df.empty:
        cov_df = cov_df.merge(strata[["cov_bin","time_bin","coverage_p50","start_date","end_date"]],
                              on=["cov_bin","time_bin"], how="left")
    if not pit_df.empty:
        pit_df = pit_df.merge(strata[["cov_bin","time_bin","coverage_p50","start_date","end_date"]],
                              on=["cov_bin","time_bin"], how="left")

    z = base["z"].to_numpy(float); z = z[np.isfinite(z)]
    z_mean = float(np.nanmean(z)) if z.size else np.nan
    z_var  = float(np.nanvar(z))  if z.size else np.nan

    return {
        "global_pit_ks_p": float(ks_p) if np.isfinite(ks_p) else np.nan,
        "global_coverage": float(cov_rate),
        "cov_by_strata": cov_df,
        "pit_ks_by_strata": pit_df,
        "z_mean": z_mean,
        "z_var": z_var,
        "rows_eval": base.shape[0],
        "base": base,
    }

def _check_gates(stats: Dict, cfg_gates: Dict):
    cov_tol   = float(cfg_gates.get("coverage_tol", 0.02))
    ks_min    = float(cfg_gates.get("pit_ks_p_min", 0.10))
    zvar_lo   = float(cfg_gates.get("z_var_low", 0.95))
    zvar_hi   = float(cfg_gates.get("z_var_high", 1.05))
    dll_q50   = float(cfg_gates.get("delta_ll_q50_min", 0.0))
    cov_df, pit_df = stats["cov_by_strata"], stats["pit_ks_by_strata"]
    cov_ok = True if cov_df.empty else bool(((cov_df["coverage_emp"] >= cov_df["coverage_target"] - cov_tol) &
                                             (cov_df["coverage_emp"] <= cov_df["coverage_target"] + cov_tol)).mean() >= 0.8)
    pit_ok = True if pit_df.empty else bool((pit_df["ks_p"] >= ks_min).mean() >= 0.8)
    z_ok   = bool((stats["z_var"] >= zvar_lo) and (stats["z_var"] <= zvar_hi) and (abs(stats["z_mean"]) < 0.05))
    return {"coverage_ok": cov_ok, "pit_ok": pit_ok, "z_ok": z_ok, "dll_q50_min": dll_q50}

# ---------------- Binomial baseline ----------------
def _binom_logpmf(y: np.ndarray, n: np.ndarray, mu: float, eps: float) -> np.ndarray:
    mu = float(np.clip(mu, eps, 1.0 - eps))
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    out = np.empty_like(y, dtype=float); log_mu, log1 = math.log(mu), math.log(1.0 - mu)
    for i in range(y.shape[0]):
        yi, ni = int(y[i]), int(n[i])
        if ni < 0 or yi < 0 or yi > ni: out[i] = -np.inf
        else:
            out[i] = (gammaln(ni + 1) - gammaln(yi + 1) - gammaln(ni - yi + 1)
                      + yi * log_mu + (ni - yi) * log1)
    return out

# ---------------- GLOBAL multi‑site fusion helpers ----------------
def _build_weight_table(df: pd.DataFrame,
                        served_col: Optional[str],
                        flow_col: Optional[str],
                        qc_col: Optional[str]) -> pd.DataFrame:
    keys = df[["site_id","date"]].drop_duplicates().copy()

    if served_col and (served_col in df.columns):
        pop_site = (df[["site_id", served_col]].dropna()
                    .drop_duplicates(subset=["site_id"]).rename(columns={served_col: "pop"}))
        keys = keys.merge(pop_site, on="site_id", how="left")
    else:
        keys["pop"] = 1.0

    if flow_col and (flow_col in df.columns):
        flow_sd = (df.groupby(["site_id","date"], as_index=False)[flow_col]
                     .mean().rename(columns={flow_col:"flow"}))
        keys = keys.merge(flow_sd, on=["site_id","date"], how="left")
    else:
        keys["flow"] = 1.0

    if qc_col and (qc_col in df.columns):
        qc_sd = (df.groupby(["site_id","date"], as_index=False)[qc_col]
                   .min().rename(columns={qc_col:"qc"}))
        if qc_sd["qc"].dtype != np.bool_:
            qc_sd["qc"] = (pd.to_numeric(qc_sd["qc"], errors="coerce").fillna(1) > 0).astype(int)
        keys = keys.merge(qc_sd, on=["site_id","date"], how="left")
    else:
        keys["qc"] = 1

    for c in ["pop","flow"]:
        keys[c] = pd.to_numeric(keys[c], errors="coerce").fillna(1.0)
        keys[c] = keys[c].clip(lower=0.0)
    keys["qc"] = pd.to_numeric(keys["qc"], errors="coerce").fillna(1).astype(int).clip(lower=0, upper=1)

    keys["w_raw"] = keys["pop"] * keys["flow"] * keys["qc"].astype(float)

    keys["w"] = keys["w_raw"].astype(float)
    per_date = keys.groupby("date")["w"].sum().reset_index().rename(columns={"w":"sumw"})
    per_date["n_sites"] = keys.groupby("date").size().values
    keys = keys.merge(per_date, on="date", how="left")
    keys["w"] = np.where(keys["sumw"] > 0, keys["w_raw"] * (keys["n_sites"] / keys["sumw"]), 1.0)
    keys["w"] = keys["w"].astype(float)
    keys = keys.drop(columns=["w_raw","sumw","n_sites"])
    return keys

def _smooth_global_multisite(daily: pd.DataFrame,
                             weights_sd: pd.DataFrame,
                             mutation: str,
                             mu0_m: float, kappa_m: float,
                             q_LL: float, q_b: float,
                             iters: int,
                             eps: float,
                             min_cov: Optional[float],
                             cap_quantile: Optional[float]) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    g = daily[daily["mutation"] == mutation].copy()
    if g.empty:
        return np.zeros((0,3)), np.zeros((0,3,3)), []

    g = g.merge(weights_sd, on=["site_id","date"], how="left")
    if "w" not in g.columns:
        g["w"] = 1.0
    g["w"] = pd.to_numeric(g["w"], errors="coerce").fillna(1.0).astype(float)
    g["w"] = np.maximum(g["w"], _MIN_W)

    dates = sorted(pd.to_datetime(g["date"]).unique())
    T = len(dates)
    if T == 0:
        return np.zeros((0,3)), np.zeros((0,3,3)), []

    mu0 = float(np.clip(mu0_m, eps, 1-eps)); k0 = float(np.clip(kappa_m, _KAPPA_MIN, _KAPPA_MAX))
    a0, b0 = _safe_logit(mu0, eps), _safe_b_from_kappa(k0)

    dts = _compute_dt_days(dates)
    I3 = np.eye(3); C = np.array([[1.,0.,0.],[0.,0.,1.]], dtype=float)

    x = np.tile(np.array([a0, 0.0, b0], float), (T,1))

    for _ in range(max(1, int(iters))):
        x_pred = np.zeros((T,3)); P_pred = np.zeros((T,3,3))
        x_filt = np.zeros((T,3)); P_filt = np.zeros((T,3,3))

        x_prev = np.array([a0, 0.0, b0], float)
        P_prev = np.diag([10.0, 5.0, 5.0])

        for t, dt_t in enumerate(dts):
            dt_t = float(max(dt_t, 1.0))
            F = np.array([[1., dt_t, 0.],
                          [0., 1.,   0.],
                          [0., 0.,   1.]], dtype=float)
            Q = np.array([[q_LL*dt_t**3/3., q_LL*dt_t**2/2., 0.],
                          [q_LL*dt_t**2/2., q_LL*dt_t,       0.],
                          [0.,              0.,              q_b*dt_t]], dtype=float)

            x_pr = F @ x_prev
            P_pr = F @ P_prev @ F.T + Q

            day = dates[t]
            sub = g[g["date"] == day]
            if sub.empty:
                x_upd, P_upd = x_pr.copy(), P_pr.copy()
            else:
                ys = _as_1d_f64(sub["count"].to_numpy(copy=False))
                ns = _as_1d_f64(sub["coverage"].to_numpy(copy=False))
                ws = sub["w"].to_numpy(float, copy=False)

                y_eff, n_eff, mask_low = _apply_hygiene_for_pseudo(ys, ns, min_cov, cap_quantile)

                a_lin = float(np.clip(x[t,0], -_A_MAX, _A_MAX))
                b_lin = float(np.clip(x[t,2], -_B_MAX, _B_MAX))
                a_arr = jnp.array(np.full(y_eff.shape[0], a_lin))
                b_arr = jnp.array(np.full(y_eff.shape[0], b_lin))
                y_arr = jnp.array(y_eff, dtype=jnp.float64)
                n_arr = jnp.array(n_eff, dtype=jnp.float64)
                Zs, Rs = _vmap_pseudo(a_arr, b_arr, y_arr, n_arr)
                Zs = np.array(Zs); Rs = np.array(Rs)

                for i in range(Zs.shape[0]):
                    R_i = Rs[i]
                    if (mask_low is not None) and mask_low[i]:
                        R_i = R_i + np.diag([_BIG_R, _BIG_R])
                    w_i = float(max(ws[i], _MIN_W))
                    R_i = R_i / w_i
                    S = C @ P_pr @ C.T + R_i + _JITTER*np.eye(2)
                    try:
                        S_inv = np.linalg.inv(S)
                    except Exception:
                        S_inv = np.linalg.pinv(S)
                    K = P_pr @ C.T @ S_inv
                    z = Zs[i]
                    innov = z - (C @ x_pr)
                    x_pr = x_pr + K @ innov
                    x_pr[0] = float(np.clip(x_pr[0], -_A_MAX, _A_MAX))
                    x_pr[2] = float(np.clip(x_pr[2], -_B_MAX, _B_MAX))
                    P_pr = (I3 - K @ C) @ P_pr

                x_upd, P_upd = x_pr, P_pr

            x_pred[t], P_pred[t] = F @ x_prev, F @ P_prev @ F.T + Q
            x_filt[t], P_filt[t] = x_upd, P_upd
            x_prev, P_prev = x_upd, P_upd

        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        for t in range(T-2, -1, -1):
            dt_tp1 = float(max(dts[t+1], 1.0))
            F = np.array([[1., dt_tp1, 0.],
                          [0., 1.,     0.],
                          [0., 0.,     1.]], dtype=float)
            Ppr_next = P_pred[t+1]
            try:
                invPpr = np.linalg.inv(Ppr_next + _JITTER*np.eye(3))
            except Exception:
                invPpr = np.linalg.pinv(Ppr_next + _JITTER*np.eye(3))
            A = P_filt[t] @ F.T @ invPpr
            x_smooth[t] += A @ (x_smooth[t+1] - (F @ x_filt[t]))
            x_smooth[t,0] = float(np.clip(x_smooth[t,0], -_A_MAX, _A_MAX))
            x_smooth[t,2] = float(np.clip(x_smooth[t,2], -_B_MAX, _B_MAX))
            P_smooth[t] += A @ (P_smooth[t+1] - Ppr_next) @ A.T

        x = x_smooth

    return x_smooth, P_smooth, list(dates)

# ---------------- q tuning by predictive likelihood ----------------
def _estimate_time_shift_delta(y, n, dates, g_a_vec, max_shift_days=28, eps=1e-9):
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    T = len(dates)
    if T == 0:
        return 0
    g = np.asarray(g_a_vec, float)
    best_ll, best_d = -np.inf, 0
    for d in range(-max_shift_days, max_shift_days+1):
        if d < 0:
            a_shift = np.concatenate([np.full((-d), g[0]), g[:T+d]])
        elif d > 0:
            a_shift = np.concatenate([g[d:], np.full(d, g[-1])])
        else:
            a_shift = g
        mu = 1.0/(1.0 + np.exp(-np.clip(a_shift, -_A_MAX, _A_MAX)))
        mu = np.clip(mu, eps, 1-eps)
        kap = 1e3
        ll = betabinom.logpmf(y.astype(int), n.astype(int), (mu*kap), ((1.0-mu)*kap))
        s = float(np.sum(ll[np.isfinite(ll)])) if ll.size else -np.inf
        if s > best_ll:
            best_ll, best_d = s, d
    return int(best_d)

def _build_global_a_lookup(tl_global: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if tl_global.empty: return out
    for mut, g in tl_global.groupby("mutation", sort=False):
        s = pd.Series(g["a_t"].to_numpy(float), index=pd.to_datetime(g["date"])).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[str(mut)] = s
    return out

def _global_a_for_dates(mutation: str, dates: List[pd.Timestamp],
                        lut: Dict[str, pd.Series], default_a: float) -> np.ndarray:
    ds = pd.to_datetime(pd.Series(dates))
    if mutation not in lut:
        return np.full(ds.shape[0], float(default_a), dtype=float)
    s = lut[mutation]
    vals = s.reindex(ds, method="ffill")
    if vals.isna().any():
        vals = vals.fillna(method="bfill")
    vals = vals.fillna(float(default_a))
    return vals.to_numpy(float)

def _tune_process_noise_grid(ps_list: List[PreparedSeries], pri_map: pd.DataFrame,
                             lut_global_a: Dict[str, pd.Series], eps: float,
                             qLL_grid=(1e-4,5e-4,1e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1),
                             qb_grid=(1e-5,5e-5,1e-4,5e-4,1e-3,5e-3),
                             max_series=6, min_cov: Optional[float]=10.0,
                             cap_quantile: Optional[float]=0.999,
                             max_shift_days: int = 28) -> pd.DataFrame:
    rng = np.random.RandomState(123)
    buckets = {}
    for ps in ps_list:
        if ps.site_id == _GLOBAL_SITE_ID:
            continue
        buckets.setdefault(ps.mutation, []).append(ps)

    best = {}
    for mut, ser in buckets.items():
        if len(ser) > max_series:
            ser = list(rng.choice(ser, size=max_series, replace=False))
        if mut in pri_map.index:
            kappa_m = float(pri_map.loc[mut, "kappa"])
            mu0_m   = float(pri_map.loc[mut, "mu"])
            a_default = float(pri_map.loc[mut, "a_map"])
        else:
            y_all = np.concatenate([s.y for s in ser]) if ser else np.array([0.0])
            n_all = np.concatenate([s.n for s in ser]) if ser else np.array([1.0])
            mu0_m = float(np.clip(np.sum(y_all)/max(np.sum(n_all),1.0), eps, 1-eps))
            kappa_m = 1e3
            a_default = _safe_logit(mu0_m, eps)

        best_ll, best_pair = -np.inf, (None, None)
        for qLL in qLL_grid:
            for qb in qb_grid:
                ssum = 0.0
                for ps in ser:
                    g_vec = _global_a_for_dates(mut, ps.dates, lut_global_a, default_a=a_default)
                    d_opt = _estimate_time_shift_delta(ps.y, ps.n, ps.dates, g_vec, max_shift_days=max_shift_days, eps=eps)
                    if d_opt != 0:
                        if d_opt < 0:
                            g_vec = np.concatenate([np.full((-d_opt), g_vec[0]), g_vec[:len(g_vec)+d_opt]])
                        else:
                            g_vec = np.concatenate([g_vec[d_opt:], np.full(d_opt, g_vec[-1])])

                    xd, Pd, _ = _dynamic_bb_series_ct_rw2_with_offset(
                        ps.y, ps.n, ps.dates, g_vec,
                        kappa_init=kappa_m, mu_init=mu0_m,
                        q_LL=float(qLL), q_b=float(qb),
                        iters=3, eps=eps,
                        min_cov=min_cov, cap_quantile=cap_quantile)
                    a_t = g_vec + xd[:,0]
                    b_t = xd[:,2]
                    mu_t = 1/(1+np.exp(-np.clip(a_t, -_A_MAX, _A_MAX)))
                    kap_t = np.clip(np.exp(np.clip(b_t, -_B_MAX, _B_MAX)), _KAPPA_MIN, _KAPPA_MAX)
                    ll = betabinom.logpmf(ps.y.astype(int), ps.n.astype(int), (mu_t*kap_t), ((1.0-mu_t)*kap_t))
                    ssum += float(np.sum(ll[np.isfinite(ll)]))
                if ssum > best_ll:
                    best_ll, best_pair = ssum, (qLL, qb)
        if best_pair[0] is None:
            best_pair = (qLL_grid[0], qb_grid[0])
        best[mut] = {"q_LL_hat": float(best_pair[0]), "q_b_hat": float(best_pair[1])}
    return pd.DataFrame([{"mutation": m, **v} for m,v in best.items()])

# ---------------- Defaults & main ----------------
def _defaults(pri: Dict) -> Dict:
    d = {
        "data_root": "data", "results_root": "results", "seed": _SEED, "eps": _EPS_DEFAULT,
        "max_iter_mle": 1000, "tol_mle": 1e-8, "eb_min_tau": 1e-4,
        "calibrate_kappa_to": 0.80, "calibration_iters": 6,
        "kappa_scale_min": 1/8, "kappa_scale_max": 8.0,
        "kappa_global_cap_lo": 0.05, "kappa_global_cap_hi": 20.0,
        "q_a_init": 0.05, "q_b_init": 1e-3,
        "q_iters": 2, "laplace_iek_iters": 3,
        "calibration_max_rows": 200_000,
        # Hygiene
        "min_coverage_for_update": 10,
        "cap_coverage_quantile": 0.999,
        # WW weights
        "served_population_column": None,
        "flow_column": None,
        "qc_pass_column": None,
        # δ search
        "max_shift_days": 28,
        # Tuning
        "tune_q_max_series": 6,
        "tune_qLL_grid": [1e-4,5e-4,1e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1],
        "tune_qb_grid":  [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3],
        # Gates
        "gates": {"coverage_tol": 0.02, "pit_ks_p_min": 0.10, "z_var_low": 0.95, "z_var_high": 1.05, "delta_ll_q50_min": 0.0},
        # Exhaustive outputs
        "save_detail": True,
    }
    d.update(pri or {})
    return d

def run_priors(cfg: Dict, ctx: RunContext) -> Dict:
    pri = _defaults(cfg.get("priors", cfg))
    np.random.seed(int(pri["seed"]))
    eps = float(pri["eps"])
    mle_cfg = MLEConfig(max_iter=int(pri["max_iter_mle"]), tol=float(pri["tol_mle"]), eps=eps)

    ctx.log(level="INFO", message="Loading counts")
    df = _read_and_prepare_feature_table(ctx, pri)

    outroot = os.path.join(pri.get("results_root","results"), "priors")
    outdir  = os.path.join(outroot, "tables")
    os.makedirs(outdir, exist_ok=True)
    def _write(name: str, dfobj: pd.DataFrame): dfobj.to_csv(os.path.join(outdir, name), index=False)

    # Per-mutation subfolders for detail dumps
    per_mut_root = os.path.join(outdir, "per_mutation")
    for sub in ["mle", "map", "global", "time_local", "residuals", "fit_ll"]:
        os.makedirs(os.path.join(per_mut_root, sub), exist_ok=True)

    def _ensure_cols(df: Optional[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(columns=columns)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns].copy()

    def _mu_ci_from_var_a(mu: np.ndarray, var_a: np.ndarray, z: float = 1.96):
        mu = np.asarray(mu, float)
        var_a = np.maximum(np.asarray(var_a, float), 0.0)
        se_mu = np.sqrt(np.maximum((mu*(1.0-mu))**2 * var_a, 0.0))
        lo = np.clip(mu - z*se_mu, 0.0, 1.0); hi = np.clip(mu + z*se_mu, 0.0, 1.0)
        return lo, hi

    def _kappa_ci_from_var_b(kappa: np.ndarray, var_b: np.ndarray, z: float = 1.96):
        kappa = np.asarray(kappa, float)
        var_b = np.maximum(np.asarray(var_b, float), 0.0)
        se_b = np.sqrt(var_b)
        lo = kappa * np.exp(-z * se_b)
        hi = kappa * np.exp(+z * se_b)
        return lo, hi

    prepared_mut, prepared_series, daily = _prepare_arrays_for_priors(df)

    # (1) Robust per-mutation MLE → EB
    ctx.log(level="INFO", message="Per-mutation robust MLE \u2192 EB")
    mle_rows=[]
    for pm in prepared_mut:
        try:
            y, n = pm.y, pm.n
            if y.size == 1 and n.size == 1 and n[0] == 1.0 and y[0] == 0.0:
                mle_rows.append({"mutation": pm.mutation, "n_rows": 0,
                                 "mu_hat_raw": 0.5, "kappa_hat_raw": 1e3,
                                 "a_hat": 0.0, "b_hat": _safe_b_from_kappa(1e3),
                                 "H": np.diag([1.0, 1.0]), "y_vec": y, "n_vec": n})
                continue
            mu0, k0 = _moments_start(y, n, eps)
            mu_hat, kap_hat, H = _mle_beta_binom_robust(y, n, mu0, k0, mle_cfg)
            mle_rows.append({"mutation": pm.mutation, "n_rows": int(y.size),
                             "mu_hat_raw": float(mu_hat), "kappa_hat_raw": float(kap_hat),
                             "a_hat": _safe_logit(mu_hat, eps),
                             "b_hat": _safe_b_from_kappa(kap_hat),
                             "H": H, "y_vec": y, "n_vec": n})
        except Exception:
            mle_rows.append({"mutation": pm.mutation, "n_rows": int(pm.y.size),
                             "mu_hat_raw": 0.5, "kappa_hat_raw": 1e3,
                             "a_hat": 0.0, "b_hat": _safe_b_from_kappa(1e3),
                             "H": np.diag([1.0, 1.0]), "y_vec": pm.y, "n_vec": pm.n})

    if mle_rows:
        mle_df_core = pd.DataFrame(
            [{"mutation": r["mutation"], "n_rows": r["n_rows"],
              "mu_hat_raw": r["mu_hat_raw"], "kappa_hat_raw": r["kappa_hat_raw"],
              "a_hat": r["a_hat"], "b_hat": r["b_hat"]} for r in mle_rows])
        _write("mle_per_mutation.csv", mle_df_core)
        _write("mle_curvature.csv", pd.DataFrame(
            [{"mutation": r["mutation"], "Haa": float(r["H"][0,0]), "Hab": float(r["H"][0,1]), "Hbb": float(r["H"][1,1])} for r in mle_rows]))
    else:
        mle_df_core = pd.DataFrame(columns=["mutation","n_rows","mu_hat_raw","kappa_hat_raw","a_hat","b_hat"])
        _write("mle_per_mutation.csv", mle_df_core)
        _write("mle_curvature.csv", pd.DataFrame(columns=["mutation","Haa","Hab","Hbb"]))

    # EB prior and per-mutation MAP
    mu_emp, S_pop, Si = _fit_re_normal_fullcov(mle_rows, min_tau=float(pri["eb_min_tau"]))
    _write("eb_population_prior.csv", pd.DataFrame([{
        "m_a": float(mu_emp[0]), "m_b": float(mu_emp[1]),
        "Saa": float(S_pop[0,0]), "Sab": float(S_pop[0,1]),
        "Sba": float(S_pop[1,0]), "Sbb": float(S_pop[1,1])
    }]))

    map_rows=[]
    for r in mle_rows:
        mu_map, kap_map, a_map, b_map = _map_fullcov_for_mut(r["y_vec"], r["n_vec"], r["a_hat"], r["b_hat"], mu_emp, Si, eps)
        map_rows.append({"mutation": r["mutation"], "a_map": a_map, "b_map": b_map,
                         "mu": mu_map, "kappa": kap_map,
                         "alpha": mu_map*kap_map, "beta": (1.0-mu_map)*kap_map})
    pri_global = (pd.DataFrame(map_rows).sort_values("mutation").reset_index(drop=True)
                  if map_rows else pd.DataFrame(columns=["mutation","a_map","b_map","mu","kappa","alpha","beta"]))
    _write("map_per_mutation.csv", pri_global[["mutation","a_map","b_map","mu","kappa"]].copy())
    pri_map = pri_global.set_index("mutation")

    # ---- (2) GLOBAL smoothing per mutation — MULTI‑SITE FUSION ----
    ctx.log(level="INFO", message="Global mutation trends (multi‑site fusion, population×flow×QC weighted)")
    weights_sd = _build_weight_table(
        df,
        served_col=pri.get("served_population_column"),
        flow_col=pri.get("flow_column"),
        qc_col=pri.get("qc_pass_column")
    )
    _write("weights_by_site_day.csv", _ensure_cols(weights_sd, ["site_id","date","pop","flow","qc","w"]))

    tl_global_recs = []
    qLL_init, qb_init = float(pri["q_a_init"]), float(pri["q_b_init"])
    min_cov = float(pri.get("min_coverage_for_update", 10))
    cap_q   = pri.get("cap_coverage_quantile", 0.999)

    for mut in sorted(df["mutation"].unique()):
        if mut in pri_map.index:
            kappa_m = float(pri_map.loc[mut, "kappa"])
            mu0_m   = float(pri_map.loc[mut, "mu"])
        else:
            gmut = daily[daily["mutation"] == mut]
            tot_n = float(np.sum(gmut["coverage"])) if not gmut.empty else 1.0
            tot_y = float(np.sum(gmut["count"])) if not gmut.empty else 0.0
            mu0_m = float(np.clip(tot_y/max(tot_n,1.0), eps, 1-eps)); kappa_m = 1e3

        xg, Pg, dates_g = _smooth_global_multisite(
            daily, weights_sd, mut, mu0_m, kappa_m,
            q_LL=qLL_init, q_b=qb_init,
            iters=int(pri["laplace_iek_iters"]), eps=eps,
            min_cov=min_cov, cap_quantile=cap_q)

        for i in range(len(dates_g)):
            a_t, v_t, b_t = float(xg[i,0]), float(xg[i,1]), float(xg[i,2])
            mu_t = _inv_logit(a_t, eps)
            k_t = _safe_kappa_from_b(b_t)
            tl_global_recs.append((_GLOBAL_SITE_ID, dates_g[i], mut, mu_t, k_t, a_t, b_t,
                                   float(max(Pg[i,0,0], 0.0)), float(max(Pg[i,2,2], 0.0)), float(Pg[i,0,2])))

    tl_global = (pd.DataFrame(tl_global_recs, columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab"])
                 if tl_global_recs else pd.DataFrame(columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab"]))

    lut_global_a = _build_global_a_lookup(tl_global)

    # ---- (3) Tune process noise (q_LL, q_b) per mutation ----
    ctx.log(level="INFO", message="Tuning process noise per mutation via predictive likelihood")
    qLL_b_per_mut = _tune_process_noise_grid(
        prepared_series, pri_map, lut_global_a, eps,
        qLL_grid=tuple(pri.get("tune_qLL_grid", [1e-3,5e-3,1e-2,2e-2,5e-2])),
        qb_grid=tuple(pri.get("tune_qb_grid",   [1e-4,5e-4,1e-3,5e-3])),
        max_series=int(pri.get("tune_q_max_series", 6)),
        min_cov=min_cov, cap_quantile=cap_q,
        max_shift_days=int(pri.get("max_shift_days", 28))
    )

    # ---- (4) Site deviations around global trend with tuned q ----
    ctx.log(level="INFO", message="Dynamic smoothing (site deviations around global trend, tuned q and δ alignment)")
    tl_recs=[]
    time_shift_recs = []
    q_map = qLL_b_per_mut.set_index("mutation").to_dict(orient="index") if not qLL_b_per_mut.empty else {}
    for ps in prepared_series:
        if ps.mutation in pri_map.index:
            kappa_m = float(pri_map.loc[ps.mutation, "kappa"])
            mu0_m   = float(pri_map.loc[ps.mutation, "mu"])
            a_default = float(pri_map.loc[ps.mutation, "a_map"])
        else:
            tot_n = float(np.sum(ps.n)); tot_y = float(np.sum(ps.y))
            mu0_m = float(np.clip(tot_y/max(tot_n,1.0), eps, 1-eps)); kappa_m = 1e3
            a_default = _safe_logit(mu0_m, eps)

        g_vec = _global_a_for_dates(ps.mutation, ps.dates, lut_global_a, default_a=a_default)

        delta = _estimate_time_shift_delta(ps.y, ps.n, ps.dates, g_vec,
                                           max_shift_days=int(pri.get("max_shift_days", 28)), eps=eps)
        time_shift_recs.append({"site_id": ps.site_id, "mutation": ps.mutation, "time_shift_days": int(delta)})
        if delta != 0:
            if delta < 0:
                g_vec = np.concatenate([np.full((-delta), g_vec[0]), g_vec[:len(g_vec)+delta]])
            else:
                g_vec = np.concatenate([g_vec[delta:], np.full(delta, g_vec[-1])])

        qLL_hat = float(q_map.get(ps.mutation, {}).get("q_LL_hat", qLL_init) or qLL_init)
        qb_hat  = float(q_map.get(ps.mutation, {}).get("q_b_hat",  qb_init) or qb_init)

        xd, Pd, dt_days = _dynamic_bb_series_ct_rw2_with_offset(ps.y, ps.n, ps.dates, g_vec,
                                                                kappa_init=kappa_m, mu_init=mu0_m,
                                                                q_LL=qLL_hat, q_b=qb_hat,
                                                                iters=int(pri["laplace_iek_iters"]), eps=eps,
                                                                min_cov=min_cov, cap_quantile=cap_q)
        for i in range(ps.y.size):
            d_t, v_t, b_t = float(xd[i,0]), float(xd[i,1]), float(xd[i,2])
            a_t = float(g_vec[i] + d_t)
            mu_t = _inv_logit(a_t, eps); k_t = _safe_kappa_from_b(b_t)
            tl_recs.append((ps.site_id, ps.dates[i], ps.mutation, mu_t, k_t, a_t, b_t,
                            float(max(Pd[i,0,0], 0.0)), float(max(Pd[i,2,2], 0.0)), float(Pd[i,0,2])))

    tl_dev = (pd.DataFrame(tl_recs, columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab"])
              if tl_recs else pd.DataFrame(columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab"]))

    _write("time_shifts.csv", _ensure_cols(pd.DataFrame(time_shift_recs), ["site_id","mutation","time_shift_days"]))

    tl = pd.concat([tl_global, tl_dev], axis=0, ignore_index=True) if not tl_global.empty else tl_dev.copy()

    # Series index
    if not daily.empty:
        ser_idx = (daily.groupby(["site_id","mutation"])
                   .agg(n_timepoints=("date","size"),
                        first_date=("date","min"),
                        last_date=("date","max"))
                   .reset_index())
        if not tl_global.empty:
            gidx = (tl_global.groupby(["site_id","mutation"])
                    .agg(n_timepoints=("date","size"),
                         first_date=("date","min"),
                         last_date=("date","max"))
                    .reset_index())
            ser_idx = pd.concat([ser_idx, gidx], ignore_index=True)
        dt_rows=[]
        for (site, mut), s in daily.groupby(["site_id","mutation"], sort=False):
            dts = _compute_dt_days(s["date"].tolist())
            dt_rows.append({"site_id":site, "mutation":mut,
                            "dt_min": float(dts.min()) if dts.size else np.nan,
                            "dt_med": float(np.median(dts)) if dts.size else np.nan,
                            "dt_max": float(dts.max()) if dts.size else np.nan})
        ser_idx = ser_idx.merge(pd.DataFrame(dt_rows), on=["site_id","mutation"], how="left")
        _write("smoothing_series_index.csv", ser_idx)
    else:
        _write("smoothing_series_index.csv", pd.DataFrame(columns=["site_id","mutation","n_timepoints","first_date","last_date","dt_min","dt_med","dt_max"]))

    # ---- (5) κ calibration (exact mid‑P) on stratified sample ----
    ctx.log(level="INFO", message="Global \u03ba calibration (exact BB mid-P, stratified sample)")
    rows_for_cal = df.merge(pri_global[["mutation","kappa"]], on="mutation", how="left")
    rows_for_cal = rows_for_cal.merge(tl[["site_id","date","mutation","mu_t"]], on=["site_id","date","mutation"], how="left")
    rows_for_cal = rows_for_cal.merge(pri_global[["mutation","mu"]], on="mutation", how="left", suffixes=("","_g"))
    rows_for_cal["mu_t"] = rows_for_cal["mu_t"].fillna(rows_for_cal["mu"])
    rows_for_cal["kappa"] = rows_for_cal["kappa"].fillna(1e3)

    scale, cal_trace = 1.0, pd.DataFrame(columns=["iter","s_lo","s_hi","s_mid","coverage_emp"])
    if 0.5 < float(pri["calibrate_kappa_to"]) < 0.99 and not rows_for_cal.empty:
        scale, cal_trace = _calibrate_kappa_global_exact_dynamic(
            rows_for_cal, target_q=float(pri["calibrate_kappa_to"]),
            iters=int(pri["calibration_iters"]),
            s_bounds=(float(pri["kappa_scale_min"]), float(pri["kappa_scale_max"])),
            return_trace=True, max_rows=int(pri["calibration_max_rows"]))
        scale = float(np.clip(scale, float(pri["kappa_global_cap_lo"]), float(pri["kappa_global_cap_hi"])))
    _write("kappa_calibration_trace.csv", cal_trace)

    # Apply κ scale
    pri_global["kappa"] = np.clip(pri_global["kappa"] * scale, _KAPPA_MIN, _KAPPA_MAX)
    pri_global["alpha"] = pri_global["mu"] * pri_global["kappa"]
    pri_global["beta"]  = (1.0 - pri_global["mu"]) * pri_global["kappa"]
    if not tl.empty: tl["kappa_t"] = np.clip(tl["kappa_t"] * scale, _KAPPA_MIN, _KAPPA_MAX)

    # Rebuild rows for diagnostics
    rows = df.merge(tl[["site_id","date","mutation","mu_t","kappa_t","var_a","var_b"]],
                    on=["site_id","date","mutation"], how="left") if not tl.empty else df.copy()
    rows = rows.merge(pri_global[["mutation","kappa","mu"]], on="mutation", how="left", suffixes=("","_mut"))
    rows["kappa_use"] = rows["kappa_t"].fillna(rows["kappa"].fillna(1e3))

    # Residuals & PIT
    mu_for_resid = rows["mu_t"].fillna(pri_global.set_index("mutation")["mu"]).to_numpy(float)
    vy = var_Y_beta_binom(rows["coverage"].to_numpy(), mu_for_resid, rows["kappa_use"].to_numpy())
    rows["var_Y_bb"] = vy
    rows["resid"] = (rows["count"] - rows["coverage"]*mu_for_resid) / np.sqrt(np.maximum(vy, _EPS_DEFAULT))
    rows["pit_mid"] = _midp_from_exact_vec_fast(rows["count"].to_numpy(int),
                                                rows["coverage"].to_numpy(int),
                                                mu_for_resid, rows["kappa_use"].to_numpy(float))

    # ΔLL (per-mutation summary & per-row detail)
    fit_rows: List[Dict] = []
    detail_ll_rows: List[Dict] = []
    for mut, g in rows.groupby("mutation", sort=False):
        y = g["count"].to_numpy(np.float64, copy=False)
        n = g["coverage"].to_numpy(np.float64, copy=False)
        w = _make_w(n)
        mu_dyn = g["mu_t"].fillna(np.clip(np.sum(y)/max(np.sum(n),1.0), _EPS_DEFAULT, 1-_EPS_DEFAULT)).to_numpy(float)
        k_mut  = float(pri_global.loc[pri_global["mutation"]==mut, "kappa"].iloc[0]) if mut in pri_global["mutation"].values else 1e3
        ll_bb  = betabinom.logpmf(y.astype(int), n.astype(int), (mu_dyn*k_mut).astype(float), ((1.0-mu_dyn)*k_mut).astype(float))
        mu_bin = float(np.clip(np.sum(y) / max(np.sum(n), 1.0), _EPS_DEFAULT, 1 - _EPS_DEFAULT))
        ll_bin = _binom_logpmf(y.astype(int), n.astype(int), mu_bin, _EPS_DEFAULT)
        mask_bb, mask_bin = np.isfinite(ll_bb), np.isfinite(ll_bin)
        ll_bb_w  = float(np.sum(w[mask_bb]  * ll_bb[mask_bb])) if np.any(mask_bb)  else -np.inf
        ll_bin_w = float(np.sum(w[mask_bin] * ll_bin[mask_bin])) if np.any(mask_bin) else -np.inf
        fit_rows.append({"mutation": mut, "delta_ll": float(ll_bb_w - ll_bin_w),
                         "ll_betabinom_dynamic": ll_bb_w, "ll_binom_pooled": ll_bin_w})

        for i, idx in enumerate(g.index):
            detail_ll_rows.append({
                "site_id": g.loc[idx, "site_id"],
                "date": g.loc[idx, "date"],
                "mutation": mut,
                "ll_bb": float(ll_bb[i]) if np.isfinite(ll_bb[i]) else np.nan,
                "ll_bin": float(ll_bin[i]) if np.isfinite(ll_bin[i]) else np.nan,
                "delta_ll": float(ll_bb[i] - ll_bin[i]) if np.isfinite(ll_bb[i]) and np.isfinite(ll_bin[i]) else np.nan,
                "mu_t": float(mu_dyn[i]),
                "kappa_use": float(k_mut),
                "count": int(y[i]),
                "coverage": int(n[i]),
                "weight": float(w[i]),
            })

    tbl_fitll = pd.DataFrame(fit_rows).sort_values("mutation").reset_index(drop=True) if fit_rows else \
                pd.DataFrame(columns=["mutation","delta_ll","ll_betabinom_dynamic","ll_binom_pooled"])

    # Gates & stratified stats
    stats = _compute_stratified_stats(rows[["count","coverage","date","mu_t","kappa_use"]].copy(),
                                      target_q=float(pri["calibrate_kappa_to"]))
    gates = _check_gates(stats, pri.get("gates", {}))
    dll_ok = True if tbl_fitll.empty else bool(np.nanpercentile(tbl_fitll["delta_ll"].to_numpy(float), 50) >= float(gates["dll_q50_min"]))
    gates_summary = pd.DataFrame([{
        "coverage_ok": bool(gates["coverage_ok"]),
        "pit_ok": bool(gates["pit_ok"]),
        "z_ok": bool(gates["z_ok"]),
        "delta_ll_median_ok": bool(dll_ok),
        "global_pit_ks_p": float(stats["global_pit_ks_p"]),
        "global_coverage": float(stats["global_coverage"]),
        "z_mean": float(stats["z_mean"]),
        "z_var": float(stats["z_var"]),
        "rows_eval": int(stats["rows_eval"]),
    }])

    # Core outputs (contract)
    pri_global[["mutation","mu","kappa","alpha","beta"]].to_csv(os.path.join(outdir, "priors_hyperparams.csv"), index=False)
    (tl if not tl.empty else pd.DataFrame(columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab"])
     ).to_csv(os.path.join(outdir, "priors_time_local.csv"), index=False)

    base_cols = ["site_id","date","mutation","count","coverage"]
    if "sample_id" in df.columns: base_cols.insert(2, "sample_id")
    residual_cols = base_cols + ["af","mu_t","kappa_use","resid"]
    pit_cols      = base_cols + ["pit_mid"]

    df_base = df[base_cols + (["af"] if "af" in df.columns else [])].copy()
    merged = rows.merge(df_base, on=[c for c in base_cols if c in rows.columns], how="left", suffixes=("", "_orig"))
    if "af_orig" in merged.columns:
        merged["af"] = merged["af_orig"]; merged = merged.drop(columns=["af_orig"])
    merged = merged.rename(columns={"kappa_use":"kappa"})
    merged[[c for c in residual_cols if c in merged.columns]].to_csv(os.path.join(outdir, "residuals.csv"), index=False)
    merged[[c for c in pit_cols if c in merged.columns]].to_csv(os.path.join(outdir, "prior_predictive_pit.csv"), index=False)

    tbl_fitll.to_csv(os.path.join(outdir, "fit_ll.csv"), index=False)
    gates_summary.to_csv(os.path.join(outdir, "gates_summary.csv"), index=False)
    stats["cov_by_strata"].to_csv(os.path.join(outdir, "coverage_by_strata.csv"), index=False)
    stats["pit_ks_by_strata"].to_csv(os.path.join(outdir, "pit_ks_by_strata.csv"), index=False)

    if not qLL_b_per_mut.empty:
        q_LL_global = float(np.nanmedian(qLL_b_per_mut["q_LL_hat"]))
        q_b_global  = float(np.nanmedian(qLL_b_per_mut["q_b_hat"]))
    else:
        q_LL_global, q_b_global = float(pri["q_a_init"]), float(pri["q_b_init"])
    pd.DataFrame([{"q_a_hat": float(q_LL_global), "q_b_hat": float(q_b_global)}]).to_csv(os.path.join(outdir, "process_noise_estimates.csv"), index=False)

    if not qLL_b_per_mut.empty:
        _write("process_noise_by_mutation.csv", _ensure_cols(qLL_b_per_mut, ["mutation","q_LL_hat","q_b_hat"]))
    else:
        _write("process_noise_by_mutation.csv", pd.DataFrame(columns=["mutation","q_LL_hat","q_b_hat"]))

    # -------------------- EXHAUSTIVE DETAIL TABLES --------------------
    if pri.get("save_detail", True):
        mle_df_core_small = mle_df_core[["mutation","mu_hat_raw","kappa_hat_raw","a_hat","b_hat"]].copy()
        detail_mle = df[["site_id","date","mutation","coverage","count","af"]].merge(mle_df_core_small, on="mutation", how="left")
        detail_mle.to_csv(os.path.join(outdir, "detail_mle_observations.csv"), index=False)
        for mut, g in detail_mle.groupby("mutation", sort=False):
            g.to_csv(os.path.join(per_mut_root, "mle", f"{mut}.csv"), index=False)

        eb_prior = pd.DataFrame([{
            "m_a": float(mu_emp[0]), "m_b": float(mu_emp[1]),
            "Saa": float(S_pop[0,0]), "Sab": float(S_pop[0,1]), "Sbb": float(S_pop[1,1])
        }])
        detail_map = pri_global.merge(eb_prior.assign(key=1), how="left", left_on=None, right_on=None)
        if "key" in detail_map: detail_map = detail_map.drop(columns=["key"])
        detail_map.to_csv(os.path.join(outdir, "detail_map_permutation.csv"), index=False)
        for mut, g in detail_map.groupby("mutation", sort=False):
            g.to_csv(os.path.join(per_mut_root, "map", f"{mut}.csv"), index=False)

        if not tl.empty:
            g = tl.copy()
            mu = g["mu_t"].to_numpy(float); kappa = g["kappa_t"].to_numpy(float)
            var_a = g["var_a"].to_numpy(float); var_b = g["var_b"].to_numpy(float)
            mu_lo, mu_hi = _mu_ci_from_var_a(mu, var_a, 1.96)
            k_lo, k_hi = _kappa_ci_from_var_b(kappa, var_b, 1.96)
            g["mu_lo"] = mu_lo; g["mu_hi"] = mu_hi
            g["kappa_lo"] = k_lo; g["kappa_hi"] = k_hi
            w_agg = weights_sd.groupby("date")["w"].agg(["count","sum"]).reset_index().rename(columns={"count":"n_sites","sum":"sum_w"})
            g = g.merge(w_agg, on="date", how="left")
            g["q_LL_used"] = qLL_init; g["q_b_used"] = qb_init
            g.to_csv(os.path.join(outdir, "detail_global_timeseries.csv"), index=False)
            for mut, gr in g.groupby("mutation", sort=False):
                gr.to_csv(os.path.join(per_mut_root, "global", f"{mut}.csv"), index=False)
        else:
            pd.DataFrame(columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab",
                                  "mu_lo","mu_hi","kappa_lo","kappa_hi","n_sites","sum_w","q_LL_used","q_b_used"]
                         ).to_csv(os.path.join(outdir, "detail_global_timeseries.csv"), index=False)

        if not tl.empty:
            tld = tl.copy()
            mu = tld["mu_t"].to_numpy(float); kappa = tld["kappa_t"].to_numpy(float)
            var_a = tld["var_a"].to_numpy(float) if "var_a" in tld.columns else np.zeros(len(tld))
            var_b = tld["var_b"].to_numpy(float) if "var_b" in tld.columns else np.zeros(len(tld))
            mu_lo, mu_hi = _mu_ci_from_var_a(mu, var_a, 1.96)
            k_lo, k_hi = _kappa_ci_from_var_b(kappa, var_b, 1.96)
            tld["mu_lo"] = mu_lo; tld["mu_hi"] = mu_hi
            tld["kappa_lo"] = k_lo; tld["kappa_hi"] = k_hi
            ts_df = pd.DataFrame(time_shift_recs) if len(time_shift_recs) else pd.DataFrame(columns=["site_id","mutation","time_shift_days"])
            tld = tld.merge(ts_df, on=["site_id","mutation"], how="left")
            tld = tld.merge(weights_sd[["site_id","date","w"]], on=["site_id","date"], how="left")
            q_used = qLL_b_per_mut.set_index("mutation") if not qLL_b_per_mut.empty else pd.DataFrame(columns=["q_LL_hat","q_b_hat"])
            tld = tld.merge(q_used[["q_LL_hat","q_b_hat"]], on="mutation", how="left")
            tld["q_LL_used"] = tld["q_LL_hat"].fillna(qLL_init); tld["q_b_used"] = tld["q_b_hat"].fillna(qb_init)
            tld = tld.drop(columns=[c for c in ["q_LL_hat","q_b_hat"] if c in tld.columns])
            tld.to_csv(os.path.join(outdir, "detail_time_local.csv"), index=False)
            for mut, gr in tld.groupby("mutation", sort=False):
                gr.to_csv(os.path.join(per_mut_root, "time_local", f"{mut}.csv"), index=False)
        else:
            pd.DataFrame(columns=["site_id","date","mutation","mu_t","kappa_t","a_t","b_t","var_a","var_b","cov_ab",
                                  "mu_lo","mu_hi","kappa_lo","kappa_hi","time_shift_days","w","q_LL_used","q_b_used"]
                         ).to_csv(os.path.join(outdir, "detail_time_local.csv"), index=False)

        detail_resid_cols = base_cols + ["af","mu_t","kappa_use","var_Y_bb","resid","pit_mid"]
        rows_detail = rows.merge(df_base, on=[c for c in base_cols if c in rows.columns], how="left", suffixes=("", "_orig"))
        if "af_orig" in rows_detail.columns:
            rows_detail["af"] = rows_detail["af_orig"]; rows_detail = rows_detail.drop(columns=["af_orig"])
        rows_detail = rows_detail.rename(columns={"kappa_use":"kappa_use"})
        rows_detail = rows_detail[detail_resid_cols]
        rows_detail.to_csv(os.path.join(outdir, "detail_residuals.csv"), index=False)
        for mut, gr in rows_detail.groupby("mutation", sort=False):
            gr.to_csv(os.path.join(per_mut_root, "residuals", f"{mut}.csv"), index=False)

        detail_ll_df = pd.DataFrame(detail_ll_rows) if len(detail_ll_rows) else \
                       pd.DataFrame(columns=["site_id","date","mutation","ll_bb","ll_bin","delta_ll","mu_t","kappa_use","count","coverage","weight"])
        detail_ll_df.to_csv(os.path.join(outdir, "detail_fit_ll.csv"), index=False)
        for mut, gr in detail_ll_df.groupby("mutation", sort=False):
            gr.to_csv(os.path.join(per_mut_root, "fit_ll", f"{mut}.csv"), index=False)

        cal_trace_use = _ensure_cols(cal_trace if isinstance(cal_trace, pd.DataFrame) else pd.DataFrame(), 
                                     ["iter","s_lo","s_hi","s_mid","coverage_emp"])
        cal_trace_use["target_q"] = float(pri.get("calibrate_kappa_to", 0.80))
        cal_trace_use["scale_final"] = float(scale if "scale" in locals() else 1.0)
        cal_trace_use.to_csv(os.path.join(outdir, "detail_kappa_calibration.csv"), index=False)

        # Master full-detail join
        full = df.copy()
        full = full.merge(pri_global[["mutation","mu","kappa"]].rename(columns={"mu":"prior_mu","kappa":"prior_kappa"}),
                          on="mutation", how="left")
        full = full.merge(tl[["site_id","date","mutation","mu_t","kappa_t","var_a","var_b"]],
                          on=["site_id","date","mutation"], how="left")
        full = full.merge(weights_sd[["site_id","date","w"]], on=["site_id","date"], how="left")
        ts_df = pd.DataFrame(time_shift_recs) if len(time_shift_recs) else pd.DataFrame(columns=["site_id","mutation","time_shift_days"])
        full = full.merge(ts_df, on=["site_id","mutation"], how="left")
        full = full.merge(rows_detail, on=["site_id","date","mutation","coverage","count","af"], how="left")
        if "mu_t" in full.columns and "var_a" in full.columns:
            mu_arr = full["mu_t"].to_numpy(float)
            va_arr = full["var_a"].fillna(0.0).to_numpy(float)
            mu_lo, mu_hi = _mu_ci_from_var_a(mu_arr, va_arr, 1.96)
            full["post_mu_lo"] = mu_lo; full["post_mu_hi"] = mu_hi
        if "kappa_t" in full.columns and "var_b" in full.columns:
            kt_arr = full["kappa_t"].fillna(np.nan).to_numpy(float)
            vb_arr = full["var_b"].fillna(0.0).to_numpy(float)
            k_lo, k_hi = _kappa_ci_from_var_b(kt_arr, vb_arr, 1.96)
            full["post_kappa_lo"] = k_lo; full["post_kappa_hi"] = k_hi
        full.to_csv(os.path.join(outdir, "priors_full_detail.csv"), index=False)

    with open(os.path.join(outroot, "report.md"), "a", encoding="utf-8") as f:
        ks_p = stats["global_pit_ks_p"]
        f.write("\n".join([
            "# Dynamic hierarchical priors report (global multi‑site fusion + deviations, exact BB)",
            f"- Mutations: {int(df['mutation'].nunique())}",
            f"- Rows used: {int(df.shape[0])}",
            f"- Median κ (final): {pri_global['kappa'].median():.2f}" if not pri_global.empty else "- Median κ: NA",
            f"- ΔLL median (BB‑dynamic − Binomial): {tbl_fitll['delta_ll'].median():.3f}" if not tbl_fitll.empty else "- ΔLL: NA",
            f"- PIT KS p (exact mid‑P): {ks_p:.3g}" if np.isfinite(ks_p) else "- PIT: NA",
            ""
        ]) + "\n")

    ctx.log(level="INFO", message="Done", context={"outdir": outdir})
    gc.collect()
    return {
        "tables": ["priors_hyperparams", "priors_time_local", "residuals", "prior_predictive_pit", "fit_ll",
                   "gates_summary", "coverage_by_strata", "pit_ks_by_strata", "process_noise_estimates",
                   "mle_per_mutation", "mle_curvature", "eb_population_prior", "map_per_mutation",
                   "smoothing_series_index", "kappa_calibration_trace",
                   # detail & extras
                   "weights_by_site_day", "process_noise_by_mutation", "time_shifts",
                   "detail_mle_observations", "detail_map_permutation", "detail_global_timeseries",
                   "detail_time_local", "detail_residuals", "detail_fit_ll", "detail_kappa_calibration",
                   "priors_full_detail",
                ],
    }

# Optional CLI
if __name__ == "__main__":
    cfg = {"priors": {}}
    ctx = RunContext()
    run_priors(cfg, ctx)
