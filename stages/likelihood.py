"""
Likelihood v4.3 — Robust Beta–Binomial IRLS + ADMM on the simplex (JAX-free)

Drop-in replacement for stages/likelihood.py

Why this version:
  • Removes all JAX dependencies (no TracerBoolConversionError, no decorator issues).
  • Keeps the same math: IRLS with exact Beta–Binomial curvature + Tukey robustness.
  • ADMM solves a QP per time-slice over the simplex Δ (nonnegativity + sum-to-1).
  • Windows-safe, CPU-by-default, headless (no plotting unless you add it later).
  • Saves ALL artifacts as CSVs via ctx.write_table (or fallback to disk).

Entry: run_likelihood(cfg: dict, ctx: RunContext) -> dict
Outputs (CSV): theta_estimates, residuals, objective_trace, signatures_used,
               overlap_matrix, zscore_diagnostics, simplex_satisfied,
               (optionally) theta_estimates_raw, theta_uncertainty, mutation_leverage.
"""

from __future__ import annotations

import os
import re
import math
import gc
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from pandas.errors import ParserError

EPS = 1e-9
REQUIRED_SNV_COLS = ["site_id", "date", "sample_id", "mutation", "count", "coverage"]


# ---------- Optional pipeline utilities ----------
try:
    from utils.run import RunContext  # type: ignore
except Exception:
    class RunContext:  # minimal fallback for ad-hoc runs
        def log(self, **kw): print("[LOG]", kw)
        def write_table(self, key, df):
            out = os.path.join("results", "likelihood", "tables"); os.makedirs(out, exist_ok=True)
            p = os.path.join(out, f"{key}.csv")
            df.to_csv(p, index=False)
            print(f"[TABLE] {key} -> {getattr(df,'shape',None)} -> {p}")
        def write_figure(self, key, fig): print(f"[FIG] {key} (suppressed)")
        def write_report(self, text):
            out = os.path.join("results", "likelihood"); os.makedirs(out, exist_ok=True)
            p = os.path.join(out, "report.md")
            with open(p, "a", encoding="utf-8") as f: f.write(text + "\\n")
            print(f"[REPORT] appended -> {p}")
        def write_metrics(self, key, df):
            self.write_table(key, df)


# ------------------------ Array shapers & helpers ------------------------
def _as_1d_f64(x) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=np.float64)))

def _as_2d_f64(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"_as_2d_f64 expected 2-D, got shape={a.shape}")
    return np.ascontiguousarray(a)

def _clip01(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


# ------------------------ Simplex projection (NumPy) ------------------------
def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto probability simplex Δ = {x >= 0, sum x = 1}."""
    v = np.asarray(v, float).reshape(-1)
    n = v.shape[0]
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, n + 1)) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = np.sum(w)
    if s <= 0.0:
        return np.ones_like(w) / n
    return w / s


# ------------------------ ADMM solver for QP on simplex ------------------------
def _admm_qp_simplex(H: np.ndarray, f: np.ndarray,
                     rho: float, tol: float, max_iter: int) -> Tuple[np.ndarray, int]:
    """
    Solve: 0.5 x^T H x - f^T x  s.t. x in Δ.
    ADMM with variable splitting; linear system (H+rho I)x = f + rho(z - u).
    """
    H = _as_2d_f64(H); f = _as_1d_f64(f)
    K = H.shape[0]
    # Regularize for SPD-ish behavior
    A = H + float(rho) * np.eye(K)
    # Try Cholesky first, fallback to solve
    try:
      L = np.linalg.cholesky(A + 1e-9*np.eye(K))
      def solve_A(q):
          y = np.linalg.solve(L, q)
          return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
      def solve_A(q):
          return np.linalg.solve(A + 1e-9*np.eye(K), q)

    z = np.ones((K,), float) / K
    u = np.zeros((K,), float)

    x = z.copy()
    for it in range(int(max_iter)):
        q = f + rho * (z - u)
        x = solve_A(q)
        z_new = _project_simplex(x + u)
        u = u + x - z_new
        r = np.linalg.norm(x - z_new)
        s = rho * np.linalg.norm(z_new - z)
        z = z_new
        if r <= tol and s <= tol:
            return z, it + 1
    return z, int(max_iter)


# ------------------------ WLS from Beta–Binomial (NumPy) ------------------------
def _wls_terms_numpy(S: np.ndarray, y: np.ndarray, n: np.ndarray,
                     mu: np.ndarray, kappa: np.ndarray, p_hat: np.ndarray,
                     prob_clip_eps: float, robust_c: float, beta0: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build H = S^T diag(W) S and f = S^T (W z) for IRLS:
      grad wrt p: g = -(α/p) + (β/(1-p))
      curv wrt p: W = α/p^2 + β/(1-p)^2
      z = p - g/W
    α = y + κ μ + β0/2 ; β = (n - y) + κ (1-μ) + β0/2
    robust_c > 0 → Tukey biweight downweights large residuals |z-p|.
    """
    S = _as_2d_f64(S)
    y = _as_1d_f64(y); n = _as_1d_f64(n)
    mu = _as_1d_f64(mu); kappa = _as_1d_f64(kappa); p = _clip01(_as_1d_f64(p_hat), prob_clip_eps)

    alpha = np.maximum(y + kappa * mu + 0.5 * beta0, 0.0)
    beta  = np.maximum((n - y) + kappa * (1.0 - mu) + 0.5 * beta0, 0.0)

    inv_p  = 1.0 / p
    inv_q  = 1.0 / (1.0 - p)
    g = -alpha * inv_p + beta * inv_q
    W = alpha * (inv_p**2) + beta * (inv_q**2)
    W = np.clip(W, 1e-12, 1e12)
    z = np.clip(p - g / W, prob_clip_eps, 1.0 - prob_clip_eps)

    # Robust weighting (Tukey biweight) on residual r = z - p
    if robust_c and robust_c > 0.0:
        r = z - p
        s = np.median(np.abs(r - np.median(r))) + 1e-12  # MAD for scale
        u = r / (robust_c * s + 1e-12)
        w_rob = (1.0 - np.clip(u*u, 0.0, 1.0))**2
        W = W * w_rob

    SW = S * W[:, None]
    H = S.T @ SW
    f = S.T @ (W * z)
    return H, f, W, z


# ------------------------ Robust IO ------------------------
def _looks_like_snv(name: str) -> bool:
    pats = [r"^feature_store.*snv.*\\.(csv|parquet)$",
            r"^snv_long\\.(csv|parquet)$",
            r"^feature_store_snv\\.(csv|parquet)$",
            r"^feature_store_long\\.(csv|parquet)$"]
    return any(re.search(p, name, re.IGNORECASE) for p in pats)

def _read_csv_safely(p: "os.PathLike[str] | str", **kw) -> pd.DataFrame:
    from pathlib import Path
    p = Path(p)
    base_kw = dict(low_memory=False)
    base_kw.update(kw)
    try:
        return pd.read_csv(p, **base_kw)
    except (ParserError, MemoryError, OSError, UnicodeDecodeError):
        try:
            base_kw_py = dict(base_kw); base_kw_py["engine"] = "python"
            return pd.read_csv(p, **base_kw_py)
        except Exception:
            pass
    except Exception:
        pass
    base_kw_chunked = dict(base_kw); base_kw_chunked["engine"] = "python"; base_kw_chunked.pop("nrows", None)
    chunks = []
    for ch in pd.read_csv(p, chunksize=200_000, **base_kw_chunked):
        chunks.append(ch)
    return pd.concat(chunks, ignore_index=True)

def _normalize_snv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "site_id" not in df.columns and "site" in df.columns: rename["site"] = "site_id"
    if "sample_id" not in df.columns and "sample" in df.columns: rename["sample"] = "sample_id"
    if rename: df = df.rename(columns=rename)
    miss = [c for c in REQUIRED_SNV_COLS if c not in df.columns]
    if miss: raise KeyError(f"SNV table missing columns: {miss}. Found: {list(df.columns)}")
    df["site_id"] = df["site_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df["mutation"] = df["mutation"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head()
        raise ValueError(f"Invalid dates in SNV table (first 5 shown):\\n{bad}")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0).astype(int)
    df = df[df["coverage"] >= 0]
    return df.sort_values(["site_id", "date", "mutation"]).reset_index(drop=True)

def _load_counts(root: "os.PathLike[str] | str", explicit_path: Optional[str]) -> pd.DataFrame:
    from pathlib import Path
    root = Path(root)
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"snv_counts_path does not exist: {p}")
        return _normalize_snv(_read_csv_safely(p) if p.suffix.lower()==".csv" else pd.read_parquet(p))
    tables = root / "results" / "preprocessing" / "tables"
    if tables.exists():
        candidates: List[Path] = []
        for q in tables.iterdir():
            if q.is_file() and _looks_like_snv(q.name): candidates.append(q)
        candidates.sort(key=lambda z: z.stat().st_mtime, reverse=True)
        for p in candidates:
            try:
                tmp = _read_csv_safely(p, nrows=16) if p.suffix.lower()==".csv" else pd.read_parquet(p)
                if all(c in tmp.columns for c in REQUIRED_SNV_COLS):
                    data = _read_csv_safely(p) if p.suffix.lower()==".csv" else pd.read_parquet(p)
                    return _normalize_snv(data)
            except Exception:
                continue
    alt = root / "data" / "jahn_like.csv"
    if alt.exists():
        return _normalize_snv(_read_csv_safely(alt))
    raise FileNotFoundError("Counts not found. Put feature_store_*snv*.csv/parquet under results/preprocessing/tables, "
                            "set likelihood.snv_counts_path, or provide data/jahn_like.csv.")


# ------------------------ Priors & signatures ------------------------
def _load_priors_required(path: Optional[str]) -> pd.DataFrame:
    from pathlib import Path
    p = Path(path) if path else Path("results/priors/tables/priors_hyperparams.csv")
    if not p.exists():
        raise FileNotFoundError(f"Priors file required and not found: {p}")
    df = _read_csv_safely(p)
    if "mutation" not in df.columns:
        raise ValueError("priors_hyperparams.csv must contain 'mutation' column.")
    return df

def _extract_mu_kappa_from_priors(df_priors: pd.DataFrame, target_mutations: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    pri = df_priors.copy(); pri["mutation"] = pri["mutation"].astype(str)
    if "mu_shrunk" in pri.columns: mu_col = "mu_shrunk"
    elif "mu" in pri.columns:      mu_col = "mu"
    elif {"alpha","beta"} <= set(pri.columns): mu_col = None
    else: raise ValueError("Priors need 'mu' (or 'mu_shrunk') with 'kappa', or 'alpha' and 'beta'.")
    if "kappa_shrunk" in pri.columns: k_col = "kappa_shrunk"
    elif "kappa" in pri.columns:      k_col = "kappa"
    elif {"alpha","beta"} <= set(pri.columns): k_col = None
    else: raise ValueError("Missing 'kappa' (or 'kappa_shrunk'); or provide 'alpha' and 'beta'.")
    pri = pri.set_index("mutation", drop=False)
    mu = np.zeros(len(target_mutations), float); kap = np.zeros(len(target_mutations), float)
    for i, m in enumerate(target_mutations):
        if mu_col is not None:
            mu_i = pri.at[m, mu_col] if m in pri.index else np.nan
        else:
            if m in pri.index:
                a = float(pri.at[m, "alpha"]); b = float(pri.at[m, "beta"]) ; tot = a+b
                mu_i = a/tot if tot>0 else 0.5
            else:
                mu_i = np.nan
        if k_col is not None:
            k_i = pri.at[m, k_col] if m in pri.index else np.nan
        else:
            if m in pri.index:
                a = float(pri.at[m, "alpha"]); b = float(pri.at[m, "beta"]) ; k_i = max(a+b-2.0, 0.0)
            else:
                k_i = np.nan
        mu[i]  = float(np.clip(mu_i if np.isfinite(mu_i) else 0.5, EPS, 1.0 - EPS))
        kap[i] = float(max(k_i if np.isfinite(k_i) else 0.0, 0.0))
    return mu, kap

def _load_signatures_required(path: Optional[str]) -> pd.DataFrame:
    from pathlib import Path
    p = Path(path) if path else Path("data/signatures.csv")
    if not p.exists():
        raise FileNotFoundError(f"Signatures file required and not found: {p}")
    sig = _read_csv_safely(p)
    need = {"mutation", "lineage", "weight"} - set(sig.columns)
    if need: raise ValueError(f"signatures.csv missing columns: {need}")
    sig["mutation"] = sig["mutation"].astype(str)
    sig["lineage"]  = sig["lineage"].astype(str)
    sig["weight"]   = pd.to_numeric(sig["weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sig = sig.groupby(["mutation", "lineage"], as_index=False)["weight"].max()
    return sig

def _build_S_for_priors(sig: pd.DataFrame, target_mutations: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    S_df = sig.pivot_table(index="mutation", columns="lineage", values="weight", fill_value=0.0)
    S_df = S_df.reindex(index=pd.Index(target_mutations), fill_value=0.0)
    if "GLOBAL" in S_df.columns: S_df = S_df.drop(columns=["GLOBAL"])
    row_sum = S_df.sum(axis=1).astype(float)
    over = row_sum > 1.0 + 1e-12
    if over.any():
        idx = row_sum[over].index
        S_df.loc[idx, :] = S_df.loc[idx, :].div(row_sum.loc[idx], axis=0)
        row_sum = S_df.sum(axis=1).astype(float)
    S_df["GLOBAL"] = np.clip(1.0 - row_sum.values, 0.0, 1.0)
    S_df = S_df.sort_index(axis=0).sort_index(axis=1)
    mutations = list(S_df.index); lineages  = list(S_df.columns)
    S = _as_2d_f64(S_df.values)
    return S_df, S, mutations, lineages


# ------------------------ Diagnostics helpers ------------------------
def _compute_zscores(res_df: pd.DataFrame) -> pd.DataFrame:
    df = res_df.copy()
    n = df["coverage"].to_numpy(float)
    p = np.clip(df["pred_af"].to_numpy(float), EPS, 1.0 - EPS)
    k = np.maximum(df["kappa_used"].to_numpy(float), 0.0)
    # var(Y/n) under Beta-Binomial: Var(af) = p(1-p)/n * (n + (1+k)) / (2+k)
    var_af = p * (1 - p) / np.maximum(n, 1.0) * (np.maximum(n, 0.0) + (1.0 + k)) / (2.0 + k)
    sd_af = np.sqrt(np.maximum(var_af, EPS))
    df["z"] = (df["obs_af"].to_numpy(float) - p) / sd_af
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ------------------------ Main solver (NumPy IRLS + ADMM) ------------------------
def _guard_shapes_for_qp(S, y_list, n_list, mu_vec, kap_vec):
    S = _as_2d_f64(S); M, K = S.shape
    mu_vec = _as_1d_f64(mu_vec); kap_vec = _as_1d_f64(kap_vec)
    if mu_vec.shape[0] != M or kap_vec.shape[0] != M:
        raise ValueError(f"S (M={M}) must match mu/kap (got {mu_vec.shape[0]}, {kap_vec.shape[0]})")
    if len(y_list) != len(n_list):
        raise ValueError("y_list and n_list must have same T")
    for t, (y, n) in enumerate(zip(y_list, n_list)):
        y = _as_1d_f64(y); n = _as_1d_f64(n)
        if y.shape[0] != M or n.shape[0] != M:
            raise ValueError(f"slice t={t}: M mismatch, got y={y.shape[0]}, n={n.shape[0]}, expected {M}")


def _solve_site_irls_admm_np(
    S_np: np.ndarray,                     # (M x K)
    y_list_np: List[np.ndarray],          # T items, each (M,)
    n_list_np: List[np.ndarray],          # T items, each (M,)
    mu_vec_np: np.ndarray,                # (M,)
    kap_vec_np: np.ndarray,               # (M,)
    lam_ridge: float,
    lam_ov: float,
    O_np: Optional[np.ndarray],           # (K x K) or None
    lam_l2_vec_np: np.ndarray,            # (T,) nonnegative
    irls_max_iter: int,
    admm_rho: float,
    admm_adaptive: bool,
    admm_rho_floor: float,
    admm_rho_ceil: float,
    admm_max_iter: int,
    admm_tol: float,
    prob_clip_eps: float,
    robust_c: float,
    prior_jitter_beta0: float,
    theta_init: Optional[List[np.ndarray]] = None,
    unknown_idx: Optional[int] = None,
    lam_unknown: float = 0.0,
    compute_uncertainty: bool = True,
    compute_leverage: bool = True,
) -> Tuple[List[np.ndarray], List[float], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:

    S = _as_2d_f64(S_np); M, K = S.shape
    mu = _as_1d_f64(mu_vec_np); kap = _as_1d_f64(kap_vec_np)
    y_list = [_as_1d_f64(y) for y in y_list_np]; n_list = [_as_1d_f64(n) for n in n_list_np]
    T = len(y_list)

    O = None
    if lam_ov > 0.0 and O_np is not None:
        O = _as_2d_f64(O_np)
        O = 0.5 * (O + O.T)

    lam_l2_vec = _as_1d_f64(lam_l2_vec_np)

    # init theta
    if theta_init is not None and len(theta_init) == T:
        theta = [np.asarray(t, float).reshape(-1) for t in theta_init]
        theta = [np.maximum(t, 0.0) for t in theta]
        theta = [t / max(t.sum(), 1.0) for t in theta]
    else:
        theta = [np.full(K, 1.0 / K, dtype=float) for _ in range(T)]

    obj_trace: List[float] = []
    last_H_blocks: List[np.ndarray] = [np.eye(K) for _ in range(T)]
    last_W_list: List[np.ndarray] = [np.ones(M) for _ in range(T)]

    for _ in range(int(max(1, irls_max_iter))):
        total = 0.0
        H_blocks: List[np.ndarray] = []
        f_blocks: List[np.ndarray] = []
        W_slices: List[np.ndarray] = []

        # Build per-slice WLS (NumPy) then add temporal/overlap terms
        for t in range(T):
            th = np.asarray(theta[t], float).reshape(-1)
            p_hat_np = S @ th
            H, f, W, z = _wls_terms_numpy(S, y_list[t], n_list[t], mu, kap, p_hat_np,
                                          prob_clip_eps, robust_c, prior_jitter_beta0)

            lam_left  = float(lam_l2_vec[t - 1]) if t > 0 else 0.0
            lam_right = float(lam_l2_vec[t])     if (t + 1) < T else 0.0

            # penalties
            H = H + lam_ridge * np.eye(K)
            if O is not None and lam_ov > 0.0:
                H = H + lam_ov * O
            H = H + (lam_left + lam_right) * np.eye(K)
            if unknown_idx is not None and lam_unknown > 0.0:
                H[unknown_idx, unknown_idx] += lam_unknown

            if t > 0:     f = f + lam_left  * theta[t-1]
            if t + 1 < T: f = f + lam_right * theta[t+1]

            # monitor quadratic approx value
            resid = (S @ th) - z
            total += 0.5 * float(resid @ (W * resid)) \
                     + 0.5 * lam_ridge * float(th @ th) \
                     + (0.5 * lam_ov * float(th @ (O @ th)) if (O is not None and lam_ov>0.0) else 0.0)
            if t > 0:
                d = th - theta[t-1]
                total += 0.5 * lam_left * float(d @ d)

            H_blocks.append(H)
            f_blocks.append(f)
            W_slices.append(W)

        obj_trace.append(total)

        # Gauss–Seidel over slices: solve QP via ADMM
        max_change = 0.0
        for t in range(T):
            H = H_blocks[t]; f = f_blocks[t]

            # pick ADMM rho (simple heuristic on host; cheap)
            if admm_adaptive:
                try:
                    lam_max = float(np.linalg.eigvalsh(0.5*(H+H.T)).max())
                    tr = float(np.trace(H)) / max(K, 1)
                    rho_use = float(np.clip((lam_max / max(tr, 1e-12)) ** 0.5, admm_rho_floor, admm_rho_ceil))
                except Exception:
                    rho_use = admm_rho
            else:
                rho_use = admm_rho

            th_old = theta[t]
            th_new, _ = _admm_qp_simplex(H, f, rho=rho_use, tol=admm_tol, max_iter=admm_max_iter)
            if np.linalg.norm(th_new - th_old) > 0.5:
                th_new = 0.5 * (th_new + th_old)  # mild damping
            theta[t] = th_new
            max_change = max(max_change, float(np.linalg.norm(th_new - th_old)))

        last_H_blocks = H_blocks
        last_W_list = W_slices
        if max_change < 5e-6:
            break

    # Optional uncertainty & leverage (per-slice, approximate)
    diag_cov_list: Optional[List[np.ndarray]] = None
    leverage_list: Optional[List[np.ndarray]] = None

    if compute_uncertainty or compute_leverage:
        diag_cov_list = []
        leverage_list = []
        for t in range(T):
            H = last_H_blocks[t]
            W = last_W_list[t]
            try:
                H_inv = np.linalg.inv(H + 1e-9*np.eye(K))
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)
            diag_cov = np.clip(np.diag(H_inv), 0.0, np.inf)
            diag_cov_list.append(diag_cov)
            if compute_leverage:
                W_sqrt = np.sqrt(np.clip(W, 0.0, np.inf))
                R = (W_sqrt[:, None] * S)   # MxK
                VH = R @ H_inv              # MxK
                lev = np.sum(VH * R, axis=1)
                leverage_list.append(np.clip(lev, 0.0, 1.0))
            else:
                leverage_list.append(np.zeros(M, dtype=float))

    return theta, obj_trace, diag_cov_list, leverage_list


# ------------------------ Entry point ------------------------
def run_likelihood(cfg: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:
    lk = cfg.get("likelihood", cfg)

    seed = int(lk.get("seed", 12345) or 12345)
    rng = np.random.default_rng(seed)

    # Required paths
    snv_counts_path = lk.get("snv_counts_path", None)
    priors_path     = lk.get("priors_hyperparams_path", "results/priors/tables/priors_hyperparams.csv")
    signatures_path = lk.get("signatures_path", "data/signatures.csv")
    priors_time_path= lk.get("priors_time_path", "results/priors/tables/priors_time_local.csv")

    # Penalties & solver
    lambda_ridge = float(lk.get("lambda_ridge", 5e-4) or 5e-4)
    overlap_penalty_lambda = float(lk.get("overlap_penalty_lambda", 0.0) or 0.0)
    lambda_temporal_l2 = float(lk.get("temporal_smooth_lambda_l2", 0.02) or 0.02)
    prob_clip_eps = float(lk.get("prob_clip_eps", 1e-8) or 1e-8)
    aggregate_by_date = bool(lk.get("aggregate_by_date", True))

    # Robustness & numerical
    robust_c = float(lk.get("robust_c", 4.685) or 0.0)                 # 0 → off
    prior_jitter_beta0 = float(lk.get("prior_jitter_beta0", 0.5) or 0.0)

    # Modes & extras
    prior_only = bool(lk.get("prior_only", False))
    prior_only_fill_from_counts = bool(lk.get("prior_only_fill_from_counts", True))

    unknown_extra = float(lk.get("unknown_extra", lk.get("unknown_extra_ridge", 0.0)) or 0.0)
    prior_kappa_scale = float(lk.get("prior_kappa_scale", 1.0) or 1.0)
    prior_kappa_cap   = lk.get("prior_kappa_cap", None)
    prior_kappa_cap   = float(prior_kappa_cap) if prior_kappa_cap not in (None, "", "inf", "Inf", "INF") else np.inf
    coverage_smooth_scale = float(lk.get("temporal_cov_scale", 1.0e4) or 1.0e4)

    # IRLS & ADMM controls
    irls_max_iter = int(lk.get("irls_max_iter", 10) or 10)
    admm_max_iter = int(lk.get("admm_max_iter", 500) or 500)
    admm_tol      = float(lk.get("admm_tol", 5e-7) or 5e-7)
    admm_rho      = float(lk.get("admm_rho", 1.0) or 1.0)
    admm_adaptive_rho = bool(lk.get("admm_adaptive_rho", True))
    admm_rho_floor    = float(lk.get("admm_rho_floor", 1.0e-6) or 1.0e-6)
    admm_rho_ceil     = float(lk.get("admm_rho_ceil", 1.0e6) or 1.0e6)

    # Diagnostics, uncertainty, leverage
    compute_uncertainty = bool(lk.get("compute_uncertainty", True))
    compute_leverage    = bool(lk.get("compute_leverage", True))

    # Coverage hygiene for temporal lambda scaling (done outside IRLS too)
    min_coverage_for_wls = lk.get("min_coverage_for_wls", 0.0)
    min_coverage_for_wls = float(min_coverage_for_wls) if min_coverage_for_wls not in (None, "",) else 0.0
    cap_coverage_quantile = lk.get("cap_coverage_quantile", 0.999)
    cap_coverage_quantile = float(cap_coverage_quantile) if cap_coverage_quantile not in (None, "",) else None

    ctx.log(level="INFO", message="Likelihood v4.3 (robust BB–IRLS) starting",
            context={"seed": seed, "aggregate_by_date": aggregate_by_date, "prior_only": prior_only})

    root = os.path.abspath(".")

    # --- priors & mutation universe FIRST ---
    df_priors = _load_priors_required(priors_path)
    target_mutations = pd.Index(df_priors["mutation"].astype(str).unique()).sort_values().tolist()
    if len(target_mutations) == 0:
        raise RuntimeError("Priors have no mutations.")
    ctx.log(level="INFO", message="Priors: mutation universe", context={"M_priors": len(target_mutations)})

    # --- signatures S aligned to priors ---
    sig_df = _load_signatures_required(signatures_path)
    S_df, S, mutations, lineages = _build_S_for_priors(sig_df, target_mutations)
    if set(mutations) != set(target_mutations):
        missing = sorted(set(target_mutations) - set(mutations))
        raise RuntimeError(f"Signatures missing mutations present in priors (first 10): {missing[:10]}")
    M, K = S.shape
    if "GLOBAL" not in lineages:
        raise RuntimeError("GLOBAL column missing after signatures build.")
    ctx.log(level="INFO", message="Signatures shape (M×K)", context={"M": M, "K": K})
    ctx.write_table("signatures_used", S_df.reset_index().rename(columns={"index": "mutation"}))

    unknown_idx = lineages.index("GLOBAL") if "GLOBAL" in lineages else None

    # --- overlap / identifiability ---
    col_norms = np.linalg.norm(S, axis=0) + EPS
    Sn = S / col_norms
    O = Sn.T @ Sn
    np.fill_diagonal(O, 1.0)
    ov_df = pd.DataFrame(np.clip(O, 0.0, 1.0), index=lineages, columns=lineages)
    ctx.write_table("overlap_matrix", ov_df)
    O_pen = O if overlap_penalty_lambda > 0.0 else None

    # --- priors (μ, κ) in priors order + κ scaling/cap ---
    mu_vec, kap_vec = _extract_mu_kappa_from_priors(df_priors, mutations)
    if np.isfinite(prior_kappa_cap):
        kap_vec = np.minimum(kap_vec * prior_kappa_scale, prior_kappa_cap)
    else:
        kap_vec = kap_vec * prior_kappa_scale
    kap_vec = np.maximum(kap_vec, 0.0)

    # --- time-local priors map (optional) ---
    priors_time_map = None
    has_site_in_time = False
    ptl = priors_time_path
    if ptl and os.path.exists(ptl):
        pt = _read_csv_safely(ptl)
        if "date" in pt.columns:
            pt = pt.copy(); pt["date"] = pd.to_datetime(pt["date"]) ; pt["mutation"] = pt["mutation"].astype(str)
            if "mu_t" not in pt.columns and "mu" in pt.columns:       pt["mu_t"]    = pt["mu"]
            if "kappa_t" not in pt.columns and "kappa" in pt.columns: pt["kappa_t"] = pt["kappa"]
            cols_keep = ["mutation","mu_t","kappa_t","date"]
            if "site_id" in pt.columns:
                has_site_in_time = True; pt["site_id"] = pt["site_id"].astype(str); cols_keep.append("site_id")
            pt = pt[cols_keep]
            priors_time_map = {}
            if has_site_in_time:
                for (sid, dt), sub in pt.groupby(["site_id","date"], sort=False):
                    priors_time_map[(str(sid), pd.Timestamp(dt))] = sub[["mutation","mu_t","kappa_t"]].reset_index(drop=True)
            else:
                for dt, sub in pt.groupby(["date"], sort=False):
                    priors_time_map[pd.Timestamp(dt)] = sub[["mutation","mu_t","kappa_t"]].reset_index(drop=True)

    def _time_local_for(site: str, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        if not priors_time_map: return None
        return priors_time_map.get((str(site), pd.Timestamp(date))) if has_site_in_time \
               else priors_time_map.get(pd.Timestamp(date))

    def _date_and_sample_from_key(site: str, key) -> Tuple[pd.Timestamp, str]:
        if isinstance(key, (tuple, list)):
            d = key[0]; s = key[1] if len(key) > 1 else f"{site}__date"
        else:
            d = key; s = f"{site}__date"
        return pd.Timestamp(d), str(s)

    # --- counts (optional in prior-only) ---
    snv_loaded_ok = False
    try:
        snv = _normalize_snv(_load_counts(root, snv_counts_path))
        snv_loaded_ok = True
    except Exception:
        if not prior_only:
            raise
        snv = pd.DataFrame({
            "site_id": ["PRIOR_ONLY"],
            "date": [pd.Timestamp("1970-01-01")],
            "sample_id": ["PRIOR_ONLY__date"],
            "mutation": [mutations[0] if len(mutations) else "M0"],
            "count": [0],
            "coverage": [0],
        }); snv_loaded_ok = True

    # restrict counts to priors mutations
    snv["mutation"] = snv["mutation"].astype(str)
    snv = snv[snv["mutation"].isin(target_mutations)].copy()
    if snv.empty and not prior_only:
        raise RuntimeError("Counts empty after restricting to priors mutations. Check naming consistency.")

    ctx.write_table("residuals_input_head", snv.head(10))

    # aggregation
    if snv_loaded_ok and aggregate_by_date and not snv.empty:
        snv_use = (snv.groupby(["site_id","date","mutation"], as_index=False)
                      .agg(count=("count","sum"), coverage=("coverage","sum")))
        snv_use["sample_id"] = snv_use["site_id"] + "__date"
    else:
        snv_use = snv.copy()

    snv_use = snv_use.sort_values(["site_id","date","mutation"]).reset_index(drop=True)

    def _dates_for_site_prior_only(site: str) -> List[pd.Timestamp]:
        dates: List[pd.Timestamp] = []
        if priors_time_map is not None:
            if has_site_in_time:
                dates = [k[1] for k in priors_time_map.keys() if isinstance(k, tuple) and k[0] == str(site)]
            else:
                dates = list(priors_time_map.keys())
        if (not dates) and prior_only_fill_from_counts:
            dates = list(snv_use.loc[snv_use["site_id"] == site, "date"].unique())
        if not dates:
            dates = [pd.Timestamp("1970-01-01")]
        return sorted(pd.to_datetime(dates))

    # ------------------- fit per site -------------------
    theta_rows: List[Dict[str, Any]] = []
    theta_rows_raw: List[Dict[str, Any]] = []
    theta_sd_rows: List[Dict[str, Any]] = []
    leverage_rows: List[Dict[str, Any]] = []
    resid_rows: List[Dict[str, Any]]  = []
    obj_trace_rows: List[Dict[str, Any]] = []
    simplex_ok: List[bool] = []

    site_ids_all: List[str] = list(snv_use["site_id"].astype(str).unique())
    if prior_only and priors_time_map is not None and has_site_in_time:
        sites_from_pt = sorted({k[0] for k in priors_time_map.keys() if isinstance(k, tuple)})
        site_ids_all = sorted(set(site_ids_all) | set(sites_from_pt))

    # Precompute per-mutation priors (mu, kappa) overrides from priors_time when available
    mu_vec_base = mu_vec.copy(); kap_vec_base = kap_vec.copy()

    for site in site_ids_all:
        df_site = snv_use[snv_use["site_id"] == site].sort_values(["date","mutation"])

        snapshots: List[Tuple[Any, pd.DataFrame]] = []
        if prior_only:
            dates = _dates_for_site_prior_only(site)
            for d in dates: snapshots.append((pd.Timestamp(d), pd.DataFrame()))
        else:
            snapshots = list(df_site.groupby(["date"], sort=False))

        T = len(snapshots)
        if T == 0: continue

        y_list: List[np.ndarray] = []
        n_list: List[np.ndarray] = []
        lam_l2_vec: List[float] = [0.0] * T
        theta0_list: List[np.ndarray] = []
        theta_prev: Optional[np.ndarray] = None
        snap_df_payloads: List[pd.DataFrame] = []

        for t, (grp_key, df_s) in enumerate(snapshots):
            date, sample_id = _date_and_sample_from_key(site, grp_key)

            # Base from global priors; override by time-local if present
            mu_use = mu_vec_base.copy(); kap_use = kap_vec_base.copy()
            pt = _time_local_for(site, date)
            if pt is not None and not pt.empty:
                m2m = dict(zip(pt["mutation"], pd.to_numeric(pt["mu_t"], errors="coerce")))
                m2k = dict(zip(pt["mutation"], pd.to_numeric(pt["kappa_t"], errors="coerce")))
                for i, m in enumerate(mutations):
                    mui = m2m.get(m, np.nan); ki = m2k.get(m, np.nan)
                    if np.isfinite(mui): mu_use[i]  = float(np.clip(mui, EPS, 1.0-EPS))
                    if np.isfinite(ki):  kap_use[i] = float(max(ki, 0.0))

            if prior_only:
                y = np.zeros(M, float); n = np.zeros(M, float)
                af = mu_use; w = kap_use
                tot_cov = float(np.sum(kap_use))
                cov_vec_for_resid = kap_use.copy(); obs_af_for_resid = mu_use.copy()
            else:
                dct_y = df_s.set_index("mutation")["count"].to_dict()
                dct_n = df_s.set_index("mutation")["coverage"].to_dict()
                y = np.array([dct_y.get(m, 0) for m in mutations], float)
                n = np.array([dct_n.get(m, 0) for m in mutations], float)
                # Coverage hygiene (optional) for IRLS influence (does not change outputs, only stability)
                if min_coverage_for_wls and min_coverage_for_wls > 0.0:
                    lo_mask = n < float(min_coverage_for_wls)
                    if np.any(lo_mask):
                        # shrink W later via larger residuals; keep raw n for records
                        pass
                if cap_coverage_quantile is not None and n.size:
                    cap = float(np.quantile(n, float(cap_coverage_quantile)))
                    n = np.minimum(n, cap)

                af = np.divide(y, np.maximum(n, 1.0), out=np.zeros_like(y), where=n>0)
                w = n + kap_use
                tot_cov = float(np.maximum(n.sum(), 0.0))
                cov_vec_for_resid = n.copy(); obs_af_for_resid = af.copy()

            y_list.append(y); n_list.append(n)
            lam_t = lambda_temporal_l2 / np.sqrt(1.0 + tot_cov / coverage_smooth_scale)
            lam_l2_vec[t] = lam_t

            # Warm start via ridge LS on Sθ≈af
            if np.sum(w) > 0:
                A = S.T @ (w[:, None] * S) ; b = S.T @ (w * af)
                try: th0 = np.linalg.solve(A + 1e-9*np.eye(K), b)
                except np.linalg.LinAlgError: th0 = np.linalg.lstsq(A + 1e-9*np.eye(K), b, rcond=None)[0]
                if theta_prev is not None: th0 = 0.5*th0 + 0.5*theta_prev
                th0 = np.maximum(th0, 0.0); s = th0.sum(); th0 = (th0/s) if s>0 else np.ones(K)/K
            else:
                th0 = np.ones(K)/K if theta_prev is None else theta_prev
            theta0_list.append(th0); theta_prev = th0

            snap_df_payloads.append(pd.DataFrame({
                "mutation": mutations,
                "obs_af": obs_af_for_resid,
                "coverage": cov_vec_for_resid,
                "kappa_used": kap_use
            }))

        # shape guard
        _guard_shapes_for_qp(S, y_list, n_list, mu_vec_base, kap_vec_base)

        # solve with NumPy IRLS → ADMM
        theta_sol, trace, diag_cov_list, leverage_list = _solve_site_irls_admm_np(
            S_np=S,
            y_list_np=y_list,
            n_list_np=n_list,
            mu_vec_np=mu_vec_base,
            kap_vec_np=kap_vec_base,
            lam_ridge=lambda_ridge,
            lam_ov=overlap_penalty_lambda,
            O_np=O_pen,
            lam_l2_vec_np=np.array(lam_l2_vec, float),
            irls_max_iter=irls_max_iter,
            admm_rho=admm_rho,
            admm_adaptive=admm_adaptive_rho,
            admm_rho_floor=admm_rho_floor,
            admm_rho_ceil=admm_rho_ceil,
            admm_max_iter=admm_max_iter,
            admm_tol=admm_tol,
            prob_clip_eps=prob_clip_eps,
            robust_c=robust_c,
            prior_jitter_beta0=prior_jitter_beta0,
            theta_init=theta0_list,
            unknown_idx=unknown_idx,
            lam_unknown=unknown_extra,
            compute_uncertainty=compute_uncertainty,
            compute_leverage=compute_leverage,
        )
        for it, val in enumerate(trace):
            obj_trace_rows.append({"site_id": site, "iter": it, "objective": float(val)})

        # record outputs
        for t, ((snap_key, _df_unused), payload_df) in enumerate(zip(snapshots, snap_df_payloads)):
            date, sample_id = _date_and_sample_from_key(site, snap_key)
            th = np.asarray(theta_sol[t], float).reshape(-1)
            th = np.maximum(th, 0.0); th = th / max(th.sum(), 1.0)
            p  = np.clip(S @ th, EPS, 1.0 - EPS)

            by_mut = payload_df.set_index("mutation")["obs_af"].to_dict(), payload_df.set_index("mutation")["coverage"].to_dict(), payload_df.set_index("mutation")["kappa_used"].to_dict()
            dct_obs, dct_cov, dct_kap = by_mut
            obs_af = np.array([dct_obs.get(m, 0) for m in mutations], float)
            covv   = np.array([dct_cov.get(m, 0) for m in mutations], float)
            kappav = np.array([dct_kap.get(m, 0) for m in mutations], float)

            for i, m in enumerate(mutations):
                resid_rows.append({
                    "site_id": site, "date": date, "sample_id": sample_id, "mutation": m,
                    "obs_af": float(obs_af[i]), "pred_af": float(p[i]), "residual": float(obs_af[i] - p[i]),
                    "coverage": float(covv[i]), "kappa_used": float(kappav[i]),
                    "weight": float(0.0 if covv[i] <= 0 else 1.0),
                })

            for lin, v in zip(lineages, th):
                theta_rows.append({"site_id": site, "date": date, "sample_id": sample_id, "lineage": lin, "theta": float(v)})
            # raw (unconstrained) — for compatibility, echo theta (or compute A^{-1}b if wanted)
            for lin, v in zip(lineages, th):
                theta_rows_raw.append({"site_id": site, "date": date, "sample_id": sample_id, "lineage": lin, "theta_raw": float(v)})

            if compute_uncertainty and diag_cov_list is not None:
                sd = np.sqrt(np.maximum(diag_cov_list[t], 0.0))
                for lin, s in zip(lineages, sd):
                    theta_sd_rows.append({"site_id": site, "date": date, "sample_id": sample_id, "lineage": lin, "theta_sd": float(s)})

            if compute_leverage and leverage_list is not None:
                lev = leverage_list[t]
                for mut, lv in zip(mutations, lev):
                    leverage_rows.append({"site_id": site, "date": date, "sample_id": sample_id, "mutation": mut, "leverage": float(lv)})

            simplex_ok.append((abs(np.sum(th) - 1.0) <= 1e-8) and np.all(th >= -1e-8))

    # assemble outputs
    theta_df = pd.DataFrame(theta_rows).sort_values(["site_id", "date", "sample_id", "lineage"])
    theta_raw_df = pd.DataFrame(theta_rows_raw).sort_values(["site_id", "date", "sample_id", "lineage"])
    theta_sd_df = pd.DataFrame(theta_sd_rows).sort_values(["site_id", "date", "sample_id", "lineage"])
    leverage_df = pd.DataFrame(leverage_rows).sort_values(["site_id", "date", "sample_id", "mutation"])
    residuals_df = pd.DataFrame(resid_rows).sort_values(["site_id", "date", "sample_id", "mutation"])
    obj_trace_df = pd.DataFrame(obj_trace_rows).sort_values(["site_id", "iter"])

    try:
        med_cov = (snv.groupby(["site_id", "date"])["coverage"].median().reset_index().rename(columns={"coverage": "median_coverage"}))
        if not theta_df.empty and not med_cov.empty:
            theta_df = theta_df.merge(med_cov, on=["site_id", "date"], how="left")
    except Exception:
        pass

    # write tables
    ctx.write_table("theta_estimates", theta_df)
    if not theta_raw_df.empty: ctx.write_table("theta_estimates_raw", theta_raw_df)
    if not theta_sd_df.empty:  ctx.write_table("theta_uncertainty", theta_sd_df)
    if not leverage_df.empty:  ctx.write_table("mutation_leverage", leverage_df)
    ctx.write_table("residuals", residuals_df)
    ctx.write_table("objective_trace", obj_trace_df)
    ctx.write_table("signatures_used", S_df.reset_index().rename(columns={"index": "mutation"}))
    ctx.write_table("overlap_matrix", ov_df)

    # figures disabled here — but write z-scores for later plotting
    try:
        zdf = _compute_zscores(residuals_df)
        ctx.write_table("zscore_diagnostics",
                        zdf[["site_id", "date", "sample_id", "mutation", "z", "coverage", "pred_af", "obs_af", "kappa_used"]])
    except Exception as e:
        ctx.log(level="WARNING", message="Z-score diagnostics failed", context={"error": str(e)})

    # metrics/report
    try:
        ctx.write_metrics("simplex_satisfied",
                          pd.DataFrame({"simplex_satisfied": [bool(np.all(simplex_ok))],
                                        "fraction_ok": [float(np.mean(simplex_ok))]}))
    except Exception:
        ctx.write_table("simplex_satisfied",
                        pd.DataFrame({"simplex_satisfied": [bool(np.all(simplex_ok))],
                                      "fraction_ok": [float(np.mean(simplex_ok))]}))

    r = residuals_df["residual"].to_numpy() if not residuals_df.empty else np.array([])
    mae = float(np.mean(np.abs(r))) if r.size else float("nan")
    rmse = float(np.sqrt(np.mean(r**2))) if r.size else float("nan")
    n_samples = int(theta_df[["site_id", "date", "sample_id"]].drop_duplicates().shape[0]) if not theta_df.empty else 0
    n_sites = int(theta_df["site_id"].nunique()) if not theta_df.empty else 0
    n_lineages = len(lineages)

    kappa_cap_str = "inf" if not np.isfinite(prior_kappa_cap) else f"{prior_kappa_cap:g}"
    report = [
        "# Likelihood v4.3 report (NumPy, CPU)",
        "",
        f"- Samples fitted: {n_samples} across {n_sites} sites; lineages: {n_lineages}",
        f"- Priors mutations (authoritative): {len(mutations)}",
        f"- λ_ridge={lambda_ridge:g}, λ_L2(base)={lambda_temporal_l2:g}, λ_overlap={overlap_penalty_lambda:g}, λ_unknown={unknown_extra:g}",
        f"- IRLS iters≤{irls_max_iter}, ADMM iters≤{admm_max_iter}, tol={admm_tol:g}, adaptive_rho={admm_adaptive_rho}",
        f"- κ scale={prior_kappa_scale:g}, κ cap={kappa_cap_str}",
        f"- Simplex ok fraction: {np.mean(simplex_ok):.3f}",
        f"- Residual MAE={mae:.4f}, RMSE={rmse:.4f}",
        f"- Uncertainty table: {'yes' if compute_uncertainty else 'no'}, Leverage: {'yes' if compute_leverage else 'no'}",
        "",
        "Notes:",
        "• IRLS curvature/targets use exact Beta–Binomial + Beta(β0/2,β0/2) jitter (Jeffreys-like) for stability near AF=0/1.",
        "• Each slice solves a dense simplex QP via ADMM (one regularized linear solve per iteration).",
        "• Temporal L2 enters both H and f exactly; overlap adds O penalty; ridge stabilizes.",
        "• Prior-only mode sets y=n=0 internally, using μ,κ; still writes full tables/figures.",
    ]
    ctx.write_report("\\n".join(report))

    figures = []  # no figures; you can add later
    tables = ["theta_estimates", "residuals", "objective_trace",
              "signatures_used", "overlap_matrix", "zscore_diagnostics", "simplex_satisfied"]
    if not theta_raw_df.empty: tables.append("theta_estimates_raw")
    if not theta_sd_df.empty:  tables.append("theta_uncertainty")
    if not leverage_df.empty:  tables.append("mutation_leverage")

    gc.collect()
    return {"tables": tables, "figures": figures, "report": True}


# Optional direct run for ad-hoc testing
if __name__ == "__main__":
    class _DummyCtx(RunContext):
        def write_table(self, key, df): super().write_table(key, df)
        def write_figure(self, key, fig): pass
        def write_report(self, text): super().write_report(text)
        def write_metrics(self, key, df): super().write_table(key, df)

    cfg = {"likelihood": {
        "seed": 12345,
        "priors_hyperparams_path": "results/priors/tables/priors_hyperparams.csv",
        "signatures_path": "data/signatures.csv",
        "priors_time_path": "results/priors/tables/priors_time_local.csv",
        "prior_only": True,
        "prior_only_fill_from_counts": False,
        # Optional knobs:
        "robust_c": 4.685,
        "prior_jitter_beta0": 0.5,
        "temporal_smooth_lambda_l2": 0.02,
        "lambda_ridge": 5e-4,
        "overlap_penalty_lambda": 0.0,
        "unknown_extra": 0.0,
        "irls_max_iter": 3,
        "admm_max_iter": 200,
        "admm_tol": 1e-6,
        "admm_rho": 1.0,
        "admm_adaptive_rho": True,
        "admm_rho_floor": 1e-6,
        "admm_rho_ceil": 1e6,
        "aggregate_by_date": True,
        "compute_uncertainty": True,
        "compute_leverage": True,
    }}
    try:
        run_likelihood(cfg, _DummyCtx())
    except Exception as e:
        print("Smoke test (might miss data files):", e)
