# -*- coding: utf-8 -*-
"""
Likelihood — Robust Beta–Binomial IRLS + ADMM on Δ (NumPy, strict paths, lean)

Inputs (strict paths; set in YAML under likelihood:):
  - priors_hyperparams_path: results/priors/priors_hyperparams.csv
  - priors_time_path:        results/priors/detail_global_timeseries.csv   (optional; omit or empty to disable)
  - signatures_path:         data/signatures.csv
  - snv_counts_path:         results/preprocessing/tables/feature_store_snv.csv (optional if prior_only=True)

Outputs (to results/likelihood/tables/):
  theta_estimates, residuals, objective_trace, signatures_used,
  overlap_matrix, zscore_diagnostics, simplex_satisfied,
  (optional) theta_estimates_raw, theta_uncertainty, mutation_leverage
"""

from __future__ import annotations
import os, gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.errors import ParserError

EPS = 1e-9
REQUIRED_SNV = ["site_id","date","sample_id","mutation","count","coverage"]
GLOBAL = "__GLOBAL__"

# -------------------- Minimal context fallback --------------------
try:
    from utils.run import RunContext  # type: ignore
except Exception:  # pragma: no cover
    class RunContext:
        def log(self, **kw): print("[LOG]", kw)
        def _ensure(self, d): os.makedirs(d, exist_ok=True); return d
        def write_table(self, key, df):
            p = os.path.join(self._ensure("results/likelihood/tables"), f"{key}.csv")
            df.to_csv(p, index=False); print(f"[TABLE] {key} -> {p}")
        def write_report(self, text):
            p = os.path.join(self._ensure("results/likelihood"), "report.md")
            with open(p, "a", encoding="utf-8") as f: f.write(text + "\n")
        def write_metrics(self, key, df): self.write_table(key, df)
        def write_figure(self, key, fig): pass

# -------------------- Config --------------------
@dataclass
class LikelihoodCfg:
    seed: int = 12345
    snv_counts_path: Optional[str] = None
    priors_hyperparams_path: str = "results/priors/priors_hyperparams.csv"
    priors_time_path: str = "results/priors/detail_global_timeseries.csv"   # optional
    signatures_path: str = "data/signatures.csv"

    prior_only: bool = False
    prior_only_fill_from_counts: bool = True

    lambda_ridge: float = 5e-4
    overlap_penalty_lambda: float = 0.0
    temporal_smooth_lambda_l2: float = 0.02
    unknown_extra: float = 0.0

    robust_c: float = 4.685
    prior_jitter_beta0: float = 0.5
    prob_clip_eps: float = 1e-8

    irls_max_iter: int = 10
    admm_max_iter: int = 500
    admm_tol: float = 5e-7
    admm_rho: float = 1.0
    admm_adaptive_rho: bool = True
    admm_rho_floor: float = 1e-6
    admm_rho_ceil: float = 1e6

    aggregate_by_date: bool = True
    compute_uncertainty: bool = True
    compute_leverage: bool = True

    min_coverage_for_wls: float = 0.0
    cap_coverage_quantile: Optional[float] = 0.999
    prior_kappa_scale: float = 1.0
    prior_kappa_cap: float = float("inf")
    temporal_cov_scale: float = 1e4

def _cfg_from(lk: Dict[str, Any]) -> LikelihoodCfg:
    base = LikelihoodCfg()
    for k, v in (lk.get("likelihood", lk) or {}).items():
        if hasattr(base, k): setattr(base, k, v)
    return base

# -------------------- Helpers --------------------
def _read_csv(p: str, **kw) -> pd.DataFrame:
    try: return pd.read_csv(p, low_memory=False, **kw)
    except (ParserError, UnicodeDecodeError, OSError, MemoryError):
        return pd.read_csv(p, low_memory=False, engine="python", **kw)

def _norm_ts(obj) -> Optional[pd.Timestamp]:
    """Normalize any 'date-like' to midnight Timestamp; unwrap 1-element list/tuple repeatedly."""
    d = obj
    while isinstance(d, (list, tuple)) and len(d) == 1:
        d = d[0]
    try:
        ts = pd.to_datetime(d, errors="coerce")
    except Exception:
        return None
    if getattr(ts, "ndim", 0) and len(ts) > 0:
        ts = ts[0]
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()

def _normalize_snv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"site":"site_id","sample":"sample_id"}).copy()
    miss = [c for c in REQUIRED_SNV if c not in df.columns]
    if miss: raise KeyError(f"SNV table missing: {miss}")
    df["site_id"] = df["site_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df["mutation"] = df["mutation"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any(): raise ValueError("Invalid dates in SNV table.")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0).astype(int)
    return df[df["coverage"] >= 0].sort_values(["site_id","date","mutation"]).reset_index(drop=True)

def _load_counts_strict(p: Optional[str]) -> pd.DataFrame:
    if not p:
        raise FileNotFoundError("likelihood.snv_counts_path is required unless prior_only=True")
    if not os.path.exists(p):
        raise FileNotFoundError(f"snv_counts_path not found: {p}")
    return _normalize_snv(_read_csv(p) if p.lower().endswith(".csv") else pd.read_parquet(p))

# -------------------- Priors & signatures (STRICT paths) --------------------
def _load_priors_strict(p: str, ctx: RunContext) -> pd.DataFrame:
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Priors missing: {p}")
    ctx.log(level="INFO", message="Using priors_hyperparams", context={"path": p})
    df = _read_csv(p)
    if "mutation" not in df.columns:
        raise ValueError("priors_hyperparams.csv needs 'mutation'")
    df = df.assign(mutation=lambda d: d["mutation"].astype(str)).copy()
    for col in ["mu_shrunk","mu","kappa_shrunk","kappa","alpha","beta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _select_best_prior_row(g: pd.DataFrame) -> pd.Series:
    """Choose 1 row per mutation: max kappa_shrunk → max kappa → max (alpha+beta) → first."""
    if "kappa_shrunk" in g.columns and g["kappa_shrunk"].notna().any():
        return g.loc[g["kappa_shrunk"].idxmax()]
    if "kappa" in g.columns and g["kappa"].notna().any():
        return g.loc[g["kappa"].idxmax()]
    if ("alpha" in g.columns) and ("beta" in g.columns):
        s = (g["alpha"] + g["beta"]).astype(float)
        if s.notna().any():
            return g.loc[s.idxmax()]
    return g.iloc[0]

def _mu_kappa(pri: pd.DataFrame, muts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (mu, kappa) aligned to 'muts', resolving duplicate rows per mutation robustly."""
    if "mutation" not in pri.columns:
        raise ValueError("priors_hyperparams needs 'mutation'")
    best_rows: Dict[str, pd.Series] = {}
    for mut, g in pri.groupby("mutation", sort=False, group_keys=False):
        best_rows[mut] = _select_best_prior_row(g)

    mu_arr = np.zeros(len(muts), dtype=float)
    k_arr  = np.zeros(len(muts), dtype=float)
    for i, m in enumerate(muts):
        row = best_rows.get(m)
        if row is None:
            mu_val, k_val = 0.5, 0.0
        else:
            # mu
            if "mu_shrunk" in row.index and pd.notna(row["mu_shrunk"]):
                mu_val = float(row["mu_shrunk"])
            elif "mu" in row.index and pd.notna(row["mu"]):
                mu_val = float(row["mu"])
            elif ("alpha" in row.index) and ("beta" in row.index) and pd.notna(row["alpha"]) and pd.notna(row["beta"]):
                tot = float(row["alpha"]) + float(row["beta"])
                mu_val = float(row["alpha"])/tot if tot > 0 else 0.5
            else:
                mu_val = 0.5
            # kappa
            if "kappa_shrunk" in row.index and pd.notna(row["kappa_shrunk"]):
                k_val = float(row["kappa_shrunk"])
            elif "kappa" in row.index and pd.notna(row["kappa"]):
                k_val = float(row["kappa"])
            elif ("alpha" in row.index) and ("beta" in row.index) and pd.notna(row["alpha"]) and pd.notna(row["beta"]):
                k_val = float(row["alpha"]) + float(row["beta"]) - 2.0
            else:
                k_val = 0.0
        mu_arr[i] = float(np.clip(mu_val, EPS, 1.0 - EPS))
        k_arr[i]  = float(max(k_val, 0.0))
    return mu_arr, k_arr

def _load_signatures_strict(p: str, ctx: RunContext) -> pd.DataFrame:
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"signatures.csv missing: {p}")
    ctx.log(level="INFO", message="Using signatures", context={"path": p})
    s = _read_csv(p)
    need = {"mutation","lineage","weight"} - set(s.columns)
    if need: raise ValueError(f"signatures.csv missing cols: {need}")
    s["mutation"] = s["mutation"].astype(str)
    s["lineage"]  = s["lineage"].astype(str)
    s["weight"]   = pd.to_numeric(s["weight"], errors="coerce").fillna(0.0).clip(0.0,1.0)
    return s.groupby(["mutation","lineage"], as_index=False)["weight"].max()

def _build_S(sig: pd.DataFrame, muts: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    Sdf = sig.pivot_table(index="mutation", columns="lineage", values="weight", fill_value=0.0)
    Sdf = Sdf.reindex(index=pd.Index(muts), fill_value=0.0)
    if "GLOBAL" in Sdf.columns: Sdf = Sdf.drop(columns=["GLOBAL"])
    rs = Sdf.sum(1).astype(float)
    if (rs>1+1e-12).any(): Sdf.loc[rs>1+1e-12] = Sdf.loc[rs>1+1e-12].div(rs[rs>1+1e-12], axis=0)
    Sdf["GLOBAL"] = np.clip(1.0 - Sdf.sum(1).values, 0.0, 1.0)
    Sdf = Sdf.sort_index().sort_index(axis=1)
    return Sdf, Sdf.values.astype(float, copy=False), list(Sdf.index), list(Sdf.columns)

def _overlap(S: np.ndarray, names: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    col_norm = np.linalg.norm(S, axis=0) + EPS
    O = (S/col_norm).T @ (S/col_norm); np.fill_diagonal(O, 1.0)
    return O, pd.DataFrame(np.clip(O,0.0,1.0), index=names, columns=names)

# -------------------- Time-local priors map (STRICT path; optional) --------------------
def _time_local_map_strict(p: Optional[str], ctx: RunContext) -> Tuple[Optional[Dict[Any, pd.DataFrame]], bool]:
    """
    Returns:
      - tmap: Dict with keys:
          has_site=True  -> (site_id:str, Timestamp)
          has_site=False -> Timestamp
      - has_site: whether site_id was present
    Keys are canonicalized; 'date' normalized to midnight to match SNV dates.
    """
    if not p:
        ctx.log(level="INFO", message="No time-local priors path provided.")
        return None, False
    if not os.path.exists(p):
        ctx.log(level="INFO", message="time-local priors not found (optional)", context={"path": p})
        return None, False

    ctx.log(level="INFO", message="Using time-local priors", context={"path": p})
    t = _read_csv(p)
    if "date" not in t.columns or "mutation" not in t.columns:
        return None, False

    t = t.copy()
    t["mutation"] = t["mutation"].astype(str)
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    if "mu_t" not in t.columns and "mu" in t.columns:       t["mu_t"]    = t["mu"]
    if "kappa_t" not in t.columns and "kappa" in t.columns: t["kappa_t"] = t["kappa"]

    cols = ["mutation","mu_t","kappa_t","date"]
    has_site = "site_id" in t.columns
    if has_site:
        t["site_id"] = t["site_id"].astype(str)
        cols.append("site_id")

    t = t[cols].dropna(subset=["date"])
    tmap: Dict[Any, pd.DataFrame] = {}

    if has_site:
        for (site_id, dt), sub in t.groupby(["site_id","date"], sort=False):
            key = (str(site_id), pd.Timestamp(dt))
            tmap[key] = sub[["mutation","mu_t","kappa_t"]].reset_index(drop=True)
    else:
        for dt, sub in t.groupby(["date"], sort=False):
            key = pd.Timestamp(dt)
            tmap[key] = sub[["mutation","mu_t","kappa_t"]].reset_index(drop=True)

    return tmap, has_site

# -------------------- Math kernels --------------------
def _project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    u = np.sort(v)[::-1]; cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1,len(v)+1)) > (cssv - 1))[0]
    theta = 0.0 if rho.size==0 else (cssv[rho[-1]] - 1.0) / (rho[-1] + 1.0)
    w = np.maximum(v - theta, 0.0); s = w.sum()
    return w/s if s>0 else np.ones_like(w)/len(w)

def _admm_qp_simplex(H: np.ndarray, f: np.ndarray, rho: float, tol: float, iters: int) -> Tuple[np.ndarray,int]:
    K = H.shape[0]; A = H + rho * np.eye(K)
    try:
        L = np.linalg.cholesky(A + 1e-9*np.eye(K))
        solve = lambda q: np.linalg.solve(L.T, np.linalg.solve(L, q))
    except np.linalg.LinAlgError:
        solve = lambda q: np.linalg.solve(A + 1e-9*np.eye(K), q)
    z = np.ones(K)/K; u = np.zeros(K); x = z
    for it in range(int(iters)):
        x = solve(f + rho*(z - u))
        z_new = _project_simplex(x + u)
        u += x - z_new
        if max(np.linalg.norm(x - z_new), rho*np.linalg.norm(z_new - z)) <= tol: return z_new, it+1
        z = z_new
    return z, iters

def _wls_bb(S: np.ndarray, y: np.ndarray, n: np.ndarray, mu: np.ndarray, kappa: np.ndarray,
            p_hat: np.ndarray, clip: float, c_rob: float, beta0: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    p = np.clip(p_hat, clip, 1-clip)
    a = np.maximum(y + kappa*mu + 0.5*beta0, 0.0)
    b = np.maximum((n - y) + kappa*(1-mu) + 0.5*beta0, 0.0)
    invp, invq = 1.0/p, 1.0/(1.0-p)
    g = -a*invp + b*invq
    W = np.clip(a*invp**2 + b*invq**2, 1e-12, 1e12)
    z = np.clip(p - g/np.maximum(W, 1e-12), clip, 1-clip)
    if c_rob and c_rob>0:
        r = z - p; s = np.median(np.abs(r - np.median(r))) + 1e-12
        W *= (1 - np.clip((r/(c_rob*s+1e-12))**2, 0.0, 1.0))**2
    SW = S * W[:,None]
    return S.T @ SW, S.T @ (W*z), W, z

def _zscore_df(res: pd.DataFrame) -> pd.DataFrame:
    n = res["coverage"].to_numpy(float)
    p = np.clip(res["pred_af"].to_numpy(float), EPS, 1-EPS)
    k = np.maximum(res["kappa_used"].to_numpy(float), 0.0)
    var_af = p*(1-p)*(np.maximum(n,1.0)+k)/(np.maximum(n,1.0)*(k+1.0))
    sd = np.sqrt(np.maximum(var_af, EPS))
    out = res.copy()
    out["z"] = (out["obs_af"].to_numpy(float) - p)/sd
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

# -------------------- Site solver (IRLS → ADMM) --------------------
def _solve_site(S: np.ndarray, yL: List[np.ndarray], nL: List[np.ndarray],
                mu_vec: np.ndarray, kap_vec: np.ndarray, lam_ridge: float,
                lam_l2_vec: np.ndarray, O_pen: Optional[np.ndarray], lam_ov: float,
                cfg: LikelihoodCfg, theta0: List[np.ndarray], unknown_idx: Optional[int]) \
                -> Tuple[List[np.ndarray], List[float], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    M, K, T = S.shape[0], S.shape[1], len(yL)
    theta = [t/np.sum(t) if np.sum(t)>0 else np.ones(K)/K for t in [np.maximum(t0,0.0) for t0 in theta0]]
    obj, last_H, last_W = [], [np.eye(K)]*T, [np.ones(M)]*T

    for _ in range(max(1, cfg.irls_max_iter)):
        total, Hs, fs, Ws = 0.0, [], [], []
        for t in range(T):
            th = theta[t]; p_hat = S @ th
            H, f, W, _ = _wls_bb(S, yL[t], nL[t], mu_vec, kap_vec, p_hat,
                                 cfg.prob_clip_eps, cfg.robust_c, cfg.prior_jitter_beta0)
            lam_left  = float(lam_l2_vec[t-1]) if t>0 else 0.0
            lam_right = float(lam_l2_vec[t])   if (t+1)<T else 0.0
            H = H + (lam_ridge + lam_left + lam_right) * np.eye(K)
            if O_pen is not None and lam_ov>0: H = H + lam_ov * O_pen
            if unknown_idx is not None and cfg.unknown_extra>0: H[unknown_idx, unknown_idx] += cfg.unknown_extra
            if t>0:      f = f + lam_left  * theta[t-1]
            if t+1 < T:  f = f + lam_right * theta[t+1]
            total += 0.5*float(th @ H @ th) - float(f @ th)
            Hs.append(H); fs.append(f); Ws.append(W)
        obj.append(total)

        max_change = 0.0
        for t in range(T):
            H, f = Hs[t], fs[t]
            if cfg.admm_adaptive_rho:
                try:
                    lam_max = float(np.linalg.eigvalsh(0.5*(H+H.T)).max())
                    tr = float(np.trace(H))/max(K,1)
                    rho_use = float(np.clip((lam_max/max(tr,1e-12))**0.5, cfg.admm_rho_floor, cfg.admm_rho_ceil))
                except Exception:
                    rho_use = cfg.admm_rho
            else:
                rho_use = cfg.admm_rho
            th_old = theta[t]
            th_new, _ = _admm_qp_simplex(H, f, rho=rho_use, tol=cfg.admm_tol, iters=cfg.admm_max_iter)
            if np.linalg.norm(th_new - th_old) > 0.5: th_new = 0.5*(th_new + th_old)
            theta[t] = th_new
            max_change = max(max_change, float(np.linalg.norm(th_new - th_old)))
        last_H, last_W = Hs, Ws
        if max_change < 5e-6: break

    diag_cov_list = lev_list = None
    if cfg.compute_uncertainty or cfg.compute_leverage:
        diag_cov_list, lev_list = [], []
        for t in range(T):
            H = last_H[t]; W = last_W[t]
            try: Hinv = np.linalg.inv(H + 1e-9*np.eye(K))
            except np.linalg.LinAlgError: Hinv = np.linalg.pinv(H)
            diag_cov_list.append(np.clip(np.diag(Hinv), 0.0, np.inf))
            if cfg.compute_leverage:
                R = (np.sqrt(np.clip(W,0.0,np.inf))[:,None] * S)
                VH = R @ Hinv
                lev = np.sum(VH * R, axis=1)
                lev_list.append(np.clip(lev, 0.0, 1.0))
            else:
                lev_list.append(np.zeros(M))
    return theta, obj, diag_cov_list, lev_list

# -------------------- Entry point --------------------
def run_likelihood(cfg_in: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:

    cfg = _cfg_from(cfg_in)
    np.random.default_rng(int(cfg.seed))
    ctx.log(level="INFO", message="Likelihood start (strict paths)", context={
        "prior_only": cfg.prior_only,
        "priors_hyperparams_path": cfg.priors_hyperparams_path,
        "priors_time_path": cfg.priors_time_path,
        "signatures_path": cfg.signatures_path,
        "snv_counts_path": cfg.snv_counts_path,
        "cwd": os.getcwd(),
    })

    # --- priors & signatures
    pri = _load_priors_strict(cfg.priors_hyperparams_path, ctx)
    muts = pd.Index(pri["mutation"].astype(str).unique()).sort_values().tolist()
    if not muts: raise RuntimeError("Priors have no mutations.")

    sig = _load_signatures_strict(cfg.signatures_path, ctx)
    Sdf, S, mutations, lineages = _build_S(sig, muts)
    if set(mutations) != set(muts):
        missing = sorted(set(muts)-set(mutations)); raise RuntimeError(f"Signatures missing mutations (first 10): {missing[:10]}")
    ctx.write_table("signatures_used", Sdf.reset_index().rename(columns={"index":"mutation"}))
    O, Odf = _overlap(S, lineages); ctx.write_table("overlap_matrix", Odf)
    O_pen = O if cfg.overlap_penalty_lambda>0 else None
    unknown_idx = lineages.index("GLOBAL") if "GLOBAL" in lineages else None

    mu_vec, kap_vec = _mu_kappa(pri, mutations)
    kap_vec = np.minimum(kap_vec * cfg.prior_kappa_scale, cfg.prior_kappa_cap) if np.isfinite(cfg.prior_kappa_cap) else kap_vec*cfg.prior_kappa_scale
    kap_vec = np.maximum(kap_vec, 0.0)

    tmap, has_site = _time_local_map_strict(cfg.priors_time_path, ctx)

    # --- counts
    if cfg.prior_only:
        if cfg.snv_counts_path and os.path.exists(cfg.snv_counts_path):
            snv = _normalize_snv(_read_csv(cfg.snv_counts_path))
        else:
            snv = pd.DataFrame({"site_id":["PRIOR_ONLY"],"date":[pd.Timestamp("1970-01-01")],
                                "sample_id":["PRIOR_ONLY__date"],"mutation":[mutations[0]],"count":[0],"coverage":[0]})
    else:
        snv = _load_counts_strict(cfg.snv_counts_path)

    snv = snv[snv["mutation"].astype(str).isin(mutations)].copy()
    if snv.empty and not cfg.prior_only:
        raise RuntimeError("Counts empty after restricting to priors mutations.")
    if cfg.aggregate_by_date and not snv.empty:
        snv_use = (snv.groupby(["site_id","date","mutation"], as_index=False)
                      .agg(count=("count","sum"), coverage=("coverage","sum")))
        snv_use["sample_id"] = snv_use["site_id"] + "__date"
    else:
        snv_use = snv.copy()
    snv_use = snv_use.sort_values(["site_id","date","mutation"]).reset_index(drop=True)
    ctx.write_table("residuals_input_head", snv_use.head(10))

    # --- sites to fit
    sites = list(snv_use["site_id"].astype(str).unique())
    if cfg.prior_only and tmap is not None and has_site:
        sites = sorted(set(sites) | {k[0] for k in tmap.keys() if isinstance(k, tuple)})

    # --- collect outputs
    theta_rows: List[Dict[str, Any]] = []
    theta_raw_rows: List[Dict[str, Any]] = []
    theta_sd_rows: List[Dict[str, Any]] = []
    lev_rows: List[Dict[str, Any]] = []
    resid_rows: List[Dict[str, Any]] = []
    obj_rows: List[Dict[str, Any]] = []
    simplex_ok: List[bool] = []

    # Build canonical snapshot list: (date_key: Timestamp, df_s: DataFrame)
    def _snapshots_for_site(df_site: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
        out: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
        for key, df_s in df_site.groupby(["date"], sort=False):
            dkey = _norm_ts(key)
            if dkey is None:  # skip invalid keys
                continue
            out.append((dkey, df_s))
        return out

    def _dates_for_site_prior_only(site: str) -> List[pd.Timestamp]:
        if tmap is not None:
            if has_site:
                ds = [k[1] for k in tmap.keys() if isinstance(k, tuple) and k[0]==str(site)]
            else:
                ds = [k for k in tmap.keys() if not isinstance(k, tuple)]
        else:
            ds = []
        if (not ds) and cfg.prior_only_fill_from_counts:
            ds = list(snv_use.loc[snv_use["site_id"]==site,"date"].unique())
        if not ds:
            ds = [pd.Timestamp("1970-01-01")]
        return sorted([_norm_ts(d) or pd.Timestamp("1970-01-01") for d in ds])

    # Lookup into tmap with robust date normalization
    def _time_local_for(site, date_key: pd.Timestamp):
        if tmap is None: return None
        return tmap.get((str(site), date_key)) if has_site else tmap.get(date_key)

    for site in sites:
        df_site = snv_use[snv_use["site_id"]==site].sort_values(["date","mutation"])
        snaps = ([(d, pd.DataFrame()) for d in _dates_for_site_prior_only(site)]
                 if cfg.prior_only else _snapshots_for_site(df_site))
        T = len(snaps)
        if T == 0: continue

        yL: List[np.ndarray] = []
        nL: List[np.ndarray] = []
        lamL: List[float] = [0.0]*T
        theta0: List[np.ndarray] = []
        prev: Optional[np.ndarray] = None
        payloads: List[Tuple[pd.Timestamp, pd.DataFrame]] = []

        for t, (date_key, df_s) in enumerate(snaps):
            # start from global priors
            mu_use, kap_use = mu_vec.copy(), kap_vec.copy()
            # possibly override with time-local priors
            pt = _time_local_for(site, date_key)
            if pt is not None and not pt.empty:
                m2m = dict(zip(pt["mutation"], pd.to_numeric(pt["mu_t"], errors="coerce")))
                m2k = dict(zip(pt["mutation"], pd.to_numeric(pt["kappa_t"], errors="coerce")))
                for i, m in enumerate(mutations):
                    mui, ki = m2m.get(m, np.nan), m2k.get(m, np.nan)
                    if np.isfinite(mui): mu_use[i]  = float(np.clip(mui, EPS, 1-EPS))
                    if np.isfinite(ki):  kap_use[i] = float(max(ki, 0.0))

            if cfg.prior_only:
                y, n = kap_use * mu_use, kap_use
                af, w = mu_use, kap_use
                tot_cov = float(np.sum(kap_use))
                cov_for_resid, obs_af_resid, kap_for_resid = n.copy(), af.copy(), kap_use.copy()
            else:
                dct_y = df_s.set_index("mutation")["count"].to_dict()
                dct_n = df_s.set_index("mutation")["coverage"].to_dict()
                y = np.array([dct_y.get(m,0) for m in mutations], float)
                n = np.array([dct_n.get(m,0) for m in mutations], float)
                if cfg.cap_coverage_quantile is not None and n.size:
                    cap = float(np.quantile(n, float(cfg.cap_coverage_quantile))); n = np.minimum(n, cap)
                af = np.divide(y, np.maximum(n,1.0), out=np.zeros_like(y), where=n>0)
                w  = n + kap_use
                tot_cov = float(np.maximum(n.sum(),0.0))
                cov_for_resid, obs_af_resid, kap_for_resid = n.copy(), af.copy(), kap_use.copy()

            yL.append(y); nL.append(n)
            lamL[t] = cfg.temporal_smooth_lambda_l2 / np.sqrt(1.0 + tot_cov / cfg.temporal_cov_scale)

            # warm start
            K = S.shape[1]
            if np.sum(w)>0:
                A = S.T @ (w[:,None]*S) + 1e-9*np.eye(K); b = S.T @ (w*af)
                try: th0 = np.linalg.solve(A, b)
                except np.linalg.LinAlgError: th0 = np.linalg.lstsq(A, b, rcond=None)[0]
                if prev is not None: th0 = 0.5*th0 + 0.5*prev
                th0 = np.maximum(th0, 0.0); s = th0.sum(); th0 = (th0/s) if s>0 else np.ones(K)/K
            else:
                th0 = np.ones(K)/K if prev is None else prev
            theta0.append(th0); prev = th0

            # store residual payload with a normalized timestamp
            payloads.append((date_key, pd.DataFrame({
                "mutation":mutations, "obs_af":obs_af_resid, "coverage":cov_for_resid,
                "kappa_used":kap_for_resid, "date":[date_key]*len(mutations)
            })))

        mu_for_solver  = np.full(S.shape[0], 0.5) if cfg.prior_only else mu_vec
        kap_for_solver = np.zeros(S.shape[0])     if cfg.prior_only else kap_vec

        thetas, trace, diag_covs, levs = _solve_site(
            S, yL, nL, mu_for_solver, kap_for_solver, cfg.lambda_ridge,
            np.array(lamL, float), O_pen, cfg.overlap_penalty_lambda,
            cfg, theta0, unknown_idx
        )
        for it, val in enumerate(trace):
            obj_rows.append({"site_id": site, "iter": it, "objective": float(val)})

        for t, (date_key, payload) in enumerate(payloads):
            th = np.maximum(thetas[t], 0.0); th = th / max(th.sum(), 1.0)
            p  = np.clip(S @ th, EPS, 1-EPS)

            dct_obs = payload.set_index("mutation")["obs_af"].to_dict()
            dct_cov = payload.set_index("mutation")["coverage"].to_dict()
            dct_kap = payload.set_index("mutation")["kappa_used"].to_dict()
            obs_af = np.array([dct_obs.get(m,0) for m in mutations], float)
            covv   = np.array([dct_cov.get(m,0) for m in mutations], float)
            kappav = np.array([dct_kap.get(m,0) for m in mutations], float)

            for i, m in enumerate(mutations):
                resid_rows.append({
                    "site_id":site,"date":date_key,"sample_id":f"{site}__date","mutation":m,
                    "obs_af":float(obs_af[i]),"pred_af":float(p[i]),"residual":float(obs_af[i]-p[i]),
                    "coverage":float(covv[i]),"kappa_used":float(kappav[i]),
                    "weight": float(0.0 if covv[i]<=0 else 1.0)
                })

            for lin, v in zip(lineages, th):
                theta_rows.append({"site_id":site,"date":date_key,"sample_id":f"{site}__date","lineage":lin,"theta":float(v)})
                theta_raw_rows.append({"site_id":site,"date":date_key,"sample_id":f"{site}__date","lineage":lin,"theta_raw":float(v)})

            if cfg.compute_uncertainty and diag_covs is not None:
                sd = np.sqrt(np.maximum(diag_covs[t], 0.0))
                for lin, s in zip(lineages, sd):
                    theta_sd_rows.append({"site_id":site,"date":date_key,"sample_id":f"{site}__date","lineage":lin,"theta_sd":float(s)})

            if cfg.compute_leverage and levs is not None:
                lev = levs[t]
                for mut, lv in zip(mutations, lev):
                    lev_rows.append({"site_id":site,"date":date_key,"sample_id":f"{site}__date","mutation":mut,"leverage":float(lv)})

            simplex_ok.append((abs(np.sum(th)-1.0)<=1e-8) and np.all(th>=-1e-8))

    # assemble + write
    theta_df = pd.DataFrame(theta_rows).sort_values(["site_id","date","sample_id","lineage"])
    theta_raw_df = pd.DataFrame(theta_raw_rows).sort_values(["site_id","date","sample_id","lineage"])
    theta_sd_df = pd.DataFrame(theta_sd_rows).sort_values(["site_id","date","sample_id","lineage"])
    leverage_df = pd.DataFrame(lev_rows).sort_values(["site_id","date","sample_id","mutation"])
    residuals_df = pd.DataFrame(resid_rows).sort_values(["site_id","date","sample_id","mutation"])
    obj_df = pd.DataFrame(obj_rows).sort_values(["site_id","iter"])

    try:
        med_cov = (snv_use.groupby(["site_id","date"])["coverage"].median().reset_index().rename(columns={"coverage":"median_coverage"}))
        if not theta_df.empty and not med_cov.empty:
            theta_df = theta_df.merge(med_cov, on=["site_id","date"], how="left")
    except Exception:
        pass

    ctx.write_table("theta_estimates", theta_df)
    if not theta_raw_df.empty: ctx.write_table("theta_estimates_raw", theta_raw_df)
    if not theta_sd_df.empty:  ctx.write_table("theta_uncertainty", theta_sd_df)
    if not leverage_df.empty:  ctx.write_table("mutation_leverage", leverage_df)
    ctx.write_table("residuals", residuals_df)
    ctx.write_table("objective_trace", obj_df)

    try:
        zdf = _zscore_df(residuals_df)
        ctx.write_table("zscore_diagnostics", zdf[["site_id","date","sample_id","mutation","z","coverage","pred_af","obs_af","kappa_used"]])
    except Exception as e:
        ctx.log(level="WARNING", message="Z-score diagnostics failed", context={"error": str(e)})

    met = pd.DataFrame({"simplex_satisfied":[bool(np.all(simplex_ok))],"fraction_ok":[float(np.mean(simplex_ok))]})
    try: ctx.write_metrics("simplex_satisfied", met)
    except Exception: ctx.write_table("simplex_satisfied", met)

    r = residuals_df["residual"].to_numpy() if not residuals_df.empty else np.array([])
    report = [
        "# Likelihood (NumPy, strict paths)",
        f"- samples: {int(theta_df[['site_id','date','sample_id']].drop_duplicates().shape[0]) if not theta_df.empty else 0}",
        f"- sites: {theta_df['site_id'].nunique() if not theta_df.empty else 0}, lineages: {len(lineages)}",
        f"- prior_only={cfg.prior_only}, ridge={cfg.lambda_ridge:g}, L2={cfg.temporal_smooth_lambda_l2:g}, overlap={cfg.overlap_penalty_lambda:g}",
        f"- MAE={float(np.mean(np.abs(r))) if r.size else float('nan'):.4f}, RMSE={float(np.sqrt(np.mean(r**2))) if r.size else float('nan'):.4f}",
    ]
    ctx.write_report("\n".join(report))

    tables = ["theta_estimates","residuals","objective_trace","signatures_used","overlap_matrix","zscore_diagnostics","simplex_satisfied"]
    if not theta_raw_df.empty: tables.append("theta_estimates_raw")
    if not theta_sd_df.empty:  tables.append("theta_uncertainty")
    if not leverage_df.empty:  tables.append("mutation_leverage")
    gc.collect()
    return {"tables": tables, "figures": [], "report": True}


# optional smoke test
if __name__ == "__main__":
    class _DummyCtx(RunContext): ...
    cfg = {"likelihood": {
        "seed": 12345,
        "priors_hyperparams_path": "results/priors/priors_hyperparams.csv",
        "priors_time_path": "results/priors/detail_global_timeseries.csv",  # or "" to disable
        "signatures_path": "data/signatures.csv",
        "snv_counts_path": "results/preprocessing/tables/feature_store_snv.csv",
        "prior_only": False,
        "prior_only_fill_from_counts": True,
        "robust_c": 4.685, "prior_jitter_beta0": 0.5,
        "temporal_smooth_lambda_l2": 0.02, "lambda_ridge": 5e-4,
        "overlap_penalty_lambda": 0.0, "unknown_extra": 0.0,
        "irls_max_iter": 5, "admm_max_iter": 200, "admm_tol": 1e-6,
        "admm_rho": 1.0, "admm_adaptive_rho": True, "admm_rho_floor": 1e-6, "admm_rho_ceil": 1e6,
        "aggregate_by_date": True, "compute_uncertainty": True, "compute_leverage": True,
    }}
    try:
        run_likelihood(cfg, _DummyCtx())
    except Exception as e:
        print("Smoke test note:", e)
