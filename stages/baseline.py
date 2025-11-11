# -*- coding: utf-8 -*-
"""
Baseline Bayesian evidence & model comparison
Entry:
    run_baseline(cfg: Dict[str, Any], ctx: Any) -> Dict[str, Any]

Families:
- binomial  (Beta prior)               -> (count, coverage)
- bernoulli (Beta prior)               -> 0/1 column
- poisson   (Gamma prior, RATE β)      -> non-negative integer counts
- gaussian  (Normal–Inverse–Gamma)     -> continuous column

Outputs (overwritten each run) under a single 'baseline' root:
tables/
  - baselines_detailed.csv
  - bayes_factors_full.csv
  - postpred_all_models.csv       (per-row posterior-predictive with 'ps')
  - all_models.csv                (evidence + ranks + merged metrics)
metrics/
  - baselines_metrics.csv
  - baselines_pit_hist.csv

Folder rule (no duplicated 'baseline'):
- If cfg.baseline.save_root is set -> use it AS-IS
- Else -> (cfg.io.results_dir or 'results')/baseline
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import betabinom, nbinom, t as student_t, kstest, norm, chisquare

# ---------------- Speed-safe defaults ----------------
# Cap threads to avoid oversubscription; deterministic runs.
max_threads = max((os.cpu_count() or 8) - 1, 1)
os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.seterr(all="ignore")  # safely handle tail behavior; we clip where needed.

# ---------------- Constants ----------------
# Use singular folder name to match your tree: results\\baseline\\...
OUT_DIR_NAME = "baseline"
PIT_BINS = 20  # PIT histogram bins (mid-CDF binned uniformity check)


# ============================ Logging ============================
def _log(ctx: Any, level: str = "INFO", message: str = "", context: Optional[dict] = None) -> None:
    try:
        if hasattr(ctx, "log") and callable(getattr(ctx, "log")):
            try:
                ctx.log(level, message, context or {})
                return
            except TypeError:
                ctx.log(level=level, message=message, context=context or {})
                return
    except Exception:
        pass
    print(json.dumps({
        "time": pd.Timestamp.utcnow().isoformat(),
        "level": level,
        "stage": "baseline",
        "message": message,
        "context": context or {}
    }))


# ============================ Utilities ============================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _read_csv_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def _clip01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip to (eps, 1-eps) with explicit float dtype."""
    return np.clip(np.asarray(x, float), eps, 1.0 - eps)

def _is_binary(x: np.ndarray) -> bool:
    return x.size > 0 and np.all(np.isin(x, [0, 1])) and np.all(np.isfinite(x))

def _is_nonneg_int(x: np.ndarray) -> bool:
    return x.size > 0 and np.all((x >= 0) & np.isfinite(x) & (np.floor(x) == x))

def _sanitize_series(s: pd.Series, fill=0) -> pd.Series:
    """Coerce to numeric, count bads for logging (filled)."""
    sn = pd.to_numeric(s, errors="coerce")
    n_bad = int((~np.isfinite(sn)).sum())
    if n_bad:
        # only logging at call sites to include column name
        pass
    return sn.fillna(fill)


# ============================ Config & IO ============================
def _cfg_root(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("baseline", cfg or {}) if isinstance(cfg, dict) else {}


def _resolve_out_root(cfg: Dict[str, Any]) -> Path:
    """
    Always create results/baseline as the output root.
      - If cfg.baseline.save_root is provided → use that path.
      - Else → (cfg.io.results_dir or 'results')/baseline.
    """
    io = cfg.get("io", {}) if isinstance(cfg, dict) else {}
    bl = _cfg_root(cfg)

    # if user defines save_root, use it directly
    save_root = bl.get("save_root")
    if isinstance(save_root, str) and save_root.strip():
        base = Path(save_root)
    else:
        base = Path(io.get("results_dir") or "results")

    # Always append 'baseline' unless the path already ends with it
    if base.name.lower() != "baseline":
        base = base / "baseline"

    base.mkdir(parents=True, exist_ok=True)
    return base

def _cfg_data_path(cfg: Dict[str, Any]) -> Optional[str]:
    bl = _cfg_root(cfg)
    p = bl.get("data")
    if isinstance(p, str) and p.strip():
        return p
    return None


# ============================ Input autodetection ============================
def _pick_latest(paths: List[str]) -> str:
    """
    Pick the newest file by modification time; ties broken lexicographically.
    Raises if the list is empty.
    """
    if not paths:
        raise FileNotFoundError("No candidate files to choose from.")
    return max(paths, key=lambda p: (os.path.getmtime(p), p))

def _find_input(ctx: Any, explicit_path: Optional[str], base_dir: Optional[str]) -> str:
    """
    Detect input CSV (prefer newest):
      1) If cfg.baseline.data is a file that exists -> use it.
         If it is a glob -> pick newest match.
      2) Else search {base_dir}/preprocessing/tables/feature_store_snv*.csv (if base_dir given), pick newest.
      3) Else search results/preprocessing/tables/feature_store_snv*.csv (relative), pick newest.
      4) Else fallback to data/jahn_like.csv.
    """
    # 1) explicit file or glob
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            chosen = str(p.resolve())
            _log(ctx, "INFO", "Using explicit baseline input", {"path": chosen})
            return chosen
        if any(ch in explicit_path for ch in "*?[]"):
            matches = sorted(glob.glob(explicit_path))
            if matches:
                chosen = _pick_latest(matches)
                _log(ctx, "INFO", "Using explicit glob for baseline input", {"chosen": str(Path(chosen).resolve())})
                return chosen
        _log(ctx, "WARN", "Explicit input not found; falling back", {"path": explicit_path})

    # 2) within configured results_dir
    patterns: List[str] = []
    if base_dir:
        patterns.append(str(Path(base_dir) / "preprocessing" / "tables" / "feature_store_snv*.csv"))

    # 3) relative results/ (for when base_dir is not set)
    patterns.append(str(Path("results") / "preprocessing" / "tables" / "feature_store_snv*.csv"))

    for patt in patterns:
        cands = sorted(glob.glob(patt))
        if cands:
            chosen = _pick_latest(cands)
            _log(ctx, "INFO", "Auto-detected feature store", {"path": str(Path(chosen).resolve())})
            return chosen

    # 4) fallback
    fb = Path("data") / "jahn_like.csv"
    if fb.exists():
        _log(ctx, "INFO", "Falling back to jahn_like.csv", {"path": str(fb.resolve())})
        return str(fb)

    raise FileNotFoundError(
        "SNV table not found; expected <results_root>/preprocessing/tables/feature_store_snv*.csv "
        "or data/jahn_like.csv (you can also set cfg.baseline.data)."
    )


# ============================ Column detection ============================
def _pick_numeric_col(df: pd.DataFrame,
                      preferred: Optional[List[str]] = None,
                      requested: Optional[str] = None,
                      ctx: Any = None) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    if requested:
        hit = lower.get(requested.lower())
        if hit: return hit

    for name in (preferred or []):
        if not name: continue
        hit = lower.get(name.lower())
        if hit is not None: return hit

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        if ctx and requested and requested.lower() not in (c.lower() for c in cols):
            _log(ctx, "WARN", "Requested column missing; using first numeric",
                 {"requested": requested, "chosen": numeric[0]})
        return numeric[0]
    raise ValueError("No numeric column found for baseline evidence.")

def _pick_count_size_cols(df: pd.DataFrame, ctx: Any) -> Tuple[str, str]:
    lower = {c.lower(): c for c in df.columns}
    count_alias = ["count", "alt_count", "y", "k", "success", "altcount"]
    size_alias  = ["coverage", "n", "size", "total", "denom", "trials"]
    c = next((lower[x] for x in count_alias if x in lower), None)
    n = next((lower[x] for x in size_alias  if x in lower), None)
    if c is None or n is None:
        raise KeyError("Need count & size columns (e.g., 'count' and 'coverage') for binomial baseline.")
    return c, n

def _prepare_binomial_arrays(df: pd.DataFrame, c_col: str, n_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare Binomial y,n without corrupting information:
    - Keep n==0 rows as-is (valid, and must be preserved for correct evidence).
    - Clip y to [0, n].
    - Return also p_obs computed safely with max(n,1) for metrics only.
    """
    y_s = _sanitize_series(df[c_col])
    n_s = _sanitize_series(df[n_col])
    y = y_s.astype(np.int64, copy=False).to_numpy()
    n = n_s.astype(np.int64, copy=False).to_numpy()
    n = np.clip(n, 0, None)
    y = np.clip(y, 0, n)
    p_obs = y / np.maximum(n, 1)
    return y, n, p_obs


# ============================ Evidence (marginal likelihoods) ============================
def _log_binom_coeff(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float); k = np.asarray(k, float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)

def _evidence_binom_beta(y: np.ndarray, n: np.ndarray, a: float, b: float, include_comb: bool = True) -> float:
    s = float(np.sum(y)); t = float(np.sum(n))
    log_ev = betaln(a + s, b + (t - s)) - betaln(a, b)
    if include_comb:
        log_ev += float(np.sum(_log_binom_coeff(n, y)))
    return float(log_ev)

def _evidence_bern_beta(x: np.ndarray, a: float, b: float) -> float:
    N = int(x.size); s = float(np.sum(x))
    return float(betaln(a + s, b + (N - s)) - betaln(a, b))

def _evidence_pois_gamma_rate(x: np.ndarray, a: float, beta_rate: float) -> float:
    N = int(x.size); s = float(np.sum(x))
    # p(x|a,beta) = beta^a / Gamma(a) * Gamma(a+s) / (beta+N)^(a+s) * 1/prod(x_i!)
    return float(-np.sum(gammaln(x + 1.0)) + a*np.log(beta_rate)
                 - (a + s)*np.log(beta_rate + N) + gammaln(a + s) - gammaln(a))

def _evidence_gauss_nig(x: np.ndarray, mu0: float, k0: float, a0: float, b0: float) -> float:
    """
    Exact Normal–Inverse–Gamma marginal evidence:
    p(x) = (Γ(aN)/Γ(a0)) * (b0^a0 / bN^aN) * sqrt(k0/kN) * π^(−N/2)

    NOTE: constant is π^(−N/2), not (2π)^(−N/2).
    """
    x = np.asarray(x, float)
    N = int(x.size)
    if N == 0:
        return float("nan")
    xbar = float(np.mean(x))
    sse  = float(np.sum((x - xbar)**2))
    kN = k0 + N
    aN = a0 + 0.5*N
    bN = b0 + 0.5*sse + (k0*N*(xbar - mu0)**2)/(2.0*kN)
    return float(
        gammaln(aN) - gammaln(a0)
        + a0*np.log(b0) - aN*np.log(bN)
        + 0.5*(np.log(k0) - np.log(kN))
        - 0.5*N*np.log(np.pi)  # corrected constant
    )


# ============================ Beta–Binomial fallbacks ============================
def _bb_mean_var(n: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float); a = np.asarray(a, float); b = np.asarray(b, float)
    mu = a/(a + b)
    var = n * (a*b/(a + b)**2) * (a + b + n)/(a + b + 1.0)
    return n*mu, var

def _bb_ppf_normal(q: float, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, v = _bb_mean_var(n, a, b)
    z = norm.ppf(q)
    qv = m + z*np.sqrt(np.maximum(v, 1e-18))
    qv += np.sign(qv - m)*0.5  # continuity correction
    return np.clip(np.round(qv), 0, n).astype(int)

def _safe_bb_ppf(q: float, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Beta–Binomial percentile:
    - Try SciPy's exact ppf
    - Fall back to Normal approximation with continuity correction in extreme tails
    """
    n = np.asarray(n, int)
    shape = n.shape
    a = np.broadcast_to(np.asarray(a, float), shape)
    b = np.broadcast_to(np.asarray(b, float), shape)
    try:
        r = np.asarray(betabinom.ppf(q, n, a, b), float)
        if r.shape != shape:
            r = np.full(shape, float(r), float)
    except Exception:
        r = np.full(shape, np.nan, float)
    bad = ~np.isfinite(r)
    if np.any(bad):
        r[bad] = _bb_ppf_normal(q, n[bad], a[bad], b[bad])
    return np.clip(r, 0, n).astype(int, copy=False)

def _bb_cdf_normal(y: np.ndarray, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float); n = np.asarray(n, float)
    a = np.asarray(a, float); b = np.asarray(b, float)
    m, v = _bb_mean_var(n, a, b)
    z = (y + 0.5 - m)/np.sqrt(np.maximum(v, 1e-18))
    return _clip01(norm.cdf(z), 1e-12)

def _safe_bb_cdf(y: np.ndarray, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Beta–Binomial CDF:
    - Try SciPy's exact cdf
    - Fall back to Normal approximation in extreme tails
    """
    y = np.asarray(y, int)
    n = np.asarray(n, int)
    shape = y.shape
    a = np.broadcast_to(np.asarray(a, float), shape)
    b = np.broadcast_to(np.asarray(b, float), shape)
    try:
        F = np.asarray(betabinom.cdf(y, n, a, b), float)
        if F.shape != shape:
            F = np.full(shape, float(F), float)
    except Exception:
        F = np.full(shape, np.nan, float)
    bad = ~np.isfinite(F)
    if np.any(bad):
        F[bad] = _bb_cdf_normal(y[bad], n[bad], a[bad], b[bad])
    return _clip01(F, 1e-12)


# ============================ Posterior predictive ============================
def _coverage_rate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m): return float("nan")
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))

def _eval_binom_postpred(y: np.ndarray, n: np.ndarray, a: float, b: float):
    # Posterior parameters (conjugate), using all rows (in-sample PPC)
    s = float(np.sum(y)); t = float(np.sum(n))
    a_post = a + s; b_post = b + (t - s)
    p_mean = a_post / (a_post + b_post)   # posterior mean of p
    y_hat  = n * p_mean                    # mean of posterior predictive for each row

    # Log posterior predictive (Beta-Binomial) and discrete PIT via mid-CDF
    logp = (gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
            + gammaln(y + a_post) + gammaln(n - y + b_post) - gammaln(n + a_post + b_post)
            - (gammaln(a_post) + gammaln(b_post) - gammaln(a_post + b_post)))
    Fy   = _safe_bb_cdf(y, n, a_post, b_post)
    Fym1 = _safe_bb_cdf(np.maximum(y-1, 0), n, a_post, b_post)
    pit  = _clip01(0.5*(Fy + Fym1), 1e-12)

    lo50 = _safe_bb_ppf(0.25, n, a_post, b_post)
    hi50 = _safe_bb_ppf(0.75, n, a_post, b_post)
    lo90 = _safe_bb_ppf(0.05,  n, a_post, b_post)
    hi90 = _safe_bb_ppf(0.95,  n, a_post, b_post)
    return p_mean, y_hat, logp, pit, (lo50, hi50), (lo90, hi90)

def _eval_bern_postpred(x: np.ndarray, a: float, b: float):
    y = x.astype(int); n = np.ones_like(y, int)
    return _eval_binom_postpred(y, n, a, b)

def _eval_pois_postpred(x: np.ndarray, a: float, beta_rate: float):
    """
    Poisson-Gamma(rate) posterior predictive:
    x | λ ~ Pois(λ), λ ~ Gamma(a, beta)
    => x | data ~ NegBin(r=a+sum x, p=(beta+N)/(beta+N+1))
    """
    x = np.asarray(x, np.int64)
    s = float(np.sum(x))
    N = int(x.size)
    a_post = a + s
    b_post = beta_rate + N
    r = a_post
    p = b_post / (b_post + 1.0)  # SciPy nbinom parameterization uses "n=r, p=p"
    y_hat = np.full_like(x, a_post / b_post, dtype=float)

    logp = nbinom.logpmf(x, n=r, p=p).astype(float, copy=False)
    Fy   = nbinom.cdf(x, n=r, p=p).astype(float, copy=False)
    Fym1 = nbinom.cdf(np.maximum(x - 1, 0), n=r, p=p).astype(float, copy=False)
    pit  = _clip01(0.5 * (Fy + Fym1), 1e-12)

    lo50 = nbinom.ppf(0.25, n=r, p=p).astype(float, copy=False)
    hi50 = nbinom.ppf(0.75, n=r, p=p).astype(float, copy=False)
    lo90 = nbinom.ppf(0.05,  n=r, p=p).astype(float, copy=False)
    hi90 = nbinom.ppf(0.95,  n=r, p=p).astype(float, copy=False)
    return None, y_hat, logp, pit, (lo50, hi50), (lo90, hi90)

def _eval_gauss_postpred(x: np.ndarray, mu0: float, k0: float, a0: float, b0: float):
    """
    Normal–Inverse–Gamma posterior predictive:
    x_new | data ~ Student-t(df=2aN, loc=muN, scale = sqrt(bN*(kN+1)/(aN*kN)).
    """
    x = np.asarray(x, float)
    N = x.size
    if N == 0:
        raise ValueError("Gaussian column has no finite values.")
    S = float(np.sum(x))
    Q = float(np.sum(x * x))
    xbar = S / N
    SSE = Q - N * xbar * xbar

    kN = k0 + N
    aN = a0 + 0.5 * N
    bN = b0 + 0.5 * SSE + (k0 * N * (xbar - mu0) ** 2) / (2.0 * kN)
    muN = (k0 * mu0 + S) / kN

    df = 2.0 * aN
    scale = np.sqrt(bN * (kN + 1.0) / (aN * kN))

    mean_pred = np.full_like(x, muN, dtype=float)

    logp = student_t.logpdf(x, df=df, loc=muN, scale=scale).astype(float, copy=False)
    pit  = _clip01(student_t.cdf(x, df=df, loc=muN, scale=scale), 1e-12)

    lo50 = student_t.ppf(0.25, df=df, loc=muN, scale=scale).astype(float, copy=False)
    hi50 = student_t.ppf(0.75, df=df, loc=muN, scale=scale).astype(float, copy=False)
    lo90 = student_t.ppf(0.05,  df=df, loc=muN, scale=scale).astype(float, copy=False)
    hi90 = student_t.ppf(0.95,  df=df, loc=muN, scale=scale).astype(float, copy=False)
    return None, mean_pred, logp, pit, (lo50, hi50), (lo90, hi90)


# ============================ Default priors ============================
DEFAULT_BASELINES: Dict[str, List[Dict[str, Any]]] = {
    "bernoulli": [
        {"name": "jeffreys", "alpha": 0.5, "beta": 0.5},
        {"name": "uniform",  "alpha": 1.0, "beta": 1.0},
        {"name": "beta22",   "alpha": 2.0, "beta": 2.0},
    ],
    "binomial": [
        {"name": "jeffreys", "alpha": 0.5, "beta": 0.5},
        {"name": "uniform",  "alpha": 1.0, "beta": 1.0},
        {"name": "beta22",   "alpha": 2.0, "beta": 2.0},
    ],
    "poisson": [
        {"name": "gamma_1_1_rate", "alpha": 1.0, "beta": 1.0},
        {"name": "gamma_2_1_rate", "alpha": 2.0, "beta": 1.0},
        {"name": "gamma_3_1_rate", "alpha": 3.0, "beta": 1.0},
    ],
    "gaussian": [
        {"name": "weak_auto", "mu0": "auto_mean", "kappa0": 1e-3, "alpha0": 2.0, "beta0": "auto_scale"},
        {"name": "mild_auto", "mu0": "auto_mean", "kappa0": 1e-1, "alpha0": 2.0, "beta0": "auto_scale"},
    ],
}

def _resolve_nig_from_data(x: np.ndarray, pr: Dict[str, Any]) -> Dict[str, Any]:
    mu_hat = float(np.mean(x)) if x.size else 0.0
    s2_hat = float(np.var(x, ddof=1)) if x.size > 1 else float(np.var(x)) if x.size else 1.0
    # In Inv-Gamma(alpha0, beta0), E[sigma^2] = beta0/(alpha0-1) for alpha0>1
    beta_auto = float(max(1e-12, s2_hat * max(1.0, pr.get("alpha0", 2.0) - 1.0)))
    return {
        "name": pr.get("name", "auto"),
        "mu0": (mu_hat if pr.get("mu0") in ("auto", "auto_mean") else float(pr["mu0"])),
        "kappa0": float(pr["kappa0"]),
        "alpha0": float(pr["alpha0"]),
        "beta0": (beta_auto if pr.get("beta0") in ("auto", "auto_scale") else float(pr["beta0"])),
    }


# ============================ Writers & Bayes factors ============================
def _flatten_row(run_id: str, family: str, prior: Dict[str, Any], log_evidence_family: float,
                 extras: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": run_id,
        "family": family,
        "prior_name": prior.get("name", ""),
        "log_evidence": float(log_evidence_family),
    }
    for k, v in prior.items():
        if k != "name":
            row[f"prior_{k}"] = v
    row.update(extras)
    return row

def _bayes_factors_from_scores(ids: List[str], log_evs: np.ndarray, scope: str, fam: str):
    """Numerically stable Bayes factors; include log-BF and cap linear BF to avoid overflow."""
    out = []
    order = np.argsort(-log_evs)
    ids_sorted = [ids[i] for i in order]
    le_sorted  = log_evs[order]
    for i in range(len(ids_sorted)):
        for j in range(i+1, len(ids_sorted)):
            lbf = float(le_sorted[i] - le_sorted[j])
            bf  = float(np.exp(lbf)) if lbf < 700 else float('inf')
            out.append({"scope": scope, "family": fam,
                        "better_model": ids_sorted[i], "worse_model": ids_sorted[j],
                        "log_bayes_factor": lbf, "bayes_factor": bf})
    return out

def _write_tables(paths: Dict[str, Path], all_rows: List[Dict[str, Any]], drop_constant_used: bool) -> None:
    if not all_rows: return
    tables_dir = paths["tables"]; _ensure_dir(tables_dir)
    df = pd.DataFrame(all_rows).copy()

    for col in ("log_evidence_family", "log_evidence_raw", "log_evidence_noconst", "dropped_constant"):
        if col not in df.columns:
            df[col] = drop_constant_used if col == "dropped_constant" else df["log_evidence"].astype(float)

    # within-family posteriors & ranks
    df["posterior_prob_family"] = np.nan; df["family_rank"] = np.nan
    for fam, idx in df.groupby("family").groups.items():
        g = df.loc[idx]; le = g["log_evidence_family"].astype(float).to_numpy()
        w = np.exp(le - logsumexp(le))
        order = np.argsort(-le); rank = np.empty_like(order); rank[order] = np.arange(1, len(le)+1)
        df.loc[idx, "posterior_prob_family"] = w
        df.loc[idx, "family_rank"] = rank

    # global posteriors & ranks
    le_g = df["log_evidence_raw"].astype(float).to_numpy()
    df["posterior_prob_global"] = np.exp(le_g - logsumexp(le_g))
    order_g = np.argsort(-le_g); rank_g = np.empty_like(order_g); rank_g[order_g] = np.arange(1, len(df)+1)
    df["global_rank"] = rank_g; df["best_in_family"] = df["family_rank"] == 1; df["best_global"] = df["global_rank"] == 1

    df_out = df.sort_values(["family", "family_rank", "global_rank"])
    _safe_write_csv(df_out, tables_dir / "baselines_detailed.csv")

    # Bayes factors (stable)
    bf_rows = []
    for fam, g in df.groupby("family"):
        g_sorted = g.sort_values("log_evidence_family", ascending=False)
        ids = g_sorted["model_id"].tolist(); le = g_sorted["log_evidence_family"].astype(float).to_numpy()
        bf_rows.extend(_bayes_factors_from_scores(ids, le, scope="family", fam=fam))
    g = df.sort_values("log_evidence_raw", ascending=False)
    ids = g["model_id"].tolist(); le = g["log_evidence_raw"].astype(float).to_numpy()
    bf_rows.extend(_bayes_factors_from_scores(ids, le, scope="global", fam="*"))
    if bf_rows:
        bf_df = pd.DataFrame(bf_rows)
        _safe_write_csv(bf_df, tables_dir / "bayes_factors_full.csv")

def _append_postpred_csv(paths: Dict[str, Path], model_id: str, family: str,
                         obs: np.ndarray, size: Optional[np.ndarray] = None,
                         ps: Optional[np.ndarray] = None, p_mean: Optional[np.ndarray] = None,
                         y_hat: Optional[np.ndarray] = None, logp: Optional[np.ndarray] = None,
                         pit: Optional[np.ndarray] = None, lo50: Optional[np.ndarray] = None,
                         hi50: Optional[np.ndarray] = None, lo90: Optional[np.ndarray] = None,
                         hi90: Optional[np.ndarray] = None) -> None:
    tables_dir = paths["tables"]; _ensure_dir(tables_dir)
    out_path = tables_dir / "postpred_all_models.csv"

    def as_col(v: Optional[np.ndarray], N: int) -> np.ndarray:
        if v is None: return np.full(N, np.nan, float)
        v = np.asarray(v)
        return (np.full(N, float(v), float) if v.ndim == 0 else v.astype(float, copy=False))

    N = int(len(obs))
    df_out = pd.DataFrame({
        "model_id": np.repeat(model_id, N),
        "family":   np.repeat(family, N),
        "row":      np.arange(N, dtype=int),
        "obs":      as_col(obs,  N),
        "size":     as_col(size, N),
        "ps":       as_col(ps,   N),
        "p_mean":   as_col(p_mean, N),
        "y_hat":    as_col(y_hat, N),
        "logp":     as_col(logp,  N),
        "pit":      as_col(pit,   N),
        "lo50":     as_col(lo50,  N),
        "hi50":     as_col(hi50,  N),
        "lo90":     as_col(lo90,  N),
        "hi90":     as_col(hi90,  N),
    })
    # Append-by-chunk for per-model PPC
    df_out.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)


# ============================ PIT χ² (extra calibration) ============================
def _pit_chisq(pit: np.ndarray, bins: int = PIT_BINS) -> Tuple[float, float]:
    """Binned χ² test vs. uniform; returns (chi2, pvalue)."""
    u = pit[np.isfinite(pit)]
    if u.size == 0:
        return float("nan"), float("nan")
    h, _ = np.histogram(u, bins=bins, range=(0, 1))
    exp = np.full_like(h, u.size / bins, dtype=float)
    chi2, p = chisquare(h, f_exp=exp)
    return float(chi2), float(p)


# ============================ Stage main ============================
def run_baseline(cfg: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    _log(ctx, "INFO", "Stage started", {"stage": "baseline"})

    out_root = _resolve_out_root(cfg); _ensure_dir(out_root)
    _log(ctx, "INFO", "Resolved output root", {"out_root": str(out_root.resolve())})

    base_dir = (cfg.get("io", {}) or {}).get("results_dir")
    data_path = _find_input(ctx, _cfg_data_path(cfg), base_dir)
    paths = {
        "root": out_root,
        "runs": out_root / "runs",
        "tables": out_root / "tables",
        "metrics": out_root / "metrics",
        "logs": out_root / "logs",
        "figures": out_root / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)


    # Reset per-run per-row CSV (we always write full per-row outputs).
    pp_all = paths["tables"] / "postpred_all_models.csv"
    if pp_all.exists():
        try:
            pp_all.unlink()
        except Exception as e:
            _log(ctx, "WARN", "Could not remove previous postpred_all_models.csv", {"error": str(e)})

    df = _read_csv_any(data_path)
    data_info = {"data_path": str(Path(data_path).resolve()), "data_n_rows": int(len(df))}

    # Fair within-family comparison for binomial: drop combinational constant.
    drop_binom_comb = True

    all_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    pit_hist_rows: List[Dict[str, Any]] = []

    # ------------ Binomial ------------
    try:
        c_col, n_col = _pick_count_size_cols(df, ctx)

        # sanitize columns (log how many we fix)
        bad_y = int((~np.isfinite(pd.to_numeric(df[c_col], errors="coerce"))).sum())
        bad_n = int((~np.isfinite(pd.to_numeric(df[n_col], errors="coerce"))).sum())
        if bad_y or bad_n:
            _log(ctx, "WARN", "Non-finite values in binomial columns coerced to 0",
                 {"count_bad_y": bad_y, "count_bad_n": bad_n})

        # Correct preparation that preserves n==0 exactly:
        y, n, p_obs = _prepare_binomial_arrays(df, c_col, n_col)

        priors = DEFAULT_BASELINES["binomial"]
        run_dir = paths["runs"] / "binomial"; _ensure_dir(run_dir)
        (run_dir / "meta.json").write_text(
            json.dumps({"family": "binomial", **data_info}, indent=2),
            encoding="utf-8"
        )
        (run_dir / "run_summary.jsonl").write_text("", encoding="utf-8")

        for i, pr in enumerate(priors):
            model_id = f"binomial_{i:02d}_{pr['name']}"
            a, b = float(pr["alpha"]), float(pr["beta"])

            # Evidence (with/without combinational constant)
            log_raw = _evidence_binom_beta(y, n, a, b, include_comb=True)
            log_nc  = _evidence_binom_beta(y, n, a, b, include_comb=False)
            log_fam = log_nc if drop_binom_comb else log_raw

            p_mean, y_hat, logp, pit, (lo50, hi50), (lo90, hi90) = _eval_binom_postpred(y, n, a, b)

            rmse_p = float(np.sqrt(np.mean((p_mean - p_obs) ** 2)))
            mae_p  = float(np.mean(np.abs(p_mean - p_obs)))
            rmse_y = float(np.sqrt(np.mean((y_hat - y) ** 2)))
            mae_y  = float(np.mean(np.abs(y_hat - y)))
            cov50  = _coverage_rate(y, lo50, hi50)
            cov90  = _coverage_rate(y, lo90, hi90)
            finite = np.isfinite(logp)
            lpd_sum = float(np.sum(logp[finite])) if np.any(finite) else float("-inf")
            u = pit[np.isfinite(pit)]
            ks_p = float(kstest(u, 'uniform').pvalue) if u.size else float('nan')
            var_ratio = float(np.var(u) / (1.0 / 12.0)) if u.size else float('nan')
            chi2_stat, chi2_p = _pit_chisq(pit, bins=PIT_BINS)

            row = _flatten_row("binomial", "binomial", pr, log_fam, {
                "model_id": model_id, "prior_type": "beta",
                "log_evidence_family": float(log_fam),
                "log_evidence_raw": float(log_raw),
                "log_evidence_noconst": float(log_nc),
                "dropped_constant": bool(drop_binom_comb),
                "data_count_col": c_col, "data_size_col": n_col
            })
            all_rows.append(row)

            metrics_rows.append({
                "model_id": model_id, "family": "binomial", "prior_name": pr["name"],
                "rmse_p": rmse_p, "mae_p": mae_p, "rmse_y": rmse_y, "mae_y": mae_y,
                "lpd_sum": lpd_sum, "coverage_50": cov50, "coverage_90": cov90,
                "pit_ks_p": ks_p, "pit_var_ratio": var_ratio, "pit_chi2_stat": chi2_stat, "pit_chi2_p": chi2_p,
                "n_obs": int(len(y))
            })

            if u.size:
                h, edges = np.histogram(u, bins=PIT_BINS, range=(0, 1))
                for b_idx, cnt in enumerate(h):
                    pit_hist_rows.append({
                        "model_id": model_id, "family": "binomial",
                        "bin_left": float(edges[b_idx]), "bin_right": float(edges[b_idx + 1]),
                        "count": int(cnt), "sample_n": int(u.size)
                    })

            with open(run_dir / "run_summary.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"model_id": model_id, **row}) + "\n")

            _append_postpred_csv(
                paths, model_id, "binomial",
                obs=y, size=n, ps=p_mean, p_mean=p_mean, y_hat=y_hat,
                logp=logp, pit=pit, lo50=lo50, hi50=hi50, lo90=lo90, hi90=hi90
            )
    except Exception as e:
        _log(ctx, "WARN", "Binomial skipped", {"error": str(e)})

    # ------------ Poisson ------------
    try:
        x_col = _pick_numeric_col(df, preferred=["count", "alt_count", "y"], ctx=ctx)
        x_s = _sanitize_series(df[x_col])
        bad_x = int((~np.isfinite(pd.to_numeric(df[x_col], errors="coerce"))).sum())
        if bad_x:
            _log(ctx, "WARN", "Non-finite values in Poisson column coerced to 0", {"count_bad": bad_x})

        x = x_s.astype(float).to_numpy()
        x = np.clip(np.round(x), 0, None).astype(np.int64)
        if not _is_nonneg_int(x):
            _log(ctx, "WARN", "Poisson coerced non-neg ints", {"column": x_col})

        priors = DEFAULT_BASELINES["poisson"]
        run_dir = paths["runs"] / "poisson"; _ensure_dir(run_dir)
        (run_dir / "meta.json").write_text(
            json.dumps({"family": "poisson", **data_info}, indent=2),
            encoding="utf-8"
        )
        (run_dir / "run_summary.jsonl").write_text("", encoding="utf-8")

        for i, pr in enumerate(priors):
            model_id = f"poisson_{i:02d}_{pr['name']}"
            a, beta_rate = float(pr["alpha"]), float(pr["beta"])

            log_ev = _evidence_pois_gamma_rate(x, a, beta_rate)
            _, y_hat, logp, pit, (lo50, hi50), (lo90, hi90) = _eval_pois_postpred(x, a, beta_rate)

            rmse_y = float(np.sqrt(np.mean((y_hat - x) ** 2)))
            mae_y  = float(np.mean(np.abs(y_hat - x)))
            finite = np.isfinite(logp)
            lpd_sum = float(np.sum(logp[finite])) if np.any(finite) else float("-inf")
            u = pit[np.isfinite(pit)]
            ks_p = float(kstest(u, 'uniform').pvalue) if u.size else float('nan')
            var_ratio = float(np.var(u) / (1.0 / 12.0)) if u.size else float('nan')
            chi2_stat, chi2_p = _pit_chisq(pit, bins=PIT_BINS)

            row = _flatten_row("poisson", "poisson", pr, log_ev, {
                "model_id": model_id, "prior_type": "gamma_rate",
                "log_evidence_family": float(log_ev),
                "log_evidence_raw": float(log_ev),
                "log_evidence_noconst": float(log_ev),
                "dropped_constant": False, "data_column": x_col
            })
            all_rows.append(row)

            metrics_rows.append({
                "model_id": model_id, "family": "poisson", "prior_name": pr["name"],
                "rmse_p": np.nan, "mae_p": np.nan,
                "rmse_y": rmse_y, "mae_y": mae_y,
                "lpd_sum": lpd_sum, "coverage_50": _coverage_rate(x, lo50, hi50),
                "coverage_90": _coverage_rate(x, lo90, hi90),
                "pit_ks_p": ks_p, "pit_var_ratio": var_ratio, "pit_chi2_stat": chi2_stat, "pit_chi2_p": chi2_p,
                "n_obs": int(len(x))
            })

            if u.size:
                h, edges = np.histogram(u, bins=PIT_BINS, range=(0, 1))
                for b_idx, cnt in enumerate(h):
                    pit_hist_rows.append({
                        "model_id": model_id, "family": "poisson",
                        "bin_left": float(edges[b_idx]), "bin_right": float(edges[b_idx + 1]),
                        "count": int(cnt), "sample_n": int(u.size)
                    })

            with open(run_dir / "run_summary.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"model_id": model_id, **row}) + "\n")

            _append_postpred_csv(
                paths, model_id, "poisson",
                obs=x, size=None, ps=y_hat, p_mean=None, y_hat=y_hat,
                logp=logp, pit=pit, lo50=lo50, hi50=hi50, lo90=lo90, hi90=hi90
            )
    except Exception as e:
        _log(ctx, "WARN", "Poisson skipped", {"error": str(e)})

    # ------------ Gaussian ------------
    try:
        g_col = _pick_numeric_col(df, preferred=["value", "growth_rate", "x"], ctx=ctx)
        g_s = _sanitize_series(df[g_col])
        bad_g = int((~np.isfinite(pd.to_numeric(df[g_col], errors="coerce"))).sum())
        if bad_g:
            _log(ctx, "WARN", "Non-finite values in Gaussian column dropped", {"count_bad": bad_g})

        g = g_s.dropna().astype(float).to_numpy()
        if g.size == 0:
            raise ValueError("Gaussian column has no finite values.")

        priors_resolved = [_resolve_nig_from_data(g, pr) for pr in DEFAULT_BASELINES["gaussian"]]
        run_dir = paths["runs"] / "gaussian"; _ensure_dir(run_dir)
        (run_dir / "meta.json").write_text(
            json.dumps({"family": "gaussian", **data_info}, indent=2),
            encoding="utf-8"
        )
        (run_dir / "run_summary.jsonl").write_text("", encoding="utf-8")

        for i, pr in enumerate(priors_resolved):
            model_id = f"gaussian_{i:02d}_{pr['name']}"
            mu0, k0, a0, b0 = float(pr["mu0"]), float(pr["kappa0"]), float(pr["alpha0"]), float(pr["beta0"])

            log_ev = _evidence_gauss_nig(g, mu0, k0, a0, b0)
            _, mean_pred, logp, pit, (lo50, hi50), (lo90, hi90) = _eval_gauss_postpred(g, mu0, k0, a0, b0)

            rmse_y = float(np.sqrt(np.mean((mean_pred - g) ** 2)))
            mae_y  = float(np.mean(np.abs(mean_pred - g)))
            finite = np.isfinite(logp)
            lpd_sum = float(np.sum(logp[finite])) if np.any(finite) else float("-inf")
            u = pit[np.isfinite(pit)]
            ks_p = float(kstest(u, 'uniform').pvalue) if u.size else float('nan')
            var_ratio = float(np.var(u) / (1.0 / 12.0)) if u.size else float('nan')
            chi2_stat, chi2_p = _pit_chisq(pit, bins=PIT_BINS)

            row = _flatten_row("gaussian", "gaussian", pr, log_ev, {
                "model_id": model_id, "prior_type": "nig",
                "log_evidence_family": float(log_ev),
                "log_evidence_raw": float(log_ev),
                "log_evidence_noconst": float(log_ev),
                "dropped_constant": False, "data_column": g_col
            })
            all_rows.append(row)

            metrics_rows.append({
                "model_id": model_id, "family": "gaussian", "prior_name": pr["name"],
                "rmse_p": np.nan, "mae_p": np.nan,
                "rmse_y": rmse_y, "mae_y": mae_y,
                "lpd_sum": lpd_sum, "coverage_50": _coverage_rate(g, lo50, hi50),
                "coverage_90": _coverage_rate(g, lo90, hi90),
                "pit_ks_p": ks_p, "pit_var_ratio": var_ratio, "pit_chi2_stat": chi2_stat, "pit_chi2_p": chi2_p,
                "n_obs": int(len(g))
            })

            if u.size:
                h, edges = np.histogram(u, bins=PIT_BINS, range=(0, 1))
                for b_idx, cnt in enumerate(h):
                    pit_hist_rows.append({
                        "model_id": model_id, "family": "gaussian",
                        "bin_left": float(edges[b_idx]), "bin_right": float(edges[b_idx + 1]),
                        "count": int(cnt), "sample_n": int(u.size)
                    })

            with open(run_dir / "run_summary.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"model_id": model_id, **row}) + "\n")

            _append_postpred_csv(
                paths, model_id, "gaussian",
                obs=g, size=None, ps=mean_pred, p_mean=None, y_hat=mean_pred,
                logp=logp, pit=pit, lo50=lo50, hi50=hi50, lo90=lo90, hi90=hi90
            )
    except Exception as e:
        _log(ctx, "WARN", "Gaussian skipped", {"error": str(e)})

    # ------------ Bernoulli ------------
    try:
        b_col = _pick_numeric_col(df, preferred=["is_present", "present", "binary", "y"], ctx=ctx)
        x_s = _sanitize_series(df[b_col])
        bad_b = int((~np.isfinite(pd.to_numeric(df[b_col], errors="coerce"))).sum())
        if bad_b:
            _log(ctx, "WARN", "Non-finite values in Bernoulli column coerced to 0", {"count_bad": bad_b})

        x = x_s.fillna(0).to_numpy()
        x = np.where(x > 0, 1.0, 0.0).astype(np.int64)
        if not _is_binary(x):
            _log(ctx, "WARN", "Bernoulli column coerced to {0,1}", {"column": b_col})

        priors = DEFAULT_BASELINES["bernoulli"]
        run_dir = paths["runs"] / "bernoulli"; _ensure_dir(run_dir)
        (run_dir / "meta.json").write_text(
            json.dumps({"family": "bernoulli", **data_info}, indent=2),
            encoding="utf-8"
        )
        (run_dir / "run_summary.jsonl").write_text("", encoding="utf-8")

        for i, pr in enumerate(priors):
            model_id = f"bernoulli_{i:02d}_{pr['name']}"
            a, b = float(pr["alpha"]), float(pr["beta"])

            log_ev = _evidence_bern_beta(x, a, b)
            p_mean, y_hat, logp, pit, (lo50, hi50), (lo90, hi90) = _eval_bern_postpred(x, a, b)

            rmse_p = float(np.sqrt(np.mean((p_mean - x) ** 2)))
            mae_p  = float(np.mean(np.abs(p_mean - x)))
            rmse_y = float(np.sqrt(np.mean((y_hat - x) ** 2)))
            mae_y  = float(np.mean(np.abs(y_hat - x)))
            cov50  = _coverage_rate(x, lo50, hi50)
            cov90  = _coverage_rate(x, lo90, hi90)
            finite = np.isfinite(logp)
            lpd_sum = float(np.sum(logp[finite])) if np.any(finite) else float("-inf")
            u = pit[np.isfinite(pit)]
            ks_p = float(kstest(u, 'uniform').pvalue) if u.size else float('nan')
            var_ratio = float(np.var(u) / (1.0 / 12.0)) if u.size else float('nan')
            chi2_stat, chi2_p = _pit_chisq(pit, bins=PIT_BINS)

            row = _flatten_row("bernoulli", "bernoulli", pr, log_ev, {
                "model_id": model_id, "prior_type": "beta",
                "log_evidence_family": float(log_ev),
                "log_evidence_raw": float(log_ev),
                "log_evidence_noconst": float(log_ev),
                "dropped_constant": False, "data_column": b_col
            })
            all_rows.append(row)

            metrics_rows.append({
                "model_id": model_id, "family": "bernoulli", "prior_name": pr["name"],
                "rmse_p": rmse_p, "mae_p": mae_p, "rmse_y": rmse_y, "mae_y": mae_y,
                "lpd_sum": lpd_sum, "coverage_50": cov50, "coverage_90": cov90,
                "pit_ks_p": ks_p, "pit_var_ratio": var_ratio, "pit_chi2_stat": chi2_stat, "pit_chi2_p": chi2_p,
                "n_obs": int(len(x))
            })

            if u.size:
                h, edges = np.histogram(u, bins=PIT_BINS, range=(0, 1))
                for b_idx, cnt in enumerate(h):
                    pit_hist_rows.append({
                        "model_id": model_id, "family": "bernoulli",
                        "bin_left": float(edges[b_idx]), "bin_right": float(edges[b_idx + 1]),
                        "count": int(cnt), "sample_n": int(u.size)
                    })

            with open(run_dir / "run_summary.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"model_id": model_id, **row}) + "\n")

            _append_postpred_csv(
                paths, model_id, "bernoulli",
                obs=x, size=np.ones_like(x, int), ps=p_mean, p_mean=p_mean, y_hat=y_hat,
                logp=logp, pit=pit, lo50=lo50, hi50=hi50, lo90=lo90, hi90=hi90
            )
    except Exception as e:
        _log(ctx, "WARN", "Bernoulli skipped", {"error": str(e)})

    # ------------ Combined outputs ------------
    _write_tables(paths, all_rows, drop_constant_used=drop_binom_comb)

    # Merge evidence+ranks with metrics
    try:
        detailed_path = paths["tables"] / "baselines_detailed.csv"
        if detailed_path.exists():
            detailed = pd.read_csv(detailed_path)
            metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
            merged = detailed.merge(metrics_df, on=["model_id", "family", "prior_name"], how="left") \
                if not metrics_df.empty else detailed.copy()
            merged = merged.sort_values(["global_rank", "family", "family_rank"], na_position="last")
            _safe_write_csv(merged, paths["tables"] / "all_models.csv")
            _log(ctx, "INFO", "Wrote all_models.csv", {"rows": int(len(merged))})
    except Exception as e:
        _log(ctx, "WARN", "Failed to write all_models.csv", {"error": str(e)})

    metrics_dir = paths["metrics"]; _ensure_dir(metrics_dir)
    if metrics_rows:
        _safe_write_csv(pd.DataFrame(metrics_rows), metrics_dir / "baselines_metrics.csv")
    if pit_hist_rows:
        _safe_write_csv(pd.DataFrame(pit_hist_rows), metrics_dir / "baselines_pit_hist.csv")

    _log(ctx, "INFO", "Stage completed", {"tables_root": str(paths["tables"]), "metrics_root": str(metrics_dir)})

    return {
        "tables": ["baselines_detailed", "bayes_factors_full", "postpred_all_models", "all_models"],
        "metrics": ["baselines_metrics", "baselines_pit_hist"],
        "tables_dir": str(paths["tables"]), "metrics_dir": str(metrics_dir)
    }
