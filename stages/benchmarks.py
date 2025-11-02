from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

# Flat repo imports (root-anchored)
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))

from utils.run import RunContext  # type: ignore
from utils.plotting import set_matplotlib_style, place_legend_below  # type: ignore

try:
    from utils.seeds import set_global_seeds  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    def set_global_seeds(seed: int) -> None:
        np.random.seed(seed)


_EPS = 1e-12


# === Utilities ================================================================

def _ensure_datetime_series(x):
    import pandas as _pd
    if isinstance(x, _pd.Series):
        return _pd.to_datetime(x)
    return _pd.to_datetime(_pd.Series(x))


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.ones_like(v) / max(len(v), 1)
    return v / s


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Duchi et al. (2008) projection onto the probability simplex."""
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, n + 1)) > (cssv - 1))[0]
    if rho_idx.size == 0:
        theta = (cssv[-1] - 1) / n
    else:
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    return w / s if s > 0 else np.ones_like(w) / n


def rmse(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = a - b
    if weights is None:
        return float(np.sqrt(np.mean(r ** 2)))
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + _EPS)
    return float(np.sqrt(np.sum(w * (r ** 2))))


def r2_score(y: np.ndarray, yhat: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if weights is None:
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + _EPS
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + _EPS)
        ybar = float(np.sum(w * y))
        ss_res = float(np.sum(w * (y - yhat) ** 2))
        ss_tot = float(np.sum(w * (y - ybar) ** 2)) + _EPS
    return float(1.0 - ss_res / ss_tot)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _safe_normalize(np.asarray(p, dtype=float))
    q = _safe_normalize(np.asarray(q, dtype=float))
    m = 0.5 * (p + q)

    def kl(x, y):
        x = np.clip(x, _EPS, 1.0)
        y = np.clip(y, _EPS, 1.0)
        return np.sum(x * (np.log(x) - np.log(y)))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compute_overlap_score(S_sub: np.ndarray) -> float:
    """Mean off-diagonal column cosine similarity as an overlap score."""
    if S_sub.size == 0:
        return 0.0
    col_norms = np.linalg.norm(S_sub, axis=0) + _EPS
    C = (S_sub.T @ S_sub) / np.outer(col_norms, col_norms)
    L = C.shape[0]
    if L <= 1:
        return 0.0
    off_diag = C[~np.eye(L, dtype=bool)]
    return float(np.mean(off_diag))


def _log(ctx: RunContext, level: str, message: str,
         site_id: Optional[str] = None,
         lineage: Optional[str] = None,
         context: Optional[dict] = None) -> None:
    rec = {
        "time": pd.Timestamp.utcnow().isoformat(),
        "level": level.upper(),
        "stage": "benchmarks",
        "site_id": site_id or "",
        "lineage": lineage or "",
        "message": message,
        "context": context or {},
    }
    try:
        ctx.log(rec)  # type: ignore[attr-defined]
    except Exception:
        pass


# === Likelihood, weights, signatures =========================================

def beta_binom_logpmf_mean_phi(k: np.ndarray, n: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Beta–Binomial log PMF with mean/precision parameterization:
      a = mu * phi, b = (1-mu) * phi.
    This *depends on mu*, unlike a fixed-(alpha,beta) form.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), _EPS, 1 - _EPS)
    phi = np.maximum(np.asarray(phi, dtype=float), _EPS)

    a = mu * phi
    b = (1.0 - mu) * phi
    lcomb = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    return lcomb + betaln(k + a, (n - k) + b) - betaln(a, b)


def load_priors_alpha_beta(priors_path: Optional[Path],
                           mutations: List[str],
                           default_kappa: float = 100.0) -> Dict[str, Tuple[float, float]]:
    """Load mutation-specific (alpha, beta). Fallback: symmetric default_kappa."""
    ab: Dict[str, Tuple[float, float]] = {}
    if priors_path is not None and priors_path.exists():
        try:
            df = pd.read_csv(priors_path)
            cols = {c.lower(): c for c in df.columns}
            mut_col = cols.get("mutation", "mutation")
            if "alpha" in cols and "beta" in cols:
                for _, row in df.iterrows():
                    ab[str(row[mut_col])] = (float(row[cols["alpha"]]), float(row[cols["beta"]]))
        except Exception:
            pass
    for m in mutations:
        if m not in ab:
            ab[m] = (default_kappa / 2.0, default_kappa / 2.0)
    return ab


def compute_weights(n: np.ndarray,
                    y_af: np.ndarray,
                    alpha: np.ndarray,
                    beta: np.ndarray,
                    mode: str = "priors") -> np.ndarray:
    """Per-mutation weights for WLS on AF residuals."""
    n = np.asarray(n, dtype=float)
    p = np.clip(y_af, _EPS, 1 - _EPS)

    if mode == "identity":
        return np.ones_like(n)

    if mode == "coverage":
        return np.sqrt(n + _EPS)

    # mode == "priors": Beta–Binomial variance for AF (uses phi = alpha+beta)
    phi = np.asarray(alpha, dtype=float) + np.asarray(beta, dtype=float)
    rho = 1.0 / (phi + 1.0)
    var_af = p * (1 - p) / (n + _EPS) * (1.0 + (n - 1.0) * rho)
    w = 1.0 / np.sqrt(var_af + _EPS)
    # cap extreme weights
    if np.any(np.isfinite(w)):
        q = np.quantile(w[np.isfinite(w)], 0.99)
        w = np.clip(w, 0.0, float(q))
    return w


# === Data loading & signature matrix =========================================

def _prepare_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load feature store if available else raw data."""
    root = Path(cfg.get("root_dir", ".")).resolve()
    fs_candidates = list((root / "results" / "preprocessing" / "tables").glob("feature_store*.csv"))
    if len(fs_candidates) > 0:
        df_long = pd.read_csv(sorted(fs_candidates)[-1])
    else:
        df_long = pd.read_csv(root / "data" / "jahn_like.csv")

    sig_path = root / "data" / "signatures.csv"
    signatures = pd.read_csv(sig_path)

    lineages_path = root / "data" / "lineages.csv"
    lineages: Optional[pd.DataFrame] = None
    if lineages_path.exists():
        try:
            lineages = pd.read_csv(lineages_path)
        except Exception:
            lineages = None

    # Basic schema checks
    req_cols = ["sample_id", "site_id", "date", "mutation", "count", "coverage"]
    missing = [c for c in req_cols if c not in df_long.columns]
    if missing:
        raise ValueError(f"Input jahn_like missing columns: {missing}")
    sig_cols = ["mutation", "lineage", "weight"]
    missing_s = [c for c in sig_cols if c not in signatures.columns]
    if missing_s:
        raise ValueError(f"Input signatures missing columns: {missing_s}")

    # Parse date & restrict to mutations present in signatures
    df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")
    df_long = df_long.dropna(subset=["date"]).copy()
    sig_mutations = set(signatures["mutation"].astype(str).unique().tolist())
    df_long = df_long[df_long["mutation"].astype(str).isin(sig_mutations)].copy()
    return df_long, signatures, lineages


def _build_S_matrix(signatures: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build dense S matrix (M x L) from long signatures."""
    mutations = sorted(signatures["mutation"].astype(str).unique().tolist())
    lineages = sorted(signatures["lineage"].astype(str).unique().tolist())
    m_index = {m: i for i, m in enumerate(mutations)}
    l_index = {l: j for j, l in enumerate(lineages)}

    M = len(mutations)
    L = len(lineages)
    S = np.zeros((M, L), dtype=float)
    for _, row in signatures.iterrows():
        mi = m_index[str(row["mutation"])]
        lj = l_index[str(row["lineage"])]
        w = float(row["weight"])
        if w > 0:
            S[mi, lj] += w

    # Normalize columns to sum to 1 (stabilizes)
    col_sums = S.sum(axis=0) + _EPS
    S = S / col_sums[None, :]
    return S, mutations, lineages


# === Robust sample column picker =============================================

def _pick_col_by_sample(mat, samples, site_id, sample_id, date):
    """
    Find column in `mat` (features x samples) corresponding to (site_id, sample_id, date).
    Falls back to nearest date if exact triple not present.
    """
    import pandas as _pd, numpy as _np
    n_cols = mat.shape[1]
    ss = samples.copy()
    if not _np.issubdtype(ss["date"].dtype, _np.datetime64):
        ss["date"] = _pd.to_datetime(ss["date"])
    target_date = _pd.to_datetime(date)
    ss_reset = ss.reset_index(drop=True)

    mask = (ss_reset["site_id"] == site_id) & (ss_reset["sample_id"] == sample_id) & (ss_reset["date"] == target_date)
    if bool(mask.any()):
        pos = int(_np.flatnonzero(mask.values)[0])
    else:
        mask2 = (ss_reset["site_id"] == site_id) & (ss_reset["sample_id"] == sample_id)
        if bool(mask2.any()):
            idxs = _np.flatnonzero(mask2.values)
            diffs = _np.abs((ss_reset.loc[idxs, "date"].values.astype("datetime64[ns]") - target_date.to_datetime64())
                            .astype("timedelta64[ns]").astype(_np.int64))
            pos = int(idxs[int(_np.argmin(diffs))])
        else:
            pos = 0
    if pos < 0 or pos >= n_cols:
        pos = int(max(0, min(n_cols - 1, pos)))
    return mat[:, pos]


# === Optimization helpers (PGD, IRLS, NMF, FISTA, OMP) =======================

@dataclass
class MethodResult:
    theta: np.ndarray      # (L,)
    mu_pred: np.ndarray    # (M_used,)
    runtime_s: float
    peak_mem_mb: float
    status: str            # "ok" | "fail"
    message: str = ""


def armijo_backtracking(theta: np.ndarray, grad: np.ndarray, obj_fn, project,
                        t0: float = 1.0, beta: float = 0.5, c: float = 1e-4,
                        max_backtracks: int = 50) -> Tuple[np.ndarray, float]:
    """Backtracking line search with projection."""
    t = float(t0)
    f0 = float(obj_fn(theta))
    gdotd = -float(np.dot(grad, grad))  # d = -grad
    theta_new = theta
    f_new = f0
    for _ in range(max_backtracks):
        theta_new = project(theta - t * grad)
        f_new = float(obj_fn(theta_new))
        if f_new <= f0 + c * t * gdotd:
            break
        t *= beta
    return theta_new, f_new


def pgd_nnls(y_af: np.ndarray, S: np.ndarray, w: np.ndarray, lam: float,
             tol: float, max_iter: int) -> Tuple[np.ndarray, List[float]]:
    """Projected gradient descent on simplex for weighted least squares + L2."""
    M, L = S.shape
    theta = np.ones(L) / L
    Sw = S * (w[:, None] ** 2)

    def obj(th: np.ndarray) -> float:
        r = S @ th - y_af
        return 0.5 * float(np.sum((w * r) ** 2)) + 0.5 * lam * float(np.dot(th, th))

    history = [obj(theta)]
    for _ in range(max_iter):
        r = S @ theta - y_af
        grad = S.T @ (w ** 2 * r) + lam * theta
        theta_new, f_new = armijo_backtracking(theta, grad, obj, project_to_simplex)
        history.append(f_new)
        if abs(history[-2] - history[-1]) <= tol * (abs(history[-2]) + _EPS):
            theta = theta_new
            break
        theta = theta_new
    theta = project_to_simplex(theta)
    return theta, history


def irls_identity_binomial(y_af: np.ndarray, n_cov: np.ndarray, S: np.ndarray, lam: float,
                           tol: float, max_iter: int) -> Tuple[np.ndarray, List[float]]:
    """IRLS with identity link + simplex projection; WLS proxy to Binomial NLL."""
    M, L = S.shape
    theta = np.ones(L) / L
    hist: List[float] = []

    def obj(th: np.ndarray) -> float:
        p = np.clip(S @ th, _EPS, 1 - _EPS)
        var = p * (1 - p) / (n_cov + _EPS)
        w = 1.0 / (var + _EPS)
        r = p - y_af
        return 0.5 * float(np.sum(w * (r ** 2))) + 0.5 * lam * float(np.dot(th, th))

    for _ in range(max_iter):
        p = np.clip(S @ theta, _EPS, 1 - _EPS)
        var = p * (1 - p) / (n_cov + _EPS)
        w = 1.0 / (var + _EPS)
        z = p + (y_af - p)  # identity link
        A = S.T @ (w[:, None] * S) + lam * np.eye(L)
        b = S.T @ (w * z)
        try:
            th_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            th_new, _ = pgd_nnls(y_af, S, np.sqrt(w), lam=lam, tol=tol, max_iter=max_iter)
            theta = th_new
            hist.append(obj(theta))
            break
        theta = project_to_simplex(th_new)
        hist.append(obj(theta))
        if len(hist) >= 2 and abs(hist[-2] - hist[-1]) <= tol * (abs(hist[-2]) + _EPS):
            break
    theta = project_to_simplex(theta)
    return theta, hist


def nmf_fixed_dictionary(y_af_mat: np.ndarray, S: np.ndarray, W_mat: np.ndarray,
                         lam: float, max_iter: int, tol: float) -> Tuple[np.ndarray, List[float]]:
    """
    Multiplicative updates to estimate Theta for fixed dictionary S across samples.
    y_af_mat: (M, T), S: (M, L), W_mat: (M, T) (sqrt precision). Returns Theta: (L, T)
    """
    M, T = y_af_mat.shape
    _, L = S.shape
    Theta = np.ones((L, T), dtype=float) / L
    hist: List[float] = []
    S = np.asarray(S, dtype=float)

    for it in range(max_iter):
        R = S @ Theta  # (M, T)
        resid = R - y_af_mat
        obj = 0.5 * float(np.sum((W_mat * resid) ** 2)) + 0.5 * lam * float(np.sum(Theta ** 2))
        hist.append(obj)

        numer = S.T @ (W_mat ** 2 * y_af_mat)  # (L, T)
        denom = S.T @ (W_mat ** 2 * R) + lam * Theta + _EPS
        Theta *= numer / denom
        # project each column to simplex
        for t in range(T):
            Theta[:, t] = project_to_simplex(Theta[:, t])

        if it >= 1 and abs(hist[-2] - hist[-1]) <= tol * (abs(hist[-2]) + _EPS):
            break
    return Theta, hist


# === Sparse baselines: FISTA Lasso (FREYJA-like) and OMP (VAQUERO-like) ======

def _spectral_lipschitz(Sw: np.ndarray, l2: float, iters: int = 60) -> float:
    """
    Estimate the Lipschitz constant of ∇(0.5||Sw θ - y||^2 + 0.5 l2||θ||^2),
    i.e. λ_max(Sw^T Sw) + l2 via power iteration.
    """
    Ldim = Sw.shape[1]
    v = np.random.default_rng(42).normal(size=Ldim)
    v = v / (np.linalg.norm(v) + _EPS)
    for _ in range(iters):
        v = Sw.T @ (Sw @ v)
        nrm = np.linalg.norm(v)
        v = v / (nrm + _EPS)
    lam_max = float(v @ (Sw.T @ (Sw @ v)))
    return float(lam_max + l2)


def _prox_nonneg_l1(u: np.ndarray, lam: float) -> np.ndarray:
    """Proximal map for λ||·||_1 + I{·≥0}: soft-threshold then clip."""
    return np.maximum(u - lam, 0.0)


def fista_nonneg_lasso(y_af: np.ndarray, S: np.ndarray, w: np.ndarray,
                       l1: float, l2: float, max_iter: int, tol: float) -> Tuple[np.ndarray, List[float]]:
    """
    FISTA for non-negative Lasso on weighted LS:
      min 0.5 ||W(Sθ - y)||^2 + (l2/2)||θ||^2 + l1||θ||_1  s.t. θ ≥ 0.
    After convergence, renormalize to the simplex (sparse solution survives).
    """
    y = np.asarray(y_af, dtype=float)
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    Sw = S * (w[:, None])
    yw = y * w

    Ldim = S.shape[1]
    theta = np.zeros(Ldim, dtype=float)
    z = theta.copy()
    t = 1.0
    hist: List[float] = []

    Lstep = _spectral_lipschitz(Sw, l2)

    def obj(th: np.ndarray) -> float:
        r = Sw @ th - yw
        return 0.5 * float(np.dot(r, r)) + 0.5 * l2 * float(np.dot(th, th)) + l1 * float(np.sum(th))

    for _ in range(max_iter):
        # gradient at z
        grad = Sw.T @ (Sw @ z - yw) + l2 * z
        u = z - (1.0 / Lstep) * grad
        theta_new = _prox_nonneg_l1(u, l1 / Lstep)

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = theta_new + ((t - 1.0) / t_new) * (theta_new - theta)

        hist.append(obj(theta_new))
        if len(hist) >= 2 and abs(hist[-2] - hist[-1]) <= tol * (abs(hist[-2]) + _EPS):
            theta = theta_new
            break
        theta = theta_new
        t = t_new

    theta = project_to_simplex(theta)  # enforce proportions
    return theta, hist


def omp_ridge_simplex(y_af: np.ndarray, S: np.ndarray, w: np.ndarray,
                      max_k: int, ridge: float, tol: float) -> Tuple[np.ndarray, List[int]]:
    """
    Orthogonal Matching Pursuit (weighted) to select a sparse support, then ridge LS on the support.
    Finally project to simplex.
    """
    y = np.asarray(y_af, dtype=float)
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    Sw = S * (w[:, None])
    yw = y * w

    M, L = S.shape
    support: List[int] = []
    residual = yw.copy()
    last_obj = float(np.dot(residual, residual))

    for _ in range(max_k):
        corr = Sw.T @ residual
        j = int(np.argmax(corr))
        if j in support:
            break
        support.append(j)

        A = Sw[:, support]  # (M, k)
        AtA = A.T @ A + ridge * np.eye(len(support))
        Aty = A.T @ yw
        try:
            theta_s = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            break
        theta_s = np.maximum(theta_s, 0.0)

        residual = yw - A @ theta_s
        obj = float(np.dot(residual, residual))
        if abs(last_obj - obj) <= tol * (abs(last_obj) + _EPS):
            break
        last_obj = obj

    theta = np.zeros(L, dtype=float)
    if support:
        A = Sw[:, support]
        AtA = A.T @ A + ridge * np.eye(len(support))
        Aty = A.T @ yw
        try:
            theta_s = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            theta_s, _ = fista_nonneg_lasso(y_af, S, w, l1=1e-3, l2=ridge, max_iter=300, tol=1e-6)
            return theta_s, support
        theta_s = np.maximum(theta_s, 0.0)
        theta[support] = theta_s

    theta = project_to_simplex(theta)
    return theta, support


# === Binning utils for fairness ==============================================

def _bin_strata(values, quantiles, labels):
    """
    Bin 1-D numeric `values` into len(labels) strata using empirical quantiles.
    Robust to ties and degenerate quantiles.
    """
    import numpy as _np
    if len(labels) < 1:
        return []
    qs = [q for q in quantiles if 0.0 <= float(q) <= 1.0]
    qs = sorted(qs)
    if len(qs) == 0:
        edges = _np.array([-_np.inf, _np.inf], dtype=float)
    else:
        qvals = _np.quantile(_np.asarray(values, dtype=float), qs)
        edges = _np.concatenate(([-_np.inf], qvals, [_np.inf])).astype(float)
        for i in range(1, len(edges)):
            if not _np.isfinite(edges[i]):
                edges[i] = edges[i-1] + 1e-12
            elif edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1e-12
    idx = _np.digitize(values, edges[1:-1], right=True)
    idx = _np.clip(idx, 0, len(labels) - 1)
    return [labels[int(i)] for i in idx]


# === Main benchmark ===========================================================

def run_benchmarks(cfg: dict, ctx: RunContext) -> None:
    """
    Benchmark suite comparing deconvolution baselines on per-sample AF.

    Methods supported (no placeholders):
      - PGD     : WLS + L2 with simplex projection (dense).
      - GLM     : IRLS (identity link) + simplex (dense).
      - NMF     : multiplicative updates with simplex (dense).
      - FREYJA  : non-negative Lasso (FISTA) on WLS, renormalized to simplex (sparse).
      - VAQUERO : OMP support selection + ridge LS on support, then simplex (sparse).

    All methods output θ (lineage proportions) and μ_pred = Sθ on observed mutations.
    Likelihood metrics use **Beta–Binomial** with mean μ_pred and precision φ=α+β.
    """
    stage = "benchmarks"
    bench_cfg = cfg.get("benchmarks", {})
    seed = int(bench_cfg.get("seed", 12345))
    set_global_seeds(seed)
    _log(ctx, "INFO", "Starting benchmarks", None, None, {"seed": seed})

    # Methods and hyperparameters
    methods: List[str] = list(bench_cfg.get("methods", ["PGD", "GLM", "NMF", "FREYJA", "VAQUERO"]))
    lambda_reg = float(bench_cfg.get("lambda_reg", 1e-2))
    tol = float(bench_cfg.get("tol", 1e-6))
    max_iter = int(bench_cfg.get("max_iter", 500))

    irls_lambda = float(bench_cfg.get("irls_lambda", lambda_reg))
    irls_max_iter = int(bench_cfg.get("irls_max_iter", 100))

    nmf_lambda = float(bench_cfg.get("nmf_lambda", lambda_reg))
    nmf_max_iter = int(bench_cfg.get("nmf_max_iter", 200))

    weight_mode = str(bench_cfg.get("weight_mode", "priors"))
    default_kappa = float(bench_cfg.get("default_kappa", 100.0))

    # Sparse baselines hyperparameters
    freyja_l1 = float(bench_cfg.get("freyja_l1", 1e-3))
    freyja_l2 = float(bench_cfg.get("freyja_l2", 1e-3))
    freyja_max_iter = int(bench_cfg.get("freyja_max_iter", 500))
    freyja_tol = float(bench_cfg.get("freyja_tol", 1e-6))

    vaquero_max_k = int(bench_cfg.get("vaquero_max_k", 15))
    vaquero_ridge = float(bench_cfg.get("vaquero_ridge", 1e-3))
    vaquero_tol = float(bench_cfg.get("vaquero_tol", 1e-6))

    # Fairness config
    fairness_cfg = bench_cfg.get("fairness", {})
    coverage_quantiles = fairness_cfg.get("coverage_quantiles", [0.33, 0.66])
    overlap_quantiles = fairness_cfg.get("overlap_quantiles", [0.33, 0.66])
    site_size_bins = fairness_cfg.get("site_size_bins", [10, 30])
    jsd_baseline_name = bench_cfg.get("jsd_baseline", "PGD")
    root_dir = Path(cfg.get("root_dir", ".")).resolve()

    # Load data & signatures
    df_long, signatures, lineages_df = _prepare_data(cfg)
    S, mut_order, lin_order = _build_S_matrix(signatures)
    M, L = S.shape
    _log(ctx, "INFO", "Built signature matrix", None, None, {"M": M, "L": L})

    # Prepare per-sample matrices
    df_long["mutation"] = df_long["mutation"].astype(str)
    df_long["site_id"] = df_long["site_id"].astype(str)
    df_long["sample_id"] = df_long["sample_id"].astype(str)
    df_long = df_long[df_long["mutation"].isin(mut_order)].copy()

    samples = df_long[["sample_id", "site_id", "date"]].drop_duplicates().sort_values(["site_id", "date", "sample_id"])
    site_sizes = samples.groupby("site_id")["sample_id"].nunique().to_dict()

    # Priors α,β → φ for likelihood & weights
    priors_path_candidates = [
        root_dir / "results" / "priors" / "tables" / "priors_hyperparams.csv",
        root_dir / "results" / "priors" / "tables" / "priors_alpha_beta.csv",
    ]
    priors_path = None
    for p in priors_path_candidates:
        if p.exists():
            priors_path = p
            break
    ab_map = load_priors_alpha_beta(priors_path, mut_order, default_kappa=default_kappa)
    alpha_vec = np.array([ab_map[m][0] for m in mut_order], dtype=float)
    beta_vec  = np.array([ab_map[m][1] for m in mut_order], dtype=float)
    phi_vec   = alpha_vec + beta_vec

    # Build count/coverage matrices aligned to samples
    m_index = {m: i for i, m in enumerate(mut_order)}
    sample_groups = df_long.groupby(["sample_id", "site_id", "date"])
    obs_mat = np.zeros((M, len(samples)), dtype=float)
    cov_mat = np.zeros((M, len(samples)), dtype=float)
    for col_idx, (_, sdf) in enumerate(samples.iterrows()):
        sid = sdf["sample_id"]; site = sdf["site_id"]; date = sdf["date"]
        try:
            g = sample_groups.get_group((sid, site, date))
        except KeyError:
            continue
        for _, row in g.iterrows():
            i = m_index.get(row["mutation"])
            if i is None:
                continue
            cov_mat[i, col_idx] = float(row["coverage"])
            obs_mat[i, col_idx] = float(row["count"])

    with np.errstate(divide="ignore", invalid="ignore"):
        af_mat = np.where(cov_mat > 0, obs_mat / np.maximum(cov_mat, 1.0), np.nan)

    # Results & metrics
    metrics_records: List[dict] = []
    method_status_records: List[dict] = []
    baseline_thetas: Dict[Tuple[str, str, pd.Timestamp], np.ndarray] = {}

    # Iterate samples
    for t_idx, (_, srow) in enumerate(samples.iterrows()):
        sid = srow["sample_id"]; site = srow["site_id"]; date = pd.to_datetime(srow["date"])
        y_counts = obs_mat[:, t_idx].copy()
        n_cov    = cov_mat[:, t_idx].copy()
        mask = n_cov > 0
        if not np.any(mask):
            continue

        y = y_counts[mask]
        n = n_cov[mask]
        y_af = np.clip(y / np.maximum(n, 1.0), 0.0, 1.0)
        S_sub = S[mask, :]
        alpha_sub = alpha_vec[mask]
        beta_sub  = beta_vec[mask]
        phi_sub   = phi_vec[mask]
        w = compute_weights(n, y_af, alpha_sub, beta_sub, mode=weight_mode)

        overlap_score = compute_overlap_score(S_sub)
        coverage_mean = float(np.mean(n))
        site_size = int(site_sizes.get(site, 1))

        results_for_sample: Dict[str, MethodResult] = {}

        # --- PGD baseline (always compute; used for JSD baseline) ---
        tracemalloc.start(); t0 = time.perf_counter()
        try:
            theta_pgd, _ = pgd_nnls(y_af=y_af, S=S_sub, w=w, lam=lambda_reg, tol=tol, max_iter=max_iter)
            mu_pred = np.clip(S_sub @ theta_pgd, _EPS, 1 - _EPS)
            status = "ok"; msg = ""
        except Exception as e:
            theta_pgd = np.ones(S_sub.shape[1]) / S_sub.shape[1]
            mu_pred = np.clip(S_sub @ theta_pgd, _EPS, 1 - _EPS)
            status = "fail"; msg = f"PGD error: {e}"
        runtime_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

        res = MethodResult(theta=theta_pgd, mu_pred=mu_pred,
                           runtime_s=runtime_s, peak_mem_mb=peak/(1024**2),
                           status=status, message=msg)
        results_for_sample["PGD"] = res
        baseline_thetas[(site, sid, date)] = theta_pgd
        if "PGD" in methods:
            method_status_records.append({"method": "PGD", "site_id": site, "sample_id": sid, "date": date,
                                          "status": status, "message": msg})

        # --- GLM (IRLS identity) ---
        if "GLM" in methods:
            tracemalloc.start(); t0 = time.perf_counter()
            try:
                theta_glm, _ = irls_identity_binomial(y_af=y_af, n_cov=n, S=S_sub,
                                                      lam=irls_lambda, tol=tol, max_iter=irls_max_iter)
                mu_pred = np.clip(S_sub @ theta_glm, _EPS, 1 - _EPS)
                status = "ok"; msg = ""
            except Exception as e:
                theta_glm = np.ones(S_sub.shape[1]) / S_sub.shape[1]
                mu_pred = np.clip(S_sub @ theta_glm, _EPS, 1 - _EPS)
                status = "fail"; msg = f"GLM error: {e}"
            runtime_s = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results_for_sample["GLM"] = MethodResult(theta_glm, mu_pred, runtime_s, peak/(1024**2), status, msg)
            method_status_records.append({"method": "GLM", "site_id": site, "sample_id": sid, "date": date,
                                          "status": status, "message": msg})

        # --- NMF (fixed dictionary) ---
        if "NMF" in methods:
            tracemalloc.start(); t0 = time.perf_counter()
            try:
                Theta, _ = nmf_fixed_dictionary(y_af_mat=y_af[:, None], S=S_sub,
                                                W_mat=w[:, None], lam=nmf_lambda,
                                                max_iter=nmf_max_iter, tol=tol)
                theta_nmf = Theta[:, 0]
                mu_pred = np.clip(S_sub @ theta_nmf, _EPS, 1 - _EPS)
                status = "ok"; msg = ""
            except Exception as e:
                theta_nmf = np.ones(S_sub.shape[1]) / S_sub.shape[1]
                mu_pred = np.clip(S_sub @ theta_nmf, _EPS, 1 - _EPS)
                status = "fail"; msg = f"NMF error: {e}"
            runtime_s = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results_for_sample["NMF"] = MethodResult(theta_nmf, mu_pred, runtime_s, peak/(1024**2), status, msg)
            method_status_records.append({"method": "NMF", "site_id": site, "sample_id": sid, "date": date,
                                          "status": status, "message": msg})

        # --- FREYJA (sparse Lasso via FISTA) ---
        if "FREYJA" in methods:
            tracemalloc.start(); t0 = time.perf_counter()
            try:
                theta_fr, _ = fista_nonneg_lasso(y_af=y_af, S=S_sub, w=w,
                                                 l1=freyja_l1, l2=freyja_l2,
                                                 max_iter=freyja_max_iter, tol=freyja_tol)
                mu_pred = np.clip(S_sub @ theta_fr, _EPS, 1 - _EPS)
                status = "ok"; msg = ""
            except Exception as e:
                theta_fr = np.ones(S_sub.shape[1]) / S_sub.shape[1]
                mu_pred = np.clip(S_sub @ theta_fr, _EPS, 1 - _EPS)
                status = "fail"; msg = f"FREYJA error: {e}"
            runtime_s = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results_for_sample["FREYJA"] = MethodResult(theta_fr, mu_pred, runtime_s, peak/(1024**2), status, msg)
            method_status_records.append({"method": "FREYJA", "site_id": site, "sample_id": sid, "date": date,
                                          "status": status, "message": msg})

        # --- VAQUERO (OMP + ridge support fit) ---
        if "VAQUERO" in methods:
            tracemalloc.start(); t0 = time.perf_counter()
            try:
                theta_vq, _supp = omp_ridge_simplex(y_af=y_af, S=S_sub, w=w,
                                                    max_k=vaquero_max_k,
                                                    ridge=vaquero_ridge,
                                                    tol=vaquero_tol)
                mu_pred = np.clip(S_sub @ theta_vq, _EPS, 1 - _EPS)
                status = "ok"; msg = ""
            except Exception as e:
                theta_vq = np.ones(S_sub.shape[1]) / S_sub.shape[1]
                mu_pred = np.clip(S_sub @ theta_vq, _EPS, 1 - _EPS)
                status = "fail"; msg = f"VAQUERO error: {e}"
            runtime_s = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results_for_sample["VAQUERO"] = MethodResult(theta_vq, mu_pred, runtime_s, peak/(1024**2), status, msg)
            method_status_records.append({"method": "VAQUERO", "site_id": site, "sample_id": sid, "date": date,
                                          "status": status, "message": msg})

        # === Metrics per method ===
        theta_base = results_for_sample.get(jsd_baseline_name, results_for_sample["PGD"]).theta

        for mname, res in results_for_sample.items():
            if mname not in methods and mname != "PGD":
                continue

            mu_pred = res.mu_pred
            mask_finite = np.isfinite(y_af) & np.isfinite(mu_pred)
            if not np.any(mask_finite):
                continue

            y_k = y[mask_finite]
            n_k = n[mask_finite]
            af_obs = y_af[mask_finite]
            mu_k = np.clip(mu_pred[mask_finite], _EPS, 1 - _EPS)
            phi_k = phi_sub[mask_finite]

            # Metrics
            rmse_val = rmse(af_obs, mu_k)
            r2_val   = r2_score(af_obs, mu_k)
            nll_vec  = -beta_binom_logpmf_mean_phi(y_k, n_k, mu_k, phi_k)
            nll_val  = float(np.sum(nll_vec))
            jsd_theta = jensen_shannon_divergence(res.theta, theta_base) if mname != jsd_baseline_name else 0.0
            jsd_mu    = jensen_shannon_divergence(af_obs, mu_k)

            metrics_records.append({
                "site_id": site,
                "sample_id": sid,
                "date": date,
                "method": mname,
                "rmse": rmse_val,
                "r2": r2_val,
                "nll": nll_val,
                "jsd_theta": jsd_theta,
                "jsd_mu": jsd_mu,
                "runtime_s": res.runtime_s,
                "peak_mem_mb": res.peak_mem_mb,
                "coverage_mean": coverage_mean,
                "site_size": site_size,
                "overlap_score": overlap_score,
                "status": res.status,
                "message": res.message,
                "L": S_sub.shape[1],
                "M_used": S_sub.shape[0],
            })

        if (t_idx + 1) % 50 == 0:
            _log(ctx, "INFO", "Processed samples", site, None, {"processed": t_idx + 1, "total": len(samples)})

    if len(metrics_records) == 0:
        _log(ctx, "ERROR", "No metrics computed; aborting benchmarks", None, None, {})
        ctx.write_report("# Benchmarks\nNo metrics could be computed.\n")
        return

    metrics_df = pd.DataFrame(metrics_records).sort_values(["site_id", "date", "method"]).reset_index(drop=True)
    ctx.write_table("benchmark_metrics", metrics_df)

    status_df = pd.DataFrame(method_status_records)
    if not status_df.empty:
        ctx.write_table("method_status", status_df)

    # === Fairness stratification =============================================
    cov_vals = metrics_df.groupby(["site_id", "sample_id", "date"])["coverage_mean"].first().values
    cov_bins = _bin_strata(cov_vals, coverage_quantiles, ["low", "mid", "high"])
    cov_map_index = metrics_df.drop_duplicates(["site_id", "sample_id", "date"]).reset_index(drop=True)
    cov_map_index["coverage_bin"] = cov_bins
    metrics_df = metrics_df.merge(
        cov_map_index[["site_id", "sample_id", "date", "coverage_bin"]],
        on=["site_id", "sample_id", "date"], how="left"
    )

    ov_vals = metrics_df.groupby(["site_id", "sample_id", "date"])["overlap_score"].first().values
    ov_bins = _bin_strata(ov_vals, overlap_quantiles, ["low", "mid", "high"])
    ov_map_index = metrics_df.drop_duplicates(["site_id", "sample_id", "date"]).reset_index(drop=True)
    ov_map_index["overlap_bin"] = ov_bins
    metrics_df = metrics_df.merge(
        ov_map_index[["site_id", "sample_id", "date", "overlap_bin"]],
        on=["site_id", "sample_id", "date"], how="left"
    )

    def _size_bin(x: int) -> str:
        if x <= site_size_bins[0]:
            return "small"
        if x <= site_size_bins[1]:
            return "medium"
        return "large"

    metrics_df["site_size_bin"] = metrics_df["site_size"].apply(_size_bin)

    fairness_groups = ["method", "coverage_bin", "overlap_bin", "site_size_bin"]
    fairness_df = metrics_df.groupby(fairness_groups).agg(
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        r2_mean=("r2", "mean"),
        nll_mean=("nll", "mean"),
        jsd_theta_mean=("jsd_theta", "mean"),
        jsd_mu_mean=("jsd_mu", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        peak_mem_mb_mean=("peak_mem_mb", "mean"),
        count=("rmse", "size")
    ).reset_index()
    ctx.write_table("fairness_stratification", fairness_df)

    # === Figures ==============================================================
    import matplotlib.pyplot as plt
    set_matplotlib_style()

    # Panel figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax0, ax1, ax2, ax3 = axes.flatten()

    # 1) RMSE grouped by method x coverage_bin
    rmse_agg = metrics_df.groupby(["method", "coverage_bin"]).agg(rmse_mean=("rmse", "mean")).reset_index()
    methods_present = sorted(metrics_df["method"].unique().tolist())
    cov_levels = ["low", "mid", "high"]
    width = 0.8 / max(len(cov_levels), 1)
    x = np.arange(len(methods_present))
    for i, cov in enumerate(cov_levels):
        vals = [rmse_agg[(rmse_agg["method"] == m) & (rmse_agg["coverage_bin"] == cov)]["rmse_mean"].values
                for m in methods_present]
        vals = [float(v[0]) if len(v) > 0 else np.nan for v in vals]
        ax0.bar(x + (i - (len(cov_levels) - 1) / 2) * width, vals, width=width, label=f"coverage={cov}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(methods_present, rotation=0)
    ax0.set_ylabel("RMSE (AF)")
    ax0.set_title("RMSE by method and coverage bin")
    place_legend_below(ax0, ncol=len(cov_levels))

    # 2) RMSE vs NLL across samples
    Npts = int(bench_cfg.get("bland_altman_points", 2000))
    rng = np.random.default_rng(seed)
    for m in methods_present:
        rows = metrics_df[metrics_df["method"] == m]
        if rows.empty:
            continue
        idx = rng.choice(rows.index.values, size=min(Npts, len(rows)), replace=False)
        ax1.scatter(rows.loc[idx, "rmse"], rows.loc[idx, "nll"], s=10, alpha=0.6, label=m)
    ax1.set_xlabel("RMSE (AF)")
    ax1.set_ylabel("NLL (Beta–Binomial)")
    ax1.set_title("RMSE vs NLL across samples")
    place_legend_below(ax1, ncol=max(1, len(methods_present)))

    # 3) Predicted vs Observed AF for a representative site
    site_counts = metrics_df.groupby("site_id")["sample_id"].nunique().sort_values(ascending=False)
    if not site_counts.empty:
        rep_site = site_counts.index[0]
        msel = metrics_df[(metrics_df["site_id"] == rep_site)]
        if not msel.empty:
            srow = msel.iloc[0]
            sid = srow["sample_id"]; date = srow["date"]
            y_counts = _pick_col_by_sample(obs_mat, samples, rep_site, sid, date)
            n_cov = _pick_col_by_sample(cov_mat, samples, rep_site, sid, date)
            mask = n_cov > 0
            y_af = np.clip(y_counts[mask] / np.maximum(n_cov[mask], 1.0), 0.0, 1.0)
            S_sub = S[mask, :]
            for m in methods_present:
                try:
                    if m == "PGD":
                        theta, _ = pgd_nnls(y_af=y_af, S=S_sub, w=np.ones_like(y_af),
                                            lam=lambda_reg, tol=1e-5, max_iter=200)
                    elif m == "GLM":
                        theta, _ = irls_identity_binomial(y_af=y_af, n_cov=n_cov[mask], S=S_sub,
                                                          lam=irls_lambda, tol=1e-5, max_iter=50)
                    elif m == "NMF":
                        Theta, _ = nmf_fixed_dictionary(y_af_mat=y_af[:, None], S=S_sub,
                                                        W_mat=np.ones((len(y_af), 1)),
                                                        lam=nmf_lambda, max_iter=100, tol=1e-5)
                        theta = Theta[:, 0]
                    elif m == "FREYJA":
                        theta, _ = fista_nonneg_lasso(y_af=y_af, S=S_sub, w=np.ones_like(y_af),
                                                      l1=freyja_l1, l2=freyja_l2,
                                                      max_iter=300, tol=1e-6)
                    elif m == "VAQUERO":
                        theta, _ = omp_ridge_simplex(y_af=y_af, S=S_sub, w=np.ones_like(y_af),
                                                     max_k=min(10, S_sub.shape[1]),
                                                     ridge=vaquero_ridge, tol=1e-6)
                    else:
                        continue
                    mu_pred = np.clip(S_sub @ theta, _EPS, 1 - _EPS)
                    idx = rng.choice(np.arange(len(y_af)), size=min(500, len(y_af)), replace=False)
                    ax2.scatter(y_af[idx], mu_pred[idx], s=12, alpha=0.6, label=m)
                except Exception:
                    continue
            ax2.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.5)
            ax2.set_xlabel("Observed AF")
            ax2.set_ylabel("Predicted AF")
            ax2.set_title(f"Predicted vs Observed AF (site {rep_site})")
            place_legend_below(ax2, ncol=max(1, len(methods_present)))

    # 4) Runtime vs site size
    runtime_agg = metrics_df.groupby(["method", "site_id"]).agg(runtime_s_mean=("runtime_s", "mean"),
                                                                site_size=("site_size", "first")).reset_index()
    for m in methods_present:
        dfm = runtime_agg[runtime_agg["method"] == m]
        if dfm.empty: continue
        ax3.plot(dfm["site_size"], dfm["runtime_s_mean"], marker="o", linestyle="-", label=m)
    ax3.set_xlabel("Site size (#samples)")
    ax3.set_ylabel("Mean runtime per sample (s)")
    ax3.set_title("Runtime scaling by site size")
    place_legend_below(ax3, ncol=max(1, len(methods_present)))

    fig.tight_layout()
    ctx.write_figure("benchmark_panels", fig)

    # JSD(θ) boxplot
    fig2, axb = plt.subplots(figsize=(8, 4))
    jsd_data = [metrics_df[metrics_df["method"] == m]["jsd_theta"].dropna().values for m in methods_present]
    axb.boxplot(jsd_data, labels=methods_present, showfliers=False)
    axb.set_ylabel("JSD(θ) vs baseline")
    axb.set_title("Theta divergence from baseline")
    place_legend_below(axb, ncol=1)
    fig2.tight_layout()
    ctx.write_figure("jsd_boxplot", fig2)

    # === Report & Overall summary ============================================
    lines = []
    lines.append("# Benchmarks\n")
    lines.append(f"- Methods requested: {', '.join(methods)}\n")

    overall = metrics_df.groupby("method").agg(
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        nll_mean=("nll", "mean"),
        jsd_theta_mean=("jsd_theta", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        peak_mem_mb_mean=("peak_mem_mb", "mean"),
        count=("rmse", "size"),
    ).reset_index().sort_values("rmse_mean")
    ctx.write_table("benchmark_overall_summary", overall)

    lines.append("\n## Overall summary (top 5 by RMSE)\n")
    for _, r in overall.head(5).iterrows():
        lines.append(
            f"* {r['method']}: RMSE={r['rmse_mean']:.4f}, R²={r['r2_mean']:.3f}, "
            f"NLL={r['nll_mean']:.2f}, JSDθ={r['jsd_theta_mean']:.4f}, "
            f"runtime={r['runtime_s_mean']:.4f}s, peak_mem={r['peak_mem_mb_mean']:.2f}MB "
            f"(n={int(r['count'])})\n"
        )
    lines.append("\n## Fairness stratification\n")
    lines.append("Metrics stratified by coverage (low/mid/high), signature overlap (low/mid/high), and site size (small/medium/large).\n")
    ctx.write_report("\n".join(lines))

    _log(ctx, "INFO", "Benchmarks completed", None, None,
         {"methods": methods, "n_samples": len(samples)})
