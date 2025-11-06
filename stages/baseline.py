#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline evidence stage (multi-family, paper-ready CSVs).

Families supported:
- binomial (Beta prior)          -> uses (count, coverage)
- bernoulli (Beta prior)         -> uses a 0/1 column
- poisson (Gamma prior, RATE β)  -> uses integer count column
- gaussian (Normal-Inverse-Gamma)-> uses continuous column

Usage via pipeline:
  python scripts/run_pipeline.py --config configs/default.yaml --stages baseline --seed 1234

Or standalone:
  python -m stages.baseline --config configs/default.yaml

Config (top-level under key 'baseline'):
  baseline:
    family: all                   # or a single family; or use 'families: [binomial, poisson, gaussian]'
    data: results/preprocessing/tables/feature_store_snv.csv
    # for binomial:
    count_column: count
    size_column: coverage
    drop_constant: true           # evidence without ∑ log C(n_i, y_i) (raw also saved)
    grid_check: true
    grid_points: 4000
    label: default
    baselines:
      binomial:
        - {name: jeffreys, alpha: 0.5, beta: 0.5}
        - {name: uniform,  alpha: 1.0, beta: 1.0}
        - {name: beta22,   alpha: 2.0, beta: 2.0}
      poisson:
        - {name: gamma_1_1_rate, alpha: 1.0, beta: 1.0}
        - {name: gamma_2_1_rate, alpha: 2.0, beta: 1.0}
        - {name: gamma_3_1_rate, alpha: 3.0, beta: 1.0}
      gaussian:
        - {name: weak_auto, mu0: auto_mean, kappa0: 1.0e-3, alpha0: 2.0, beta0: auto_scale}
        - {name: mild_auto, mu0: auto_mean, kappa0: 1.0e-1, alpha0: 2.0, beta0: auto_scale}
      bernoulli:
        - {name: jeffreys, alpha: 0.5, beta: 0.5}
        - {name: uniform,  alpha: 1.0, beta: 1.0}
        - {name: beta22,   alpha: 2.0, beta: 2.0}
"""
from __future__ import annotations

import argparse, glob, json, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgamma as invgamma_dist
from scipy.stats import norm

# ------------------------ logging ctx (project-compatible) ------------------------
@dataclass
class SimpleCtx:
    def log(self, level: str = "INFO", message: str = "", context: Optional[dict] = None,
            stage: str = "baseline", site_id: Optional[str] = None, lineage: Optional[str] = None) -> None:
        rec = {
            "time": pd.Timestamp.utcnow().isoformat(),
            "level": level, "stage": stage,
            "site_id": site_id, "lineage": lineage,
            "message": message, "context": context or {},
        }
        print(json.dumps(rec))

# ------------------------ small utils ------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_csv_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _is_binary(x: np.ndarray) -> bool:
    if x.size == 0: return False
    return np.all(np.isin(x, [0,1])) and np.all(np.isfinite(x))

def _is_nonneg_int(x: np.ndarray) -> bool:
    if x.size == 0: return False
    return np.all((x >= 0) & np.isfinite(x) & (np.floor(x) == x))

# ------------------------ column detection ------------------------
def _pick_column(df: pd.DataFrame, prefer: Optional[List[str]] = None, requested: Optional[str] = None, ctx: Optional[SimpleCtx]=None) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    if requested:
        hit = lower.get(requested.lower())
        if hit: return hit

    prefer = prefer or []
    for c in prefer:
        hit = lower.get(c.lower())
        if hit: return hit

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        if ctx and requested and requested.lower() not in (x.lower() for x in cols):
            ctx.log("WARN", "Requested column missing; using first numeric", {"requested": requested, "chosen": numeric[0]}, "baseline")
        return numeric[0]
    raise ValueError("No numeric column found for baseline evidence.")

def _pick_pair_columns(df: pd.DataFrame, requested_count: Optional[str], requested_size: Optional[str], ctx: Optional[SimpleCtx]) -> Tuple[str, str]:
    lower = {c.lower(): c for c in df.columns}

    def hit(name_list: List[str]) -> Optional[str]:
        for nm in name_list:
            if nm and lower.get(nm.lower()):
                return lower[nm.lower()]
        return None

    count_syn = [requested_count, "count", "alt_count", "y", "k", "success", "altcount"]
    size_syn  = [requested_size,  "coverage", "n", "size", "total", "denom", "trials"]

    c = hit([x for x in count_syn if x])
    n = hit([x for x in size_syn if x])

    if c is None or n is None:
        raise KeyError("Need count & size columns for binomial baseline (e.g., 'count' and 'coverage').")

    # gentle warnings if synonym chosen
    if ctx:
        if requested_count and lower.get(requested_count.lower()) != c:
            ctx.log("WARN", "Count column synonym used", {"requested": requested_count, "chosen": c}, "baseline")
        if requested_size and lower.get(requested_size.lower()) != n:
            ctx.log("WARN", "Size column synonym used", {"requested": requested_size, "chosen": n}, "baseline")
    return c, n

# ------------------------ autodetect input ------------------------
def _find_feature_store(repo_root: Path) -> Optional[str]:
    candidates = [repo_root / "results" / "preprocessing" / "tables" / "feature_store_snv.csv"]
    candidates += [Path(p) for p in glob.glob(str(repo_root / "results" / "preprocessing" / "tables" / "feature_store_snv*.csv"))]
    for p in candidates:
        if p.exists(): return str(p)
    return None

def _autodetect_data(ctx: SimpleCtx, bl_cfg: Dict, repo_root: Path) -> str:
    data = str(bl_cfg.get("data", "") or "").strip()
    if data and Path(data).exists():
        ctx.log("INFO", "Using explicit baseline input", {"path": data}, "baseline")
        return data
    fs = _find_feature_store(repo_root)
    if fs:
        ctx.log("INFO", "Auto-detected feature store", {"path": fs}, "baseline")
        return fs
    jl = repo_root / "data" / "jahn_like.csv"
    if jl.exists():
        ctx.log("INFO", "Falling back to jahn_like.csv", {"path": str(jl)}, "baseline")
        return str(jl)
    raise FileNotFoundError("SNV table not found; expected results/preprocessing/tables/feature_store_snv*.csv or data/jahn_like.csv")

# ------------------------ conjugate math ------------------------
# Bernoulli/Binomial (Beta prior)
def _bb_post_from_seq_successes(n_succ: float, n_fail: float, alpha: float, beta: float) -> Tuple[float,float]:
    return alpha + n_succ, beta + n_fail

def _bernoulli_beta_evidence(x: np.ndarray, alpha: float, beta: float, include_binom_coeff: bool=False) -> float:
    n = int(x.size); s = float(np.sum(x))
    log_ev = betaln(alpha + s, beta + (n - s)) - betaln(alpha, beta)
    if include_binom_coeff:
        # sequence of Bernoulli trials; add ∑ log C(1, x_i) == 0 (kept for symmetry)
        pass
    return float(log_ev)

def _binom_logcomb(n, k):
    n = np.asarray(n, float); k = np.asarray(k, float)
    return gammaln(n+1.0) - gammaln(k+1.0) - gammaln(n-k+1.0)

def _binomial_beta_post(y: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> Tuple[float, float]:
    s = float(np.sum(y)); t = float(np.sum(n))
    return alpha + s, beta + (t - s)

def _binomial_beta_evidence(y: np.ndarray, n: np.ndarray, alpha: float, beta: float, include_comb: bool=True) -> float:
    s = float(np.sum(y)); t = float(np.sum(n))
    log_ev = betaln(alpha + s, beta + (t - s)) - betaln(alpha, beta)
    if include_comb:
        log_ev += float(np.sum(_binom_logcomb(n, y)))
    return float(log_ev)

# Poisson (Gamma prior, RATE β)
def _poisson_gamma_post(x: np.ndarray, alpha: float, beta: float) -> Tuple[float, float]:
    n = int(x.size); s = float(np.sum(x))
    return alpha + s, beta + n

def _poisson_gamma_evidence(x: np.ndarray, alpha: float, beta: float) -> float:
    n = int(x.size); s = float(np.sum(x))
    return float(-np.sum(gammaln(x + 1.0)) + alpha*np.log(beta) - (alpha + s)*np.log(beta + n) + gammaln(alpha + s) - gammaln(alpha))

# Gaussian (Normal-Inverse-Gamma)
def _gaussian_nig_evidence(x: np.ndarray, mu0: float, kappa0: float, alpha0: float, beta0: float) -> float:
    n = int(x.size); xbar = float(np.mean(x))
    S = float(np.sum((x - xbar)**2))
    kn = kappa0 + n; an = alpha0 + 0.5*n
    bn = beta0 + 0.5*S + (kappa0*n*(xbar - mu0)**2)/(2.0*kn)
    return float((gammaln(an)-gammaln(alpha0) + alpha0*np.log(beta0) - an*np.log(bn) + 0.5*(np.log(kappa0)-np.log(kn)) - 0.5*n*np.log(2*np.pi)))

def _gaussian_nig_post(x: np.ndarray, mu0: float, kappa0: float, alpha0: float, beta0: float) -> Dict[str, float]:
    n = int(x.size); xbar = float(np.mean(x))
    S = float(np.sum((x - xbar)**2))
    kn = kappa0 + n; an = alpha0 + 0.5*n
    bn = beta0 + 0.5*S + (kappa0*n*(xbar - mu0)**2)/(2.0*kn)
    mu_n = (kappa0*mu0 + n*xbar) / kn
    return dict(mu_n=float(mu_n), kappa_n=float(kn), alpha_n=float(an), beta_n=float(bn))

# MAP helpers
def _beta_map(a: float, b: float) -> float:
    return (a-1)/(a+b-2) if (a>1 and b>1) else a/(a+b)

def _gamma_map(a: float, b: float) -> float:
    return (a-1)/b if a>1 else a/b

def _nig_map(mu_n: float, kappa_n: float, alpha_n: float, beta_n: float) -> Tuple[float, float]:
    mu_map = mu_n
    sig2_map = beta_n/(alpha_n+1.0) if alpha_n>1 else beta_n/max(alpha_n,1e-9)
    return float(mu_map), float(sig2_map)

# Grid checks
def _grid_integral_1d(lp: np.ndarray, grid: np.ndarray) -> float:
    d = np.diff(grid); w = np.zeros_like(grid)
    w[0]=0.5*d[0]; w[-1]=0.5*d[-1]
    if len(grid)>2: w[1:-1]=0.5*(d[:-1]+d[1:])
    return float(logsumexp(lp + np.log(w)))

def _grid_check_bernoulli(x: np.ndarray, a: float, b: float, n_grid: int=2000) -> Dict[str, float]:
    eps=1e-9; g=np.linspace(eps,1-eps,n_grid)
    lp=(a-1)*np.log(g) + (b-1)*np.log(1-g) - betaln(a,b)
    s=float(np.sum(x)); n=int(x.size)
    ll=s*np.log(g)+(n-s)*np.log(1-g)
    lg=_grid_integral_1d(lp+ll,g); le=_bernoulli_beta_evidence(x,a,b,False)
    return {"log_ev_grid": float(lg), "log_ev_exact": float(le), "abs_err": float(abs(lg-le))}

def _grid_check_binomial(y: np.ndarray, n: np.ndarray, a: float, b: float, n_grid: int=2000) -> Dict[str,float]:
    eps=1e-9; g=np.linspace(eps,1-eps,n_grid)
    lp=(a-1)*np.log(g) + (b-1)*np.log(1-g) - betaln(a,b)
    s=float(np.sum(y)); t=float(np.sum(n))
    ll=s*np.log(g)+(t-s)*np.log(1-g)  # comb. constant excluded
    lg=_grid_integral_1d(lp+ll,g); le=_binomial_beta_evidence(y,n,a,b,include_comb=False)
    return {"log_ev_grid": float(lg), "log_ev_exact": float(le), "abs_err": float(abs(lg-le))}

def _grid_check_poisson(x: np.ndarray, a: float, b: float, n_grid: int=4000, q_hi: float=0.9999) -> Dict[str, float]:
    ap,bp=_poisson_gamma_post(x,a,b)
    upper = float(max(gamma_dist.ppf(q_hi, a=ap, scale=1.0/bp), 10.0))
    g=np.linspace(1e-12, upper, n_grid)
    lp=(a-1)*np.log(g) - b*g + a*np.log(b) - gammaln(a)
    n=int(x.size); s=float(np.sum(x))
    ll=-n*g + s*np.log(g) - np.sum(gammaln(x+1.0))
    lg=_grid_integral_1d(lp+ll,g); le=_poisson_gamma_evidence(x,a,b)
    return {"log_ev_grid": float(lg), "log_ev_exact": float(le), "abs_err": float(abs(lg-le)), "upper": upper}

# Defaults
DEFAULT_BASELINES = {
    "bernoulli": [
        {"name":"jeffreys","alpha":0.5,"beta":0.5},
        {"name":"uniform","alpha":1.0,"beta":1.0},
        {"name":"beta22","alpha":2.0,"beta":2.0},
    ],
    "binomial": [
        {"name":"jeffreys","alpha":0.5,"beta":0.5},
        {"name":"uniform","alpha":1.0,"beta":1.0},
        {"name":"beta22","alpha":2.0,"beta":2.0},
    ],
    "poisson": [
        {"name":"gamma_1_1_rate","alpha":1.0,"beta":1.0},
        {"name":"gamma_2_1_rate","alpha":2.0,"beta":1.0},
        {"name":"gamma_3_1_rate","alpha":3.0,"beta":1.0},
    ],
    "gaussian": [
        {"name":"weak_auto","mu0":"auto_mean","kappa0":1e-3,"alpha0":2.0,"beta0":"auto_scale"},
        {"name":"mild_auto","mu0":"auto_mean","kappa0":1e-1,"alpha0":2.0,"beta0":"auto_scale"},
    ],
}

def _resolve_nig_from_data(x: np.ndarray, pr: Dict[str,Any]) -> Dict[str,Any]:
    mu_hat=float(np.mean(x)) if x.size else 0.0
    s2=float(np.var(x,ddof=1)) if x.size>1 else float(np.var(x)) if x.size else 1.0
    beta_auto=float(max(1e-12, s2*max(1.0, pr.get("alpha0",2.0)-1.0)))
    return {"name":pr.get("name","auto"),
            "mu0": (mu_hat if pr.get("mu0") in ("auto","auto_mean") else float(pr["mu0"])),
            "kappa0": float(pr["kappa0"]), "alpha0": float(pr["alpha0"]),
            "beta0": (beta_auto if pr.get("beta0") in ("auto","auto_scale") else float(pr["beta0"]))}

# ------------------------ CSV flatten helpers ------------------------
def _row_flatten_common(run_id: str, family: str, prior: Dict[str, Any], log_evidence: float, extras: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "run_id": run_id,
        "family": family,
        "prior_name": prior.get("name",""),
        "log_evidence": float(log_evidence),
    }
    # expand prior params into columns
    for k,v in prior.items():
        if k=="name": continue
        base[f"prior_{k}"] = v
    base.update(extras)
    return base

def _write_tables(paths: Dict[str, Path], stamp: str, all_rows: List[Dict[str,Any]], per_family_rows: Dict[str, List[Dict[str,Any]]]) -> None:
    tables_dir = paths["tables"]
    _ensure_dir(tables_dir)
    if all_rows:
        df_all = pd.DataFrame(all_rows).sort_values(["family","log_evidence"], ascending=[True,False])
        df_all.to_csv(tables_dir / f"baselines_summary_{stamp}.csv", index=False)
        df_all.to_csv(tables_dir / "baselines_summary_latest.csv", index=False)
        # pack minimal data stats summary
        cols = [c for c in df_all.columns if c.startswith("data_")]
        if cols:
            df_all[["run_id"] + cols].drop_duplicates().to_csv(tables_dir / f"data_stats_{stamp}.csv", index=False)
    # per-family model posteriors & bayes factors
    for fam, rows in per_family_rows.items():
        if not rows: continue
        df = pd.DataFrame(rows).sort_values("log_evidence", ascending=False)
        df.to_csv(tables_dir / f"baselines_family_{fam}_{stamp}.csv", index=False)
        # posterior model probs under equal priors
        le = df["log_evidence"].to_numpy(float)
        w = np.exp(le - logsumexp(le))
        df_post = pd.DataFrame({"model_id": df["model_id"], "family": fam, "posterior_model_prob": w})
        df_post.to_csv(tables_dir / f"model_posteriors_{fam}_{stamp}.csv", index=False)
        # bayes factors within family
        bf = []
        ids = df["model_id"].tolist()
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                bf.append({"family":fam,"better_model":ids[i],"worse_model":ids[j],"bayes_factor":float(np.exp(le[i]-le[j]))})
        if bf:
            pd.DataFrame(bf).to_csv(tables_dir / f"bayes_factors_{fam}_{stamp}.csv", index=False)

# ------------------------ Stage main ------------------------
def run_baseline(cfg: Dict, ctx: SimpleCtx) -> Dict:
    bl = cfg.get("baseline", cfg)
    repo_root = Path(".").resolve()

    # Layout
    results_root = bl.get("save_root") or cfg.get("io", {}).get("results_dir") or "results"
    out_base = Path(results_root) / "baselines"
    paths = {
        "root": out_base,
        "runs": out_base / "runs",
        "tables": out_base / "tables",
        "metrics": out_base / "metrics",
        "logs": out_base / "logs",
        "figures": out_base / "figures",
    }
    for p in paths.values(): _ensure_dir(p)

    # Data
    data_path = _autodetect_data(ctx, bl, repo_root)
    df = _read_csv_any(data_path)

    # Families to run
    family_cfg = bl.get("family", "all")
    families = bl.get("families", None)
    if families and isinstance(families, list) and families:
        fam_list = [str(f).lower().strip() for f in families]
    else:
        fam = str(family_cfg).lower().strip()
        fam_list = ["binomial","poisson","gaussian","bernoulli"] if fam in ("all","*","multi") else [fam]

    # Common options
    grid_check   = bool(bl.get("grid_check", False))
    grid_points  = int(bl.get("grid_points", 2000))
    label        = str(bl.get("label", "")).strip()
    drop_constant= bool(bl.get("drop_constant", True))  # Binomial: drop ∑log C(n,y) in primary evidence column

    count_col_cfg = bl.get("count_column")
    size_col_cfg  = bl.get("size_column")
    single_col_cfg= bl.get("column")

    stamp = time.strftime("%Y%m%d-%H%M%S")

    all_rows: List[Dict[str,Any]] = []
    per_family_rows: Dict[str, List[Dict[str,Any]]] = {}

    # Precompute dataset stats useful for papers
    data_stats_common = {
        "data_path": str(Path(data_path).resolve()),
        "data_n_rows": int(len(df)),
    }

    for family in fam_list:
        # Resolve priors for this family
        BL = bl.get("baselines", DEFAULT_BASELINES)
        if family not in BL and family not in DEFAULT_BASELINES:
            ctx.log("WARN", f"Unknown family '{family}' skipped", {"family": family}, "baseline"); continue
        if family == "gaussian":
            # We will resolve NIG from the chosen column after we pick it
            priors_raw = BL.get("gaussian", DEFAULT_BASELINES["gaussian"])
        else:
            priors = BL.get(family, DEFAULT_BASELINES[family])

        # Prepare data vectors
        rows_this_family: List[Dict[str,Any]] = []
        run_id = f"{stamp}_{family}" + (f"_{label}" if label else "")
        run_dir = paths["runs"] / run_id
        _ensure_dir(run_dir)

        # Meta file
        meta = {
            "timestamp": stamp, "family": family, "label": label,
            "python": sys.version, "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
            "data_file": str(Path(data_path).resolve()),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        # README
        (run_dir / "README.md").write_text(f"# Baseline evidence: {family}\n", encoding="utf-8")

        # Choose columns & vectors per family
        if family == "binomial":
            c_col, n_col = _pick_pair_columns(df, count_col_cfg, size_col_cfg, ctx)
            y = pd.to_numeric(df[c_col], errors="coerce").fillna(0).astype(int).to_numpy()
            n = pd.to_numeric(df[n_col], errors="coerce").fillna(0).astype(int).to_numpy()
            y = np.clip(y, 0, n)
            data_stats = dict(data_stats_common)
            data_stats.update({
                "data_count_col": c_col, "data_size_col": n_col,
                "data_success_sum": float(np.sum(y)), "data_trials_sum": float(np.sum(n)),
            })
            priors = BL.get("binomial", DEFAULT_BASELINES["binomial"])

        elif family == "poisson":
            x_col = _pick_column(df, prefer=[single_col_cfg or "", "count","alt_count","y"], requested=single_col_cfg, ctx=ctx)
            x = pd.to_numeric(df[x_col], errors="coerce").fillna(0).astype(int).to_numpy()
            if not _is_nonneg_int(x):
                ctx.log("WARN", "Selected column is not non-neg integer; coercing to int for Poisson", {"column": x_col}, "baseline")
                x = np.clip(np.round(x), 0, None).astype(int)
            data_stats = dict(data_stats_common)
            data_stats.update({
                "data_column": x_col, "data_sum_x": float(np.sum(x)), "data_n_obs": int(x.size)
            })
            priors = BL.get("poisson", DEFAULT_BASELINES["poisson"])

        elif family == "bernoulli":
            b_col = _pick_column(df, prefer=[single_col_cfg or "", "is_present","present","binary","y"], requested=single_col_cfg, ctx=ctx)
            x = pd.to_numeric(df[b_col], errors="coerce").fillna(0).to_numpy()
            x = np.where(x>0, 1.0, 0.0)
            data_stats = dict(data_stats_common)
            data_stats.update({
                "data_column": b_col, "data_s": float(np.sum(x)), "data_n_obs": int(x.size)
            })
            priors = BL.get("bernoulli", DEFAULT_BASELINES["bernoulli"])

        elif family == "gaussian":
            g_col = _pick_column(df, prefer=[single_col_cfg or "", "value","growth_rate","x"], requested=single_col_cfg, ctx=ctx)
            g = pd.to_numeric(df[g_col], errors="coerce").dropna().to_numpy(float)
            data_stats = dict(data_stats_common)
            data_stats.update({
                "data_column": g_col, "data_mean": float(np.mean(g) if g.size else np.nan),
                "data_var": float(np.var(g, ddof=1) if g.size>1 else 0.0), "data_n_obs": int(g.size)
            })
            priors = []
            for pr in (BL.get("gaussian", DEFAULT_BASELINES["gaussian"])):
                priors.append(_resolve_nig_from_data(g, pr))
        else:
            ctx.log("WARN", f"Skipping unsupported family: {family}", {"family": family}, "baseline")
            continue

        # Save resolved_config for this family
        try:
            import yaml
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump({"family": family, "label": label, "priors": priors, "data": data_stats}, sort_keys=False),
                encoding="utf-8"
            )
        except Exception:
            (run_dir / "resolved_config.json").write_text(json.dumps({"family": family, "label": label, "priors": priors, "data": data_stats}, indent=2), encoding="utf-8")

        # Run candidates
        rows = []
        jsonl_path = run_dir / "run_summary.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for i, pr in enumerate(priors):
                model_id = f"{family}_{i:02d}_{pr.get('name','prior')}"
                model_dir = run_dir / model_id
                _ensure_dir(model_dir)

                if family == "binomial":
                    a,b = float(pr["alpha"]), float(pr["beta"])
                    ap,bp = _binomial_beta_post(y,n,a,b)
                    log_ev_raw = _binomial_beta_evidence(y,n,a,b,include_comb=True)
                    log_ev_nc  = _binomial_beta_evidence(y,n,a,b,include_comb=False)
                    theta_mean = ap/(ap+bp); theta_map = _beta_map(ap,bp)
                    diag={}
                    if grid_check:
                        diag=_grid_check_binomial(y,n,a,b,n_grid=grid_points)
                        (model_dir/"grid_check.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    (model_dir/"posterior_params.json").write_text(json.dumps({"alpha":ap,"beta":bp,"mean":theta_mean,"map":theta_map}, indent=2), encoding="utf-8")
                    (model_dir/"evidence.json").write_text(json.dumps({"log_evidence": float(log_ev_nc if bl.get("drop_constant", True) else log_ev_raw),
                                                                       "log_evidence_raw": float(log_ev_raw),
                                                                       "log_evidence_noconst": float(log_ev_nc),
                                                                       "dropped_constant": bool(bl.get("drop_constant", True))}, indent=2), encoding="utf-8")
                    log_ev = log_ev_nc if drop_constant else log_ev_raw
                    extras = {
                        "model_id": model_id,
                        "prior_type": "beta",
                        "post_alpha": ap, "post_beta": bp,
                        "post_mean": theta_mean, "post_map": theta_map,
                        "data_count_col": data_stats.get("data_count_col"),
                        "data_size_col": data_stats.get("data_size_col"),
                        "data_success_sum": data_stats.get("data_success_sum"),
                        "data_trials_sum": data_stats.get("data_trials_sum"),
                    }
                    row = _row_flatten_common(run_id, family, pr, log_ev, extras)

                elif family == "poisson":
                    a,b = float(pr["alpha"]), float(pr["beta"])
                    ap,bp = _poisson_gamma_post(x,a,b)
                    log_ev = _poisson_gamma_evidence(x,a,b)
                    lam_mean = ap/bp; lam_map = _gamma_map(ap,bp)
                    diag={}
                    if grid_check:
                        diag=_grid_check_poisson(x,a,b,n_grid=grid_points)
                        (model_dir/"grid_check.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    (model_dir/"posterior_params.json").write_text(json.dumps({"alpha":ap,"beta":bp,"mean":lam_mean,"map":lam_map}, indent=2), encoding="utf-8")
                    (model_dir/"evidence.json").write_text(json.dumps({"log_evidence": float(log_ev)}, indent=2), encoding="utf-8")
                    extras = {
                        "model_id": model_id,
                        "prior_type": "gamma_rate",
                        "post_alpha": ap, "post_beta": bp,
                        "post_mean": lam_mean, "post_map": lam_map,
                        "data_column": data_stats.get("data_column"),
                        "data_sum_x": data_stats.get("data_sum_x"),
                        "data_n_obs": data_stats.get("data_n_obs"),
                    }
                    row = _row_flatten_common(run_id, family, pr, log_ev, extras)

                elif family == "bernoulli":
                    a,b = float(pr["alpha"]), float(pr["beta"])
                    ap,bp = _bb_post_from_seq_successes(float(np.sum(x)), float(x.size - np.sum(x)), a, b)
                    log_ev = _bernoulli_beta_evidence(x,a,b,include_binom_coeff=False)
                    theta_mean = ap/(ap+bp); theta_map = _beta_map(ap,bp)
                    diag={}
                    if grid_check:
                        diag=_grid_check_bernoulli(x,a,b,n_grid=grid_points)
                        (model_dir/"grid_check.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
                    (model_dir/"posterior_params.json").write_text(json.dumps({"alpha":ap,"beta":bp,"mean":theta_mean,"map":theta_map}, indent=2), encoding="utf-8")
                    (model_dir/"evidence.json").write_text(json.dumps({"log_evidence": float(log_ev)}, indent=2), encoding="utf-8")
                    extras = {
                        "model_id": model_id,
                        "prior_type": "beta",
                        "post_alpha": ap, "post_beta": bp,
                        "post_mean": theta_mean, "post_map": theta_map,
                        "data_column": data_stats.get("data_column"),
                        "data_s": data_stats.get("data_s"),
                        "data_n_obs": data_stats.get("data_n_obs"),
                    }
                    row = _row_flatten_common(run_id, family, pr, log_ev, extras)

                else:  # gaussian
                    mu0,k0 = float(pr["mu0"]), float(pr["kappa0"])
                    a0,b0  = float(pr["alpha0"]), float(pr["beta0"])
                    log_ev = _gaussian_nig_evidence(g, mu0,k0,a0,b0)
                    post   = _gaussian_nig_post(g, mu0,k0,a0,b0)
                    mu_map,sig2_map = _nig_map(post["mu_n"], post["kappa_n"], post["alpha_n"], post["beta_n"])
                    (model_dir/"posterior_params.json").write_text(json.dumps({**post,"map_mu":mu_map,"map_sigma2":sig2_map}, indent=2), encoding="utf-8")
                    (model_dir/"evidence.json").write_text(json.dumps({"log_evidence": float(log_ev)}, indent=2), encoding="utf-8")
                    extras = {
                        "model_id": model_id,
                        "prior_type": "nig",
                        "post_mu_n": post["mu_n"], "post_kappa_n": post["kappa_n"],
                        "post_alpha_n": post["alpha_n"], "post_beta_n": post["beta_n"],
                        "post_map_mu": mu_map, "post_map_sigma2": sig2_map,
                        "data_column": data_stats.get("data_column"),
                        "data_mean": data_stats.get("data_mean"),
                        "data_var": data_stats.get("data_var"),
                        "data_n_obs": data_stats.get("data_n_obs"),
                    }
                    row = _row_flatten_common(run_id, family, pr, log_ev, extras)

                rows.append(row)
                jf.write(json.dumps({"model_id": model_id, **row}) + "\n")

        # Save per-family run CSV
        df_family = pd.DataFrame(rows).sort_values("log_evidence", ascending=False)
        df_family.to_csv(run_dir / "run_summary.csv", index=False)

        # Remember for combined outputs
        rows_this_family = []
        for r in rows:
            # add data-level stats and data path for clarity
            r2 = dict(r)
            r2["data_path"] = data_stats_common["data_path"]
            r2.setdefault("data_n_rows", data_stats_common["data_n_rows"])
            rows_this_family.append(r2)
        per_family_rows[family] = rows_this_family
        all_rows.extend(rows_this_family)

        # Metrics
        best = df_family.iloc[0].to_dict() if len(df_family) else {}
        metrics = {
            "run_id": run_id, "family": family,
            "best_model_id": best.get("model_id"), "best_prior_name": best.get("prior_name"),
            "best_log_evidence": best.get("log_evidence"), "n_candidates": int(len(df_family))
        }
        (paths["metrics"] / f"baseline_{run_id}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Combined paper-ready tables
    _write_tables(paths, stamp, all_rows, per_family_rows)

    ctx.log("INFO", "Baseline stage done", {"runs_root": str(paths["runs"]), "tables_root": str(paths["tables"])}, "baseline")
    return {
        "tables": ["baselines_summary_latest"],
        "runs_dir": str(paths["runs"]),
        "tables_dir": str(paths["tables"]),
    }

# ------------------------ CLI passthrough ------------------------
def _load_cfg(path: Optional[str]) -> Dict:
    if not path: return {}
    txt = open(path,"r",encoding="utf-8").read()
    try:
        import yaml; return yaml.safe_load(txt)
    except Exception:
        try: return json.loads(txt)
        except Exception: raise RuntimeError("Provide YAML/JSON config or install PyYAML.")

def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline evidence (multi-family)")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config) if args.config else {}
    ctx = SimpleCtx()
    out = run_baseline(cfg, ctx)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
