import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import random
from collections import defaultdict

from scipy.stats import kstest, norm, betabinom

# ---------- plotting fallbacks ----------
try:
    from utils.plotting import set_matplotlib_style, place_legend_below
except Exception:  # Fallbacks if utilities are not available
    def set_matplotlib_style() -> None:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.grid": True,
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

    def place_legend_below(ax: plt.Axes, ncols: int = 3, labels: Optional[List[str]] = None) -> None:
        handles, labels_h = ax.get_legend_handles_labels()
        if labels is not None:
            labels_h = labels
        if len(labels_h) == 0:
            return
        ax.legend(
            handles,
            labels_h,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(ncols, len(labels_h)),
            frameon=False,
        )

# ---------- seeding ----------
try:
    from utils.seeds import set_global_seeds as _set_global_seeds
except Exception:
    def _set_global_seeds(seed: int) -> None:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)

# ---------- logging ----------
def _log(ctx, level: str, stage: str, message: str,
         site_id: Optional[str] = None, lineage: Optional[str] = None,
         context: Optional[Dict] = None) -> None:
    payload = {
        "time": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "stage": stage,
        "site_id": site_id,
        "lineage": lineage,
        "message": message,
        "context": context or {},
    }
    try:
        if hasattr(ctx, "log") and callable(getattr(ctx, "log")):
            ctx.log(level="INFO", message="Diagnostics payload", context=payload)
        elif hasattr(ctx, "write_log") and callable(getattr(ctx, "write_log")):
            ctx.write_log(payload)
        elif hasattr(ctx, "write_metric") and callable(getattr(ctx, "write_metric")):
            ctx.write_metric("log_fallback", pd.DataFrame([payload]))
    except Exception:
        pass

# ---------- IO helpers ----------
def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]

def _first_existing(paths: List[pathlib.Path]) -> Optional[pathlib.Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _read_any(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # try both
    csv = path.with_suffix(".csv")
    pq = path.with_suffix(".parquet")
    if csv.exists():
        return pd.read_csv(csv)
    if pq.exists():
        return pd.read_parquet(pq)
    raise FileNotFoundError(f"No CSV/Parquet found for {path}")

# ---------- data loaders ----------
def _load_observed_snv() -> pd.DataFrame:
    root = _repo_root()
    candidates = []
    prep_tables = root / "results" / "preprocessing" / "tables"
    if prep_tables.exists():
        for name in ["feature_store_snv_long", "feature_store_snv", "snv_long", "feature_store"]:
            candidates.append(_first_existing([prep_tables / f"{name}.csv",
                                              prep_tables / f"{name}.parquet"]))
    candidates.append(_first_existing([root / "data" / "jahn_like.csv"]))
    cand = [c for c in candidates if c is not None]
    if not cand:
        raise FileNotFoundError("Observed SNV data not found in results/preprocessing/tables or data/jahn_like.csv")
    df = _read_any(cand[0])
    required = {"sample_id", "site_id", "date", "mutation", "count", "coverage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Observed SNV data missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["count", "coverage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["mutation"] = df["mutation"].astype(str)
    df["site_id"] = df["site_id"].astype(str)
    g = df.groupby(["site_id", "date", "mutation"], as_index=False)[["count", "coverage"]].sum()
    g = g[g["coverage"] > 0].copy()
    g["obs_af"] = g["count"] / g["coverage"].clip(lower=1)
    return g

from pathlib import Path

def _load_signatures(root: Path | None = None) -> pd.DataFrame:
    """
    Load mutation signatures table with columns ['mutation','lineage','weight'].
    """
    root = Path(root) if root is not None else _repo_root()
    candidates = []
    prep_tables = root / "results" / "preprocessing" / "tables"
    if prep_tables.exists():
        for name in ["signatures", "signature_matrix"]:
            candidates.append(_first_existing([prep_tables / f"{name}.csv",
                                              prep_tables / f"{name}.parquet"]))
    candidates.append(_first_existing([root / "data" / "signatures.csv"]))
    cand = [c for c in candidates if c is not None]
    if not cand:
        raise FileNotFoundError(f"Signatures not found under {prep_tables} or data/signatures.csv")
    df = _read_any(cand[0]).copy()
    required = {"mutation", "lineage", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Signatures missing columns: {missing}")
    df["mutation"] = df["mutation"].astype(str)
    df["lineage"] = df["lineage"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df

def _parse_qname_to_prob(name: str) -> Optional[float]:
    """
    Convert column names like 'q05','q5','q50','p50' to a probability in [0,1].
    Returns None if unparsable.
    """
    s = name.strip().lower()
    if s.startswith("q") or s.startswith("p"):
        tail = s[1:]
        try:
            v = float(tail)
            # Interpret '05' as 0.05, '5' as 0.05 only if <=1 -> heuristics:
            if v > 1.0:  # treat as percentage
                return v / 100.0
            # if 0 < v <= 1 assume it's already probability (e.g., 0.5)
            return v
        except Exception:
            return None
    return None

def _load_smoothed_theta() -> pd.DataFrame:
    root = _repo_root()
    candidates = []
    forecast_tables = root / "results" / "forecast" / "tables"
    if forecast_tables.exists():
        candidates.append(_first_existing([forecast_tables / "forecast_smoothed_props.csv",
                                           forecast_tables / "forecast_smoothed_props.parquet"]))
    like_tables = root / "results" / "likelihood" / "tables"
    if like_tables.exists():
        for nm in ["theta_long", "deconvolved_thetas", "theta_tidy"]:
            candidates.append(_first_existing([like_tables / f"{nm}.csv",
                                               like_tables / f"{nm}.parquet"]))
    cand = [c for c in candidates if c is not None]
    if not cand:
        raise FileNotFoundError("Smoothed lineage proportions not found in forecast/likelihood tables")
    df = _read_any(cand[0]).copy()
    required = {"site_id", "date", "lineage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Theta table missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["site_id"] = df["site_id"].astype(str)
    df["lineage"] = df["lineage"].astype(str)
    theta_col = None
    for c in ["theta", "median", "mean", "value"]:
        if c in df.columns:
            theta_col = c
            break
    if theta_col is None:
        # prefer quantile nearest to 0.5
        qcols = [c for c in df.columns if c.lower().startswith(("q", "p"))]
        qpairs = []
        for c in qcols:
            q = _parse_qname_to_prob(c)
            if q is not None:
                qpairs.append((abs(q - 0.5), c))
        if qpairs:
            theta_col = sorted(qpairs, key=lambda t: t[0])[0][1]
    if theta_col is None:
        raise ValueError("Cannot identify theta value column in smoothed props")
    df = df[["site_id", "date", "lineage", theta_col]].rename(columns={theta_col: "theta"})
    df["theta"] = pd.to_numeric(df["theta"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df

def _load_priors_kappa() -> Optional[pd.DataFrame]:
    root = _repo_root()
    candidates = []
    priors_tables = root / "results" / "priors" / "tables"
    if priors_tables.exists():
        candidates.append(_first_existing([priors_tables / "priors_hyperparams.csv",
                                           priors_tables / "priors_hyperparams.parquet"]))
        candidates.append(_first_existing([priors_tables / "morita_ess.csv",
                                           priors_tables / "morita_ess.parquet"]))
    cand = [c for c in candidates if c is not None]
    if not cand:
        return None
    path = None
    for c in cand:
        if "priors_hyperparams" in str(c):
            path = c
            break
    if path is None:
        path = cand[0]
    df = _read_any(path).copy()
    if "mutation" not in df.columns:
        return None
    return df

# ---------- model projections ----------
def _compute_predicted_af(obs: pd.DataFrame, theta: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    muts = obs["mutation"].unique().tolist()
    signatures = signatures[signatures["mutation"].isin(muts)].copy()
    if signatures.empty:
        raise ValueError("No overlap between observed mutations and signatures")
    sig = signatures[["mutation", "lineage", "weight"]].copy()
    th = theta[["site_id", "date", "lineage", "theta"]].copy()
    merged = th.merge(sig, on="lineage", how="inner")
    merged["contrib"] = merged["theta"] * merged["weight"]
    pred = (
        merged.groupby(["site_id", "date", "mutation"], as_index=False)["contrib"]
        .sum()
        .rename(columns={"contrib": "pred_af"})
    )
    df = obs.merge(pred, on=["site_id", "date", "mutation"], how="left")
    df["pred_af"] = df["pred_af"].fillna(0.0).clip(0.0, 1.0)
    return df

def _extract_kappa_map(priors_df: Optional[pd.DataFrame], default_kappa: float) -> Dict[str, float]:
    kappa_map: Dict[str, float] = {}
    if priors_df is None or priors_df.empty:
        return kappa_map
    if {"alpha", "beta"}.issubset(priors_df.columns):
        tmp = priors_df[["mutation", "alpha", "beta"]].copy()
        tmp["kappa"] = pd.to_numeric(tmp["alpha"], errors="coerce").fillna(0.0) + \
                       pd.to_numeric(tmp["beta"], errors="coerce").fillna(0.0)
        for r in tmp.itertuples(index=False):
            if float(r.kappa) > 0:
                kappa_map[str(r.mutation)] = float(r.kappa)
    elif "kappa" in priors_df.columns:
        for r in priors_df[["mutation", "kappa"]].itertuples(index=False):
            if float(r.kappa) > 0:
                kappa_map[str(r.mutation)] = float(r.kappa)
    elif "ess" in priors_df.columns:
        for r in priors_df[["mutation", "ess"]].itertuples(index=False):
            if float(r.ess) > 0:
                kappa_map[str(r.mutation)] = float(r.ess)
    return kappa_map

# ---------- residuals ----------
def _beta_binomial_resid(y: np.ndarray, n: np.ndarray, p: np.ndarray,
                         kappa: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    mu = n * p
    # Var(Y) = n p (1-p) * (1 + (n-1)/(kappa+1))  with alpha=kp, beta=k(1-p)
    var = n * p * (1.0 - p) * ((n + kappa) / (1.0 + kappa))
    var = np.clip(var, eps, None)
    r = (y - mu) / np.sqrt(var)
    return r

# ---------- analytic PPC (no Monte Carlo) ----------
def _ppc_and_calibration(
    df: pd.DataFrame,
    levels: List[float],
    n_draws: int,          # kept for interface parity; not used (analytic)
    q_grid: np.ndarray,
    pit_bins: int,
    chunk_rows: int = 3000,
    seed: int = 12345,
) -> Tuple[
    Dict[str, Dict[float, Tuple[int, int]]],
    Dict[str, List[float]],
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
]:
    """
    Compute PPC diagnostics analytically under Beta–Binomial:
      - Coverage via Beta–Binomial PPF
      - PIT via Beta–Binomial CDF with randomized tie-breaking
      - Standardized residuals
      - Calibration curve: y <= q-quantile (using Beta–Binomial PPF)
      - PIT histograms per site
    """
    rng = np.random.default_rng(seed)
    coverage_counts: Dict[str, Dict[float, Tuple[int, int]]] = defaultdict(lambda: defaultdict(lambda: (0, 0)))
    pit_values: Dict[str, List[float]] = defaultdict(list)
    pit_hist: Dict[str, np.ndarray] = {}
    all_resid: List[float] = []

    calib_num = np.zeros_like(q_grid, dtype=float)
    calib_den = np.zeros_like(q_grid, dtype=float)

    site_ids = sorted(df["site_id"].unique().tolist())
    sites_aug = site_ids + ["ALL"]
    pit_edges = np.linspace(0.0, 1.0, pit_bins + 1)

    for sid in site_ids:
        sub = df[df["site_id"] == sid].copy()
        n_rows = sub.shape[0]
        if n_rows == 0:
            continue
        for start in range(0, n_rows, chunk_rows):
            sl = slice(start, min(start + chunk_rows, n_rows))
            chunk = sub.iloc[sl]
            y = chunk["count"].to_numpy(dtype=np.int64)
            n = chunk["coverage"].to_numpy(dtype=np.int64)
            p = chunk["pred_af"].to_numpy(dtype=float)
            kappa = chunk["kappa"].to_numpy(dtype=float)

            # residuals
            all_resid.append(_beta_binomial_resid(y, n, p, kappa))

            # Beta parameters
            a = np.maximum(kappa * np.clip(p, 1e-12, 1 - 1e-12), 1e-12)
            b = np.maximum(kappa * np.clip(1 - p, 1e-12, 1 - 1e-12), 1e-12)

            # ---- PIT (randomized) analytically ----
            Fy = betabinom.cdf(y, n, a, b)
            Fy_minus = betabinom.cdf(np.maximum(y - 1, 0), n, a, b)
            V = rng.random(size=y.shape[0])
            U = Fy_minus + V * np.maximum(Fy - Fy_minus, 0.0)
            pit_values[sid].extend(U.tolist())
            pit_values["ALL"].extend(U.tolist())

            # ---- Coverage at specified levels (central intervals) ----
            for L in levels:
                alpha = (1.0 - L) / 2.0
                lo = betabinom.ppf(alpha, n, a, b)
                hi = betabinom.ppf(1.0 - alpha, n, a, b)
                covered = ((y >= lo) & (y <= hi)).astype(int)
                cov_count = int(covered.sum())
                total = int(covered.shape[0])
                c_cov, c_tot = coverage_counts[sid].get(L, (0, 0))
                coverage_counts[sid][L] = (c_cov + cov_count, c_tot + total)
                c_cov, c_tot = coverage_counts["ALL"].get(L, (0, 0))
                coverage_counts["ALL"][L] = (c_cov + cov_count, c_tot + total)

            # ---- Calibration curve (empirical <= q-quantile) ----
            for iq, q in enumerate(q_grid):
                qtile = betabinom.ppf(q, n, a, b)
                emp = (y <= qtile).astype(int)
                calib_num[iq] += emp.sum()
                calib_den[iq] += emp.shape[0]

    residuals = np.concatenate(all_resid, axis=0) if len(all_resid) > 0 else np.array([], dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size > 0:
        probs = (np.arange(1, residuals.size + 1) - 0.5) / residuals.size
        qq_theory = norm.ppf(probs)
    else:
        qq_theory = np.array([], dtype=float)

    # PIT hist per site and ALL
    for sid in sites_aug:
        U = np.array(pit_values.get(sid, []), dtype=float)
        if U.size == 0:
            pit_hist[sid] = np.zeros(pit_bins, dtype=float)
        else:
            hist, _ = np.histogram(U, bins=pit_edges, density=False)
            pit_hist[sid] = hist.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        calib_empirical = np.divide(calib_num, calib_den, out=np.zeros_like(calib_num), where=calib_den > 0)

    return coverage_counts, pit_values, residuals, qq_theory, pit_hist, calib_empirical

# ---------- figures ----------
def _make_ppc_figure(
    residuals: np.ndarray,
    qq_theory: np.ndarray,
    pit_all: List[float],
    calib_nominal: np.ndarray,
    calib_empirical: np.ndarray,
    fig_title: str = "Posterior Predictive Checks",
) -> plt.Figure:
    set_matplotlib_style()
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # Panel 1: Standardized residuals histogram with N(0,1) overlay
    ax1 = fig.add_subplot(gs[0, 0])
    if residuals.size > 0:
        ax1.hist(residuals, bins=30, density=True, alpha=0.7, label="Std residuals")
        x = np.linspace(-4, 4, 200)
        ax1.plot(x, norm.pdf(x), lw=2.0, label="N(0,1)")
    ax1.set_title("Standardized residuals")
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    place_legend_below(ax1, ncols=2)

    # Panel 2: QQ plot vs N(0,1)
    ax2 = fig.add_subplot(gs[0, 1])
    if residuals.size > 0 and qq_theory.size == residuals.size:
        res_sorted = np.sort(residuals)
        ax2.scatter(qq_theory, res_sorted, s=10, alpha=0.7, label="Empirical vs Normal")
        lims = [min(qq_theory.min(), res_sorted.min()), max(qq_theory.max(), res_sorted.max())]
        ax2.plot(lims, lims, lw=1, linestyle="--", label="45°")
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
    ax2.set_title("QQ plot (std residuals)")
    ax2.set_xlabel("Theoretical quantiles (N(0,1))")
    ax2.set_ylabel("Empirical quantiles")
    ax2.grid(True, alpha=0.3)
    place_legend_below(ax2, ncols=2)

    # Panel 3: PIT histogram (ALL sites)
    ax3 = fig.add_subplot(gs[1, 0])
    U = np.array(pit_all, dtype=float)
    if U.size > 0:
        ax3.hist(U, bins=20, range=(0, 1), edgecolor="white", alpha=0.9, label="PIT (ALL)")
        ax3.axhline(y=U.size / 20.0, linestyle="--", lw=1.0, label="Uniform reference")
    ax3.set_title("PIT histogram (ALL)")
    ax3.set_xlabel("PIT")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    place_legend_below(ax3, ncols=2)

    # Panel 4: Calibration curve (empirical vs nominal)
    ax4 = fig.add_subplot(gs[1, 1])
    if calib_nominal.size > 0 and calib_empirical.size > 0:
        ax4.plot(calib_nominal, calib_empirical, marker="o", lw=1.5, label="Empirical")
        ax4.plot([0, 1], [0, 1], linestyle="--", lw=1.0, label="Ideal")
        x = calib_nominal
        y = calib_empirical
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            slope, intercept = coeffs[0], coeffs[1]
            ax4.text(0.05, 0.1, f"slope={slope:.3f}, intercept={intercept:.3f}", transform=ax4.transAxes)
    ax4.set_title("Calibration curve")
    ax4.set_xlabel("Nominal quantile")
    ax4.set_ylabel("Empirical fraction ≤ q")
    ax4.grid(True, alpha=0.3)
    place_legend_below(ax4, ncols=2)

    fig.suptitle(fig_title, y=0.98)
    fig.tight_layout()
    return fig

def _make_coverage_figure(coverage_df: pd.DataFrame) -> plt.Figure:
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    overall = coverage_df[coverage_df["site_id"] == "ALL"].sort_values("level")
    ax.bar([str(l) for l in overall["level"]], overall["empirical"], alpha=0.8, label="Empirical")
    ax.plot([str(l) for l in overall["level"]], overall["nominal"], linestyle="--", marker="o", label="Nominal")
    ax.set_xlabel("Credible interval level")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Coverage rates (overall)")
    ax.grid(True, axis="y", alpha=0.3)
    place_legend_below(ax, ncols=2)
    fig.tight_layout()
    return fig

def _make_rank_hist_figure(rank_bins_df: pd.DataFrame, pit_bins: int) -> plt.Figure:
    set_matplotlib_style()
    sites = rank_bins_df["site_id"].unique().tolist()
    sites = ["ALL"] + [s for s in sites if s != "ALL"]
    n_show = min(4, len(sites))
    cols = 2
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for i, sid in enumerate(sites[:n_show]):
        ax = axes[i]
        d = rank_bins_df[rank_bins_df["site_id"] == sid].sort_values("bin_left")
        ax.bar((d["bin_left"] + d["bin_right"]) / 2.0, d["count"], width=1.0 / pit_bins, alpha=0.85, label=f"PIT ranks ({sid})")
        ax.axhline(y=d["count"].mean() if d["count"].size > 0 else 0, linestyle="--", lw=1.0, label="Uniform ref")
        ax.set_title(f"PIT rank histogram — {sid}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        place_legend_below(ax, ncols=2)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig

# ---------- robust figure saver ----------
def _save_figure(ctx, fig: plt.Figure, name: str) -> None:
    try:
        ctx.write_figure(name, fig)
    except TypeError:
        ctx.write_figure(fig, name)
    except Exception:
        try:
            tmp = _repo_root() / f"{name}.png"
            fig.savefig(tmp, bbox_inches="tight", dpi=150)
            if hasattr(ctx, "write_image"):
                try:
                    ctx.write_image(name, tmp)
                except Exception:
                    pass
        except Exception:
            pass

# ---------- main ----------
def run_diagnostics(cfg: Dict, ctx) -> None:
    """
    Posterior predictive diagnostics (rigorous, analytic Beta–Binomial).

    Outputs (tables):
      - ppc_coverage            (also: coverage_rates)
      - ppc_pit_hist            (also: rank_hist_bins)
      - ppc_pit_ks              (also: sbc_ranks)
      - ppc_residuals_summary
      - ppc_calibration_curve
      - ppc_pit_sample
      - ppc_pred_obs_sample     (optional)

    Outputs (figures):
      - ppc_panel
      - ppc_coverage            (also: coverage_rates)
      - ppc_pit_rank            (also: sbc_rank_histograms)

    Outputs (metrics/report):
      - diagnostics_summary metric/table
      - diagnostics_report text
    """
    stage_name = "diagnostics"
    dcfg = cfg.get("diagnostics", {})
    seed = int(dcfg.get("seed", cfg.get("seed", 12345)))
    _set_global_seeds(seed)
    _log(ctx, "INFO", stage_name, "Diagnostics stage started", context={"seed": seed})

    # Config
    n_ppc_draws = int(dcfg.get("n_ppc_draws", 400))  # kept for interface; analytic PPC does not use draws
    coverage_levels = sorted([float(x) for x in dcfg.get("coverage_levels", [0.5, 0.8, 0.9, 0.95])])
    pit_bins = int(dcfg.get("pit_bins", 20))
    kappa_default = float(dcfg.get("kappa_default", 200.0))
    max_obs = dcfg.get("max_obs", None)
    if max_obs is not None:
        max_obs = int(max_obs)
    q_grid_points = int(dcfg.get("q_grid_points", 21))
    q_grid = np.linspace(0.05, 0.95, q_grid_points)
    chunk_rows = int(dcfg.get("chunk_rows", 3000))
    sample_pit_per_site = int(dcfg.get("sample_pit_per_site", 5000))
    write_pred_obs_sample = bool(dcfg.get("write_pred_obs_sample", True))
    pred_obs_sample_per_site = int(dcfg.get("pred_obs_sample_per_site", 300))

    # Load inputs
    obs = _load_observed_snv()
    sig = _load_signatures()
    theta = _load_smoothed_theta()
    priors_df = _load_priors_kappa()
    kappa_map = _extract_kappa_map(priors_df, kappa_default)

    # Align and compute predicted AF
    df = _compute_predicted_af(obs, theta, sig)
    df["kappa"] = df["mutation"].map(lambda m: float(kappa_map.get(m, kappa_default))).astype(float)

    # Optional subsample (site-stratified)
    if max_obs is not None and df.shape[0] > max_obs:
        rng = np.random.default_rng(seed)
        parts = []
        for sid, g in df.groupby("site_id", sort=False):
            n = len(g)
            take = max(1, int(np.floor(max_obs * (n / len(df)))))
            idx = rng.choice(n, size=take, replace=False)
            parts.append(g.iloc[np.sort(idx)])
        df = pd.concat(parts, axis=0, ignore_index=True)
        _log(ctx, "INFO", stage_name, "Subsampled observations for diagnostics",
             context={"max_obs": max_obs, "result_rows": df.shape[0]})

    # Analytic PPC + calibration
    (coverage_counts,
     pit_values,
     residuals,
     qq_theory,
     pit_hist,
     calib_empirical) = _ppc_and_calibration(
        df=df,
        levels=coverage_levels,
        n_draws=n_ppc_draws,
        q_grid=q_grid,
        pit_bins=pit_bins,
        chunk_rows=chunk_rows,
        seed=seed,
    )

    # ----- coverage table -----
    cov_rows = []
    for sid, lvl_dict in coverage_counts.items():
        for L, (cov, tot) in lvl_dict.items():
            rate = (cov / tot) if tot > 0 else np.nan
            cov_rows.append({"site_id": sid, "level": float(L),
                             "covered": int(cov), "total": int(tot),
                             "empirical": float(rate), "nominal": float(L)})
    coverage_df = pd.DataFrame(cov_rows).sort_values(["site_id", "level"]).reset_index(drop=True)
    # canonical + legacy name
    ctx.write_table("ppc_coverage", coverage_df)
    ctx.write_table("coverage_rates", coverage_df)

    # ----- PIT hist table -----
    edges = np.linspace(0.0, 1.0, pit_bins + 1)
    rank_rows = []
    for sid, hist in pit_hist.items():
        s = float(hist.sum()) if hist.sum() > 0 else 1.0
        for i in range(pit_bins):
            rank_rows.append({"site_id": sid,
                              "bin_left": float(edges[i]),
                              "bin_right": float(edges[i + 1]),
                              "count": float(hist[i]),
                              "density": float(hist[i] / s)})
    rank_bins_df = pd.DataFrame(rank_rows).sort_values(["site_id", "bin_left"]).reset_index(drop=True)
    ctx.write_table("ppc_pit_hist", rank_bins_df)
    ctx.write_table("rank_hist_bins", rank_bins_df)

    # ----- PIT KS per site -----
    ks_rows = []
    for sid, Ulist in pit_values.items():
        U = np.array(Ulist, dtype=float)
        U = U[np.isfinite(U)]
        if U.size >= 100:
            try:
                stat, pval = kstest(U, "uniform")
            except Exception:
                stat, pval = (np.nan, np.nan)
        else:
            stat, pval = (np.nan, np.nan)
        ks_rows.append({"site_id": sid, "n": int(U.size),
                        "pit_mean": float(np.nanmean(U)) if U.size > 0 else np.nan,
                        "pit_var": float(np.nanvar(U)) if U.size > 0 else np.nan,
                        "ks_stat": float(stat) if np.isfinite(stat) else np.nan,
                        "ks_pvalue": float(pval) if np.isfinite(pval) else np.nan})
    pit_ks_df = pd.DataFrame(ks_rows).sort_values(["site_id"]).reset_index(drop=True)
    ctx.write_table("ppc_pit_ks", pit_ks_df)
    ctx.write_table("sbc_ranks", pit_ks_df)

    # Downsample PIT for inspection
    pit_sample_rows: List[Dict] = []
    rng = np.random.default_rng(seed)
    for sid, Ulist in pit_values.items():
        U = np.array(Ulist, dtype=float)
        if U.size == 0:
            continue
        take = min(sample_pit_per_site, U.size)
        idx = rng.choice(U.size, size=take, replace=False)
        pit_sample_rows.extend([{"site_id": sid, "pit": float(U[i])} for i in idx])
    pit_sample_df = pd.DataFrame(pit_sample_rows)
    if not pit_sample_df.empty:
        ctx.write_table("ppc_pit_sample", pit_sample_df)

    # Residuals summary
    def _summary_stats(x: np.ndarray) -> Dict[str, float]:
        if x.size == 0:
            return {k: np.nan for k in ["n", "mean", "std", "q01", "q05", "q25", "q50", "q75", "q95", "q99"]}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "q01": float(np.quantile(x, 0.01)),
            "q05": float(np.quantile(x, 0.05)),
            "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
        }
    resid_stats = _summary_stats(residuals)
    resid_df = pd.DataFrame([{"site_id": "ALL", **resid_stats}])
    ctx.write_table("ppc_residuals_summary", resid_df)

    # Calibration curve table (+ slope/intercept attrs)
    calib_df = pd.DataFrame({"q_nominal": q_grid.astype(float),
                             "q_empirical": calib_empirical.astype(float)})
    mask = np.isfinite(calib_df["q_nominal"]) & np.isfinite(calib_df["q_empirical"])
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(calib_df.loc[mask, "q_nominal"],
                                      calib_df.loc[mask, "q_empirical"], 1)
    else:
        slope, intercept = (np.nan, np.nan)
    calib_df.attrs["slope"] = float(slope) if np.isfinite(slope) else np.nan
    calib_df.attrs["intercept"] = float(intercept) if np.isfinite(intercept) else np.nan
    ctx.write_table("ppc_calibration_curve", calib_df)

    # Optional: small pred-vs-obs sample
    if write_pred_obs_sample:
        cols = ["site_id", "date", "mutation", "count", "coverage", "obs_af", "pred_af", "kappa"]
        base = df[cols].copy()
        rows: List[pd.DataFrame] = []
        for sid, g in base.groupby("site_id", sort=False):
            if g.empty:
                continue
            take = min(pred_obs_sample_per_site, len(g))
            rows.append(g.sample(n=take, random_state=seed))
        if rows:
            sample_join = pd.concat(rows, axis=0).sort_values(["site_id", "date", "mutation"]).reset_index(drop=True)
            ctx.write_table("ppc_pred_obs_sample", sample_join)

    # ----- figures -----
    set_matplotlib_style()

    fig_ppc = _make_ppc_figure(
        residuals=residuals,
        qq_theory=qq_theory,
        pit_all=pit_values.get("ALL", []),
        calib_nominal=q_grid,
        calib_empirical=calib_empirical,
        fig_title="Posterior Predictive Checks",
    )
    _save_figure(ctx, fig_ppc, "ppc_panel")
    plt.close(fig_ppc)

    fig_cov = _make_coverage_figure(coverage_df)
    _save_figure(ctx, fig_cov, "ppc_coverage")
    _save_figure(ctx, fig_cov, "coverage_rates")  # legacy alias
    plt.close(fig_cov)

    fig_rank = _make_rank_hist_figure(rank_bins_df, pit_bins=pit_bins)
    _save_figure(ctx, fig_rank, "ppc_pit_rank")
    _save_figure(ctx, fig_rank, "sbc_rank_histograms")  # legacy alias
    plt.close(fig_rank)

    # ----- summary metrics & report -----
    overall_cov = coverage_df[coverage_df["site_id"] == "ALL"].copy()
    mean_abs_cov_err = float(np.mean(np.abs(overall_cov["empirical"] - overall_cov["nominal"]))) if not overall_cov.empty else np.nan
    ks_overall = pit_ks_df[pit_ks_df["site_id"] == "ALL"].copy()
    ks_stat = float(ks_overall["ks_stat"].iloc[0]) if not ks_overall.empty else np.nan
    ks_p = float(ks_overall["ks_pvalue"].iloc[0]) if not ks_overall.empty else np.nan

    summary = pd.DataFrame.from_records([{
        "mean_abs_coverage_error": mean_abs_cov_err,
        "pit_ks_stat_overall": ks_stat,
        "pit_ks_p_overall": ks_p,
        "calibration_slope": calib_df.attrs.get("slope", np.nan),
        "calibration_intercept": calib_df.attrs.get("intercept", np.nan),
        "resid_mean": resid_stats["mean"],
        "resid_sd": resid_stats["std"],
        "n_residuals": resid_stats["n"],
    }])
    try:
        ctx.write_metric("diagnostics_summary", summary)
    except Exception:
        ctx.write_table("diagnostics_summary", summary)

    # Text report
    lines: List[str] = []
    lines.append("# Diagnostics report\n")
    lines.append(f"- Pit bins: {pit_bins}\n")
    lines.append(f"- Coverage levels: {', '.join(f'{x:.2f}' for x in coverage_levels)}\n")
    lines.append(f"- Default kappa: {kappa_default}\n")
    lines.append("\n## Coverage (overall)\n")
    if not overall_cov.empty:
        for _, r in overall_cov.sort_values("level").iterrows():
            lines.append(f"* level={r['level']:.2f}: empirical={r['empirical']:.3f} (nominal={r['nominal']:.2f})\n")
        lines.append(f"\nMean absolute coverage error: {mean_abs_cov_err:.4f}\n")
    else:
        lines.append("No overall coverage rows.\n")
    lines.append("\n## PIT uniformity (overall)\n")
    lines.append(f"- KS statistic: {ks_stat:.4f}, p={ks_p:.3g}\n")
    lines.append("\n## Calibration curve\n")
    lines.append(f"- slope={calib_df.attrs.get('slope', np.nan):.3f}, intercept={calib_df.attrs.get('intercept', np.nan):.3f}\n")
    lines.append("\n## Residuals\n")
    lines.append(f"- mean={resid_stats['mean']:.4f}, sd={resid_stats['std']:.4f}, n={resid_stats['n']}\n")
    if hasattr(ctx, "write_report"):
        ctx.write_report("".join(lines))

    _log(ctx, "INFO", stage_name, "Diagnostics completed", context={
        "n_rows": int(len(df)), "n_sites": int(df["site_id"].nunique()),
        "mean_abs_cov_err": mean_abs_cov_err, "pit_ks_stat": ks_stat, "pit_ks_p": ks_p
    })

    try:
        if hasattr(ctx, "close") and callable(getattr(ctx, "close")):
            ctx.close(inputs=["results/forecast/tables/forecast_smoothed_props.*",
                              "results/preprocessing/tables/*",
                              "results/priors/tables/*"],
                      notes="Diagnostics completed")
    except Exception:
        pass
