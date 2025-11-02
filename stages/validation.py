import sys
import pathlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import gammaln, betaln
from datetime import datetime

# Root-anchored imports for flat repo
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Utils
try:
    from utils.plotting import set_matplotlib_style, place_legend_below
except Exception:
    def set_matplotlib_style() -> None:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.grid": True,
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

    def place_legend_below(ax: plt.Axes, ncol: int = 3) -> None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=ncol, frameon=False)

try:
    from utils.seeds import set_global_seeds
except Exception:
    import random
    def set_global_seeds(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)

# Stage name constant
STAGE_NAME = "validation"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _log_event(ctx, level: str, message: str, site_id: Optional[str] = None, lineage: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
    rec = {
        "time": _now_iso(),
        "level": level,
        "stage": STAGE_NAME,
        "site_id": site_id,
        "lineage": lineage,
        "message": message,
        "context": context or {},
    }
    # Attempt to use context logging hooks; degrade gracefully
    if hasattr(ctx, "write_log"):
        try:
            ctx.write_log("events", rec)
            return
        except Exception:
            pass
    if hasattr(ctx, "log"):
        try:
            ctx.log(rec)
            return
        except Exception:
            pass
    # else: silently ignore (tests may not require explicit file I/O here))


def _find_result_file(stage: str, target_name: str, subdir: Optional[str] = None) -> Optional[pathlib.Path]:
    """
    Search for a file with target_name under results/<stage> and results/runs/*/<stage>,
    prioritizing subdir if provided. Returns the most recent match by modification time.
    """
    candidates: List[pathlib.Path] = []
    stage_dir = ROOT / "results" / stage
    if subdir:
        candidates += list((stage_dir / subdir).glob(target_name))
    candidates += list(stage_dir.glob(target_name))
    runs_dir = ROOT / "results" / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("*")):
            stage_run_dir = run_dir / stage
            if subdir:
                candidates += list((stage_run_dir / subdir).glob(target_name))
            candidates += list(stage_run_dir.glob(target_name))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _load_csv_safely(path: pathlib.Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


def _beta_binom_logpmf(k: np.ndarray, n: np.ndarray, p: np.ndarray, phi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Beta-Binomial log PMF with mean p and concentration phi (alpha=p*phi, beta=(1-p)*phi).
    Numerically safe guards.
    """
    p = np.clip(p, eps, 1 - eps)
    phi = np.maximum(phi, eps)
    alpha = p * phi
    beta = (1.0 - p) * phi
    # log choose
    logc = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    # log beta ratio
    logb = betaln(k + alpha, (n - k) + beta) - betaln(alpha, beta)
    return logc + logb


def _beta_binom_cdf_mid_p(k: int, n: int, p: float, phi: float, eps: float = 1e-12) -> float:
    """
    Beta-Binomial mid-P CDF at k:
      CDF_midP = P(Y < k) + 0.5 * P(Y = k)
    Uses stable recurrence for PMF ratios.
    """
    p = float(np.clip(p, eps, 1 - eps))
    phi = max(float(phi), eps)
    alpha = p * phi
    beta = (1 - p) * phi

    # log PMF at 0
    log_pmf0 = betaln(alpha, n + beta) - betaln(alpha, beta)
    pmf = float(np.exp(log_pmf0))
    cdf = pmf if k >= 0 else 0.0
    if k == 0:
        return 0.5 * pmf
    # Recurrence for k>=1
    for i in range(0, k):
        # ratio = ((n - i) / (i + 1)) * ((i + alpha) / (n - i - 1 + beta))
        num1 = (n - i)
        den1 = (i + 1)
        num2 = (i + alpha)
        den2 = (n - i - 1 + beta)
        ratio = (num1 / den1) * (num2 / den2)
        pmf *= ratio
        cdf += pmf
    return float(cdf - 0.5 * pmf)


def _compute_rmse(obs: np.ndarray, pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    resid2 = (obs - pred) ** 2
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        m = np.average(resid2, weights=w)
    else:
        m = float(np.mean(resid2))
    return float(np.sqrt(m))


def _calibration_slope_intercept(pred: np.ndarray, obs: np.ndarray, nbins: int = 10) -> Tuple[float, float]:
    """
    Binned reliability regression: fit obs_bin ~ a + b * pred_bin over nbins.
    Returns slope b and intercept a.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.size == 0:
        return (np.nan, np.nan)
    quantiles = np.quantile(pred, np.linspace(0, 1, nbins + 1))
    # ensure unique edges
    quantiles[0] = 0.0
    quantiles[-1] = 1.0
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = np.nextafter(quantiles[i - 1], 1.0)
    bins = np.digitize(pred, quantiles[1:-1], right=True)
    bin_means_pred = []
    bin_means_obs = []
    for b in range(nbins):
        mask = bins == b
        if not np.any(mask):
            continue
        bin_means_pred.append(float(np.mean(pred[mask])))
        bin_means_obs.append(float(np.mean(obs[mask])))
    if len(bin_means_pred) < 2:
        return (np.nan, np.nan)
    x = np.asarray(bin_means_pred)
    y = np.asarray(bin_means_obs)
    # linear regression y = a + b x
    X = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    return b, a


def _build_signature_matrix(df_sig: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Pivot signature long table to matrix (mutations x lineages).
    """
    required_cols = {"mutation", "lineage", "weight"}
    if not required_cols.issubset(set(df_sig.columns)):
        raise ValueError("signatures.csv missing required columns: mutation,lineage,weight")
    S = df_sig.pivot_table(index="mutation", columns="lineage", values="weight", fill_value=0.0, aggfunc="mean")
    mutations = list(S.index)
    lineages = list(S.columns)
    return S, mutations, lineages


def _extract_theta_wide(df_theta_long: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert long theta table to wide matrix over lineages for each (site,date).
    Compatible with typical columns from forecast_smoothed_props.csv.
    """
    # expected columns: site_id, date, lineage, and one of ['median','mean','theta','value']
    col_options = ["median", "mean", "theta", "value", "prop"]
    val_col = None
    for c in col_options:
        if c in df_theta_long.columns:
            val_col = c
            break
    if val_col is None:
        # if CI columns exist, prefer median or fallback to 'estimate'
        for c in df_theta_long.columns:
            if c not in ["site_id", "date", "lineage"] and df_theta_long[c].dtype.kind in ("f", "i"):
                val_col = c
                break
    if val_col is None:
        raise ValueError("Theta long table missing value column among expected candidates.")
    pivot = df_theta_long.pivot_table(index=["site_id", "date"], columns="lineage", values=val_col, aggfunc="mean")
    pivot = pivot.fillna(0.0)
    # Project rows to simplex to guard numerical drift
    arr = pivot.to_numpy(dtype=float)
    arr[arr < 0] = 0.0
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    arr = arr / row_sums
    pivot.iloc[:, :] = arr
    lineages = list(pivot.columns)
    return pivot, lineages


def _estimate_phi_from_counts(df_counts: pd.DataFrame) -> pd.Series:
    """
    Estimate Beta-Binomial concentration phi per mutation using method-of-moments over pooled observations.
    Returns a Series indexed by mutation.
    """
    # df_counts: columns ['site_id','date','mutation','count','coverage']
    grouped = df_counts.groupby("mutation", observed=True)
    phis = {}
    for mut, grp in grouped:
        y = grp["count"].to_numpy(dtype=float)
        n = grp["coverage"].to_numpy(dtype=float)
        mask = n > 0
        y = y[mask]
        n = n[mask]
        if y.size < 2:
            phis[mut] = 100.0
            continue
        p_hat = np.sum(y) / np.sum(n)
        p_hat = float(np.clip(p_hat, 1e-6, 1 - 1e-6))
        # sample variance of counts
        var_y = float(np.var(y, ddof=1))
        n_bar = float(np.mean(n))
        if n_bar <= 1.0:
            phis[mut] = 100.0
            continue
        # Use Var[y] = n p(1-p) (1 + (n-1) rho) ; rho = 1/(phi+1)
        denom = (n_bar * p_hat * (1.0 - p_hat))
        if denom <= 0:
            phis[mut] = 100.0
            continue
        r = var_y / denom
        rho = (r - 1.0) / (n_bar - 1.0)
        rho = float(np.clip(rho, 1e-6, 1.0 - 1e-6))
        phi = (1.0 / rho) - 1.0
        phis[mut] = float(np.clip(phi, 1.0, 1e6))
    return pd.Series(phis, name="phi")


def _prepare_evaluation_tables(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load required inputs: counts (data/jahn_like.csv), signatures, theta (forecast output).
    Returns: df_eval (rows per site/date/mutation with obs and pred), theta_wide, signatures pivot.
    """
    # Load counts
    counts_path = ROOT / "data" / "jahn_like.csv"
    if not counts_path.exists():
        raise FileNotFoundError("data/jahn_like.csv not found for validation stage.")
    df_counts = _load_csv_safely(counts_path, parse_dates=["date"])
    required = {"sample_id", "site_id", "date", "mutation", "count", "coverage"}
    if not required.issubset(df_counts.columns):
        missing = sorted(required - set(df_counts.columns))
        raise ValueError(f"jahn_like.csv missing required columns: {missing}")
    # Filter coverage > 0
    df_counts = df_counts[df_counts["coverage"] > 0].copy()
    df_counts["af_obs"] = df_counts["count"] / df_counts["coverage"]
    df_counts["af_obs"] = df_counts["af_obs"].clip(0.0, 1.0)

    # Load signatures
    sig_path = ROOT / "data" / "signatures.csv"
    if not sig_path.exists():
        raise FileNotFoundError("data/signatures.csv not found for validation stage.")
    df_sig = _load_csv_safely(sig_path)
    S, mut_order, lin_order_sig = _build_signature_matrix(df_sig)

    # Load smoothed theta
    theta_path = _find_result_file("forecast", "forecast_smoothed_props.csv", subdir="tables")
    if theta_path is None:
        # try without subdir
        theta_path = _find_result_file("forecast", "forecast_smoothed_props.csv")
    if theta_path is None:
        raise FileNotFoundError("forecast_smoothed_props.csv not found in results for validation stage.")
    df_theta_long = _load_csv_safely(theta_path, parse_dates=["date"])
    if not {"site_id", "date", "lineage"}.issubset(df_theta_long.columns):
        raise ValueError("forecast_smoothed_props.csv missing required columns: site_id,date,lineage")
    theta_wide, lin_order_theta = _extract_theta_wide(df_theta_long)

    # Align lineages between signatures and theta
    common_lineages = [l for l in lin_order_theta if l in lin_order_sig]
    if len(common_lineages) == 0:
        raise ValueError("No overlapping lineages between signatures and forecast theta.")
    S_aligned = S.loc[:, common_lineages].copy()
    theta_wide_aligned = theta_wide.loc[:, common_lineages].copy()

    # Build predicted per (site,date,mutation): p_pred = S * theta
    df_site_date = theta_wide_aligned.reset_index()
    df_counts_sub = df_counts.copy()
    df_counts_sub = df_counts_sub[df_counts_sub["mutation"].isin(S_aligned.index)].copy()

    key_cols = ["site_id", "date"]
    df_eval = df_counts_sub.merge(df_site_date, on=key_cols, how="inner", suffixes=("", "_theta"))
    theta_lookup = theta_wide_aligned.reset_index().set_index(key_cols)
    S_mat = S_aligned.to_numpy(dtype=float)
    mut_to_row = {m: i for i, m in enumerate(S_aligned.index)}
    lin_cols = S_aligned.columns.tolist()

    def row_pred_p(r: pd.Series) -> float:
        m = r["mutation"]
        if m not in mut_to_row:
            return np.nan
        s_row = S_mat[mut_to_row[m], :]
        try:
            theta_vec = theta_lookup.loc[(r["site_id"], r["date"]), lin_cols].to_numpy(dtype=float)
        except KeyError:
            return np.nan
        val = float(np.dot(s_row, theta_vec))
        return max(0.0, min(1.0, val))

    df_eval["p_pred"] = df_eval.apply(row_pred_p, axis=1)
    df_eval = df_eval.dropna(subset=["p_pred"]).copy()
    df_eval["date"] = pd.to_datetime(df_eval["date"])

    return df_eval, theta_wide_aligned.reset_index(), S_aligned.reset_index()


def _load_priors_phi() -> Optional[pd.Series]:
    """
    Load priors hyperparameters and return phi per mutation if available.
    """
    pri_path = _find_result_file("priors", "priors_hyperparams.csv", subdir="tables")
    if pri_path is None:
        pri_path = _find_result_file("priors", "priors_hyperparams.csv")
    if pri_path is None:
        return None
    df = _load_csv_safely(pri_path)
    if "mutation" not in df.columns:
        return None
    if "phi" in df.columns:
        ser = df.set_index("mutation")["phi"].astype(float)
        return ser
    if {"alpha", "beta"}.issubset(df.columns):
        ser = (df["alpha"].astype(float) + df["beta"].astype(float))
        ser.index = df["mutation"].astype(str)
        ser.name = "phi"
        return ser
    # Try generic columns names
    if {"a", "b"}.issubset(df.columns):
        ser = (df["a"].astype(float) + df["b"].astype(float))
        ser.index = df["mutation"].astype(str)
        ser.name = "phi"
        return ser
    return None


def _compute_metrics(df_eval: pd.DataFrame, phi_by_mut: pd.Series, min_prob: float) -> Dict[str, float]:
    """
    Compute RMSE, logscore (sum log-likelihood), calibration slope/intercept.
    """
    df = df_eval.copy()
    # Align phi
    df = df.join(phi_by_mut.rename("phi"), on="mutation")
    # Fallback: if any phi missing, use median across mutations
    if df["phi"].isna().any():
        median_phi = float(np.nanmedian(df["phi"].to_numpy(dtype=float)))
        df["phi"] = df["phi"].fillna(median_phi)
    obs = df["af_obs"].to_numpy(dtype=float)
    pred = df["p_pred"].to_numpy(dtype=float)
    n = df["coverage"].to_numpy(dtype=float)
    y = df["count"].to_numpy(dtype=float)
    phi = df["phi"].to_numpy(dtype=float)
    rmse = _compute_rmse(obs, pred, weights=n)
    logscore = float(np.sum(_beta_binom_logpmf(y, n, np.clip(pred, min_prob, 1 - min_prob), phi)))
    slope, intercept = _calibration_slope_intercept(pred, obs, nbins=10)
    return {"rmse": rmse, "logscore": logscore, "calibration_slope": slope, "calibration_intercept": intercept}


def _time_splits(dates: np.ndarray, k: int) -> List[np.ndarray]:
    """
    Split sorted unique dates into k contiguous folds; returns list of boolean masks over the dates array.
    """
    unique = np.unique(dates)
    unique.sort()
    folds = np.array_split(unique, k)
    masks = []
    for f in folds:
        mask = np.isin(dates, f)
        masks.append(mask)
    return masks


# ---------- robust figure saver ----------
def _save_figure(ctx, fig: plt.Figure, name: str) -> None:
    try:
        ctx.write_figure(name, fig)
    except TypeError:
        ctx.write_figure(fig, name)
    except Exception:
        try:
            tmp = ROOT / f"{name}.png"
            fig.savefig(tmp, bbox_inches="tight", dpi=150)
            if hasattr(ctx, "write_image"):
                try:
                    ctx.write_image(name, tmp)
                except Exception:
                    pass
        except Exception:
            pass


def run_validation(cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    """
    Validation stage:
      - Align counts, signatures, and smoothed θ
      - Predict AF via Sθ
      - Score with coverage‑weighted RMSE and Beta–Binomial log score
      - Compute calibration slope/intercept (reliability regression)
      - Summaries: overall, per‑site, temporal splits; calibration bins
      - Figures: Obs vs Pred, calibration curve, site RMSE bars, φ histogram
    """
    vcfg = cfg.get("validation", {})
    seed = int(vcfg.get("seed", cfg.get("seed", 20240229)))
    set_global_seeds(seed)
    _log_event(ctx, "INFO", "Validation stage started", context={"seed": seed})

    # --- Config ---
    min_prob = float(vcfg.get("min_prob", 1e-6))
    phi_source = str(vcfg.get("phi_source", "auto")).lower()  # 'auto' | 'priors' | 'estimate'
    phi_default = float(vcfg.get("phi_default", 100.0))
    nbins_cal = int(vcfg.get("calibration_bins", 12))
    k_time = int(vcfg.get("k_time_splits", 5))
    topk_sites = int(vcfg.get("site_rmse_topk", 16))
    scatter_max_points = int(vcfg.get("scatter_max_points", 20000))
    hexbin_min_points = int(vcfg.get("hexbin_min_points", 15000))
    weight_calibration_by_cov = bool(vcfg.get("weight_calibration_by_coverage", True))

    # --- Load/align & predict ---
    df_eval, theta_wide, S_tbl = _prepare_evaluation_tables(cfg)

    # --- φ per mutation ---
    phi_series: Optional[pd.Series] = None
    if phi_source in ("auto", "priors"):
        phi_series = _load_priors_phi()
    if (phi_source in ("auto", "estimate") and (phi_series is None or phi_series.empty)):
        phi_series = _estimate_phi_from_counts(df_eval[["site_id", "date", "mutation", "count", "coverage"]])
    if phi_series is None or phi_series.empty:
        # Uniform default
        phi_series = pd.Series(phi_default, index=pd.Index(df_eval["mutation"].unique().tolist(), name="mutation"), name="phi")
    # Clean φ
    phi_series = phi_series.astype(float).clip(lower=1.0).sort_index()

    # --- Overall metrics ---
    overall = _compute_metrics(df_eval, phi_series, min_prob=min_prob)
    overall_df = pd.DataFrame([overall])
    try:
        ctx.write_metric("validation_overall", overall_df)
    except Exception:
        ctx.write_table("validation_overall", overall_df)

    # --- Per-site metrics ---
    site_rows: List[Dict[str, Any]] = []
    for sid, g in df_eval.groupby("site_id", sort=False):
        m = _compute_metrics(g, phi_series, min_prob=min_prob)
        m.update({
            "site_id": sid,
            "n_rows": int(len(g)),
            "mean_cov": float(np.mean(g["coverage"])) if len(g) > 0 else np.nan
        })
        site_rows.append(m)
    by_site_df = pd.DataFrame(site_rows).sort_values("rmse").reset_index(drop=True)
    ctx.write_table("validation_by_site", by_site_df)

    # --- Temporal splits (contiguous folds over dates) ---
    time_masks = _time_splits(df_eval["date"].to_numpy(), max(1, k_time))
    time_rows: List[Dict[str, Any]] = []
    dates = df_eval["date"].to_numpy()
    for i, mask in enumerate(time_masks, start=1):
        sub = df_eval.loc[mask].copy()
        if sub.empty:
            continue
        tm = _compute_metrics(sub, phi_series, min_prob=min_prob)
        tmin, tmax = pd.to_datetime(sub["date"].min()), pd.to_datetime(sub["date"].max())
        tm.update({"fold": i, "start": tmin, "end": tmax, "n_rows": int(len(sub))})
        time_rows.append(tm)
    temporal_df = pd.DataFrame(time_rows).sort_values("fold").reset_index(drop=True)
    if not temporal_df.empty:
        ctx.write_table("validation_temporal_splits", temporal_df)

    # --- Calibration bins (quantile reliability) ---
    pred = df_eval["p_pred"].to_numpy(dtype=float)
    obs = df_eval["af_obs"].to_numpy(dtype=float)
    cov = df_eval["coverage"].to_numpy(dtype=float)
    # Bin edges by quantiles, ensure strictly increasing
    q = np.linspace(0.0, 1.0, nbins_cal + 1)
    edges = np.quantile(pred, q)
    edges[0], edges[-1] = 0.0, 1.0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], 1.0)
    bins = np.digitize(pred, edges[1:-1], right=True)
    rows_cal: List[Dict[str, Any]] = []
    for b in range(nbins_cal):
        m = (bins == b)
        if not np.any(m):
            rows_cal.append({"bin": b, "bin_left": edges[b], "bin_right": edges[b + 1],
                             "mean_pred": np.nan, "mean_obs": np.nan, "count": 0})
            continue
        if weight_calibration_by_cov:
            w = cov[m]
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            mp = float(np.average(pred[m], weights=w))
            mo = float(np.average(obs[m], weights=w))
        else:
            mp = float(np.mean(pred[m]))
            mo = float(np.mean(obs[m]))
        rows_cal.append({"bin": b, "bin_left": float(edges[b]), "bin_right": float(edges[b + 1]),
                         "mean_pred": mp, "mean_obs": mo, "count": int(m.sum())})
    cal_bins_df = pd.DataFrame(rows_cal)
    ctx.write_table("validation_calibration_bins", cal_bins_df)

    # --- Sample predictions for inspection (and plotting) ---
    N = len(df_eval)
    sample_n = min(scatter_max_points, N)
    if sample_n < N:
        sample_df = df_eval.sample(n=sample_n, random_state=seed).copy()
    else:
        sample_df = df_eval.copy()
    sample_keep = ["site_id", "date", "mutation", "coverage", "count", "af_obs", "p_pred"]
    ctx.write_table("validation_predictions_sample", sample_df[sample_keep].sort_values(["site_id", "date", "mutation"]))  # tidy

    # --- Figures ---
    set_matplotlib_style()

    # 1) Obs vs Pred AF
    fig1, ax1 = plt.subplots(figsize=(6.8, 6.0))
    if N >= hexbin_min_points:
        hb = ax1.hexbin(sample_df["af_obs"], sample_df["p_pred"], gridsize=40, mincnt=1)
        cbar = fig1.colorbar(hb, ax=ax1)
        cbar.set_label("count")
    else:
        ax1.scatter(sample_df["af_obs"], sample_df["p_pred"], s=8, alpha=0.5)
    ax1.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--")
    ax1.set_xlabel("Observed AF")
    ax1.set_ylabel("Predicted AF (Sθ)")
    ax1.set_title(f"Observed vs Predicted AF (N={N}, RMSE={overall['rmse']:.4f})")
    ax1.grid(True, alpha=0.3)
    _save_figure(ctx, fig1, "validation_obs_vs_pred")
    plt.close(fig1)

    # 2) Calibration curve
    fig2, ax2 = plt.subplots(figsize=(6.8, 5.2))
    ax2.plot(cal_bins_df["mean_pred"], cal_bins_df["mean_obs"], marker="o", lw=1.5, label="Empirical")
    ax2.plot([0, 1], [0, 1], linestyle="--", lw=1.0, color="black", label="Ideal")
    slope, intercept = overall["calibration_slope"], overall["calibration_intercept"]
    ax2.text(0.04, 0.08, f"slope={slope:.3f}, intercept={intercept:.3f}", transform=ax2.transAxes)
    ax2.set_xlabel("Predicted AF (bin mean)")
    ax2.set_ylabel("Observed AF (bin mean)")
    ax2.set_title("Reliability (quantile bins)")
    ax2.grid(True, alpha=0.3)
    place_legend_below(ax2, ncol=2)
    _save_figure(ctx, fig2, "validation_calibration")
    plt.close(fig2)

    # 3) Site RMSE bars (top-k worst)
    fig3, ax3 = plt.subplots(figsize=(10, 5.5))
    worst = by_site_df.sort_values("rmse", ascending=False).head(topk_sites)
    ax3.barh(worst["site_id"], worst["rmse"], alpha=0.85)
    ax3.invert_yaxis()
    ax3.set_xlabel("RMSE (coverage-weighted)")
    ax3.set_title(f"Worst {len(worst)} sites by RMSE")
    ax3.grid(True, axis="x", alpha=0.3)
    _save_figure(ctx, fig3, "validation_site_rmse")
    plt.close(fig3)

    # 4) φ histogram
    fig4, ax4 = plt.subplots(figsize=(6.8, 5.0))
    phi_vals = np.asarray(phi_series.values, dtype=float)
    ax4.hist(phi_vals[np.isfinite(phi_vals)], bins=30, alpha=0.85)
    ax4.set_xlabel("φ (Beta–Binomial concentration)")
    ax4.set_ylabel("Mutations")
    ax4.set_title("Distribution of φ across mutations")
    ax4.grid(True, alpha=0.3)
    _save_figure(ctx, fig4, "validation_phi_hist")
    plt.close(fig4)

    # --- Report ---
    n_sites = int(df_eval["site_id"].nunique())
    n_mut = int(df_eval["mutation"].nunique())
    report_lines = [
        "# Validation report",
        f"- Rows evaluated: {len(df_eval)}",
        f"- Sites: {n_sites}",
        f"- Mutations: {n_mut}",
        f"- φ source: {'priors' if phi_source=='priors' else ('estimate' if phi_source=='estimate' else 'auto')}, default={phi_default}",
        "",
        "## Overall metrics",
        f"- RMSE (coverage‑weighted): {overall['rmse']:.6f}",
        f"- Log score (sum log‑likelihood): {overall['logscore']:.2f}",
        f"- Calibration slope: {overall['calibration_slope']:.4f}",
        f"- Calibration intercept: {overall['calibration_intercept']:.4f}",
    ]
    if hasattr(ctx, "write_report"):
        ctx.write_report("\n".join(report_lines))

    # --- Bundle return ---
    bundle = {
        "n_rows": len(df_eval),
        "n_sites": n_sites,
        "n_mutations": n_mut,
        "rmse": overall["rmse"],
        "logscore": overall["logscore"],
        "calibration_slope": overall["calibration_slope"],
        "calibration_intercept": overall["calibration_intercept"],
    }
    _log_event(ctx, "INFO", "Validation stage completed", context=bundle)
    return bundle
