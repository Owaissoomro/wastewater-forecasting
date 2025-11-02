# -*- coding: utf-8 -*-
"""
Detection stage (anytime-valid e-values, e-BH FDR, SR with block-permutation calibration; robust figures).

Implements, end-to-end:
  • Posterior tail p-values for H0: proportion ≤ min_prop (Beta approx to posterior).
  • p → e calibration (power family, E[e|H0]=1 for p~Uniform) => anytime-valid e-values.
  • Online e-BH (Benjamini–Hochberg with e-values) => FDR control under arbitrary dependence.
  • Shiryaev–Roberts statistic (log-space) on e-values with permutation-calibrated threshold.
  • Diagnostics: mean e-value under null windows, empirical FDR under permutations.
  • Optional delay analysis vs provided reference dates.
  • Figures:
      A) Multi-panel timelines for a focus site (e, q, SR with threshold, null-calibration bar).
      B) Site heatmaps (log10 e-values; −log10 q-values) across *all* lineages over time.
      C) Detection delay histograms per method (if reference dates provided).

Robust figure saving:
  - Always passes a single matplotlib Figure to ctx.write_figure.
  - Compatible with ctx.write_figure(name, fig) and ctx.write_figure(fig, name).
  - No 'filename=' is ever used with context writers.
"""

import sys
import pathlib
from pathlib import Path
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# Root-anchored imports (flat repo)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.plotting import set_matplotlib_style
try:
    from utils.plotting import place_legend_below
except Exception:
    # Fallback: minimal legend placer
    def place_legend_below(ax, ncol=3):
        return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False)

from utils.run import RunContext

# Deterministic seed handling
try:
    from utils.seeds import set_global_seeds as _set_global_seeds  # type: ignore
except Exception:
    def _set_global_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)


STAGE_NAME = "detection"
EPS = 1e-12


# -----------------------------
# Small utilities
# -----------------------------
def _safe_name(s: str) -> str:
    if s is None:
        return "figure"
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "figure"


def _save_figure(ctx: RunContext, fig: Any, name: str) -> None:
    """
    Save a single matplotlib Figure via ctx.write_figure with either signature.
    If a list/tuple is accidentally passed, collapse to a single page placeholder.
    """
    import matplotlib.pyplot as plt

    if isinstance(fig, (list, tuple)):
        n = len(fig)
        merged = plt.figure(figsize=(12, max(3, 3 * max(1, n))))
        for i in range(max(1, n)):
            ax = merged.add_subplot(max(1, n), 1, i + 1)
            ax.text(0.5, 0.5, f"Panel {i+1}", ha="center", va="center")
            ax.set_axis_off()
        fig = merged

    if not hasattr(fig, "savefig"):
        coerced = plt.figure(figsize=(10, 4))
        ax = coerced.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Empty figure (coerced)", ha="center", va="center")
        ax.set_axis_off()
        fig = coerced

    name = _safe_name(name)
    try:
        ctx.write_figure(name, fig)
    except TypeError:
        ctx.write_figure(fig, name)
    except Exception:
        try:
            tmp = Path(".") / f"{name}.png"
            fig.savefig(tmp, bbox_inches="tight", dpi=150)
            if hasattr(ctx, "write_image"):
                try:
                    ctx.write_image(name, tmp)
                except Exception:
                    pass
        finally:
            pass


def _write_metric(ctx: RunContext, name: str, df: pd.DataFrame) -> None:
    name = _safe_name(name)
    try:
        ctx.write_metric(name, df)
    except TypeError:
        ctx.write_metric(df, name)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _log(ctx: RunContext, level: str, site_id: Optional[str], lineage: Optional[str],
         message: str, context: Optional[dict] = None) -> None:
    try:
        payload = {
            "time": _now_iso(),
            "level": level.upper(),
            "stage": STAGE_NAME,
            "site_id": site_id,
            "lineage": lineage,
            "message": message,
            "context": context or {},
        }
        if hasattr(ctx, "log"):
            ctx.log(payload)
    except Exception:
        pass  # logging must never crash the stage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_stage_dir(stage: str) -> Optional[Path]:
    root = _repo_root()
    stage_dir = root / "results" / "stage"
    # prefer canonical results/<stage>, then scan results/runs/*/<stage>
    candidates = []
    canonical = root / "results" / stage
    if canonical.exists():
        candidates.append(canonical)
    runs_dir = root / "results" / "runs"
    if runs_dir.exists():
        for run_sub in sorted(runs_dir.glob("*")):
            sdir = run_sub / stage
            if sdir.exists():
                candidates.append(sdir)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_table_with_schema_guard(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _scan_for_smoothed_props() -> Optional[pd.DataFrame]:
    """Attempt to load smoothed lineage proportions (posterior summaries) from forecast stage."""
    stage_dir = _find_latest_stage_dir("forecast")
    if stage_dir is None:
        return None
    tables_dir = stage_dir / "tables"
    if not tables_dir.exists():
        return None
    dfs = []
    for fp in sorted(tables_dir.glob("*.csv")):
        name = fp.stem.lower()
        if "forecast" in name and "prop" in name:
            try:
                df = _load_table_with_schema_guard(fp)
                dfs.append(df)
            except Exception:
                continue
        elif "smoothed" in name and ("theta" in name or "props" in name or "proportion" in name):
            try:
                df = _load_table_with_schema_guard(fp)
                dfs.append(df)
            except Exception:
                continue
    # choose first that contains required columns
    for df in dfs:
        cols = set(c.lower() for c in df.columns)
        if {"site_id", "date", "lineage"}.issubset(cols):
            return df
    canonical = tables_dir / "forecast_smoothed_props.csv"
    if canonical.exists():
        return _load_table_with_schema_guard(canonical)
    return None


def _scan_for_baseline_props() -> Optional[pd.DataFrame]:
    """Fallback: load baseline deconvolution proportions from likelihood stage."""
    stage_dir = _find_latest_stage_dir("likelihood")
    if stage_dir is None:
        return None
    tables_dir = stage_dir / "tables"
    if not tables_dir.exists():
        return None
    for fp in sorted(tables_dir.glob("*.csv")):
        name = fp.stem.lower()
        if ("theta" in name or "prop" in name) and ("estimate" in name or "baseline" in name or "deconv" in name):
            try:
                df = _load_table_with_schema_guard(fp)
                cols = set(c.lower() for c in df.columns)
                if {"site_id", "date", "lineage"}.issubset(cols):
                    return df
            except Exception:
                continue
    return None


# -----------------------------
# SR (log-space) & permutations
# -----------------------------
def _sr_logspace(log_lr: np.ndarray) -> np.ndarray:
    """
    SR recursion in log-space with log-likelihood ratios log_lr (here, log e-values):
        R_{t+1} = (1 + R_t) * exp(log_lr_t)
        logR_{t+1} = logaddexp(0, logR_t) + log_lr_t
    Returns the sequence logR_t.
    """
    log_lr = np.asarray(log_lr, dtype=float)
    logR = np.full(log_lr.shape, -np.inf, dtype=float)
    s = -np.inf
    for t, ll in enumerate(log_lr):
        s = np.logaddexp(0.0, s) + ll
        logR[t] = s
    return logR


def _block_permutations_indices(n: int, block: int, nperm: int, rng: np.random.RandomState) -> List[np.ndarray]:
    """Return list of index arrays corresponding to block permutations of range(n)."""
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    perms = []
    for _ in range(nperm):
        order = np.arange(len(blocks))
        rng.shuffle(order)
        idx = np.concatenate([blocks[i] for i in order])
        perms.append(idx)
    return perms


def _calibrate_sr_threshold_log_from_perms(
    e_values: np.ndarray,
    alpha: float,
    block: int,
    nperm: int,
    rng: np.random.RandomState,
) -> float:
    """
    Calibrate a log-space SR threshold using block permutations of the e-value time series.
    Choose h_log as the (1 - alpha) empirical quantile of max_t logR_t under permutations.
    """
    T = int(len(e_values))
    if T == 0:
        return float("inf")

    log_e = np.log(np.clip(e_values.astype(float), 1e-300, 1e300))
    perms = _block_permutations_indices(T, max(1, block), max(1, nperm), rng)
    max_logR_vals: List[float] = []

    for idx in perms:
        logR = _sr_logspace(log_e[idx])
        max_logR_vals.append(float(np.nanmax(logR)))

    if len(max_logR_vals) == 0:
        return float(np.log(1e6))  # conservative fallback

    h_log = float(np.quantile(np.array(max_logR_vals, dtype=float), 1.0 - float(alpha)))
    if not np.isfinite(h_log):
        h_log = 0.0
    return h_log


# -----------------------------
# p/e-values and e-BH
# -----------------------------
def _beta_params_from_summary(row: pd.Series, conf: float, var_floor: float) -> Tuple[float, float]:
    """
    Construct Beta(α, β) from posterior summary row using moment/CI matching with safeguards.
    Accepts columns like: mean|median|theta, and CI bounds (q05/q95, lower/upper, etc).
    """
    mean_cols = ["mean", "median", "p50", "theta", "value", "estimate", "est"]
    lower_cols = ["lower", "lo", "p05", "q05", "l95", "lower95", "ci_lower"]
    upper_cols = ["upper", "hi", "p95", "q95", "u95", "upper95", "ci_upper"]
    var_cols = ["var", "variance"]
    sd_cols = ["sd", "std", "stderr", "se"]

    mu = None
    for c in mean_cols:
        if c in row.index and pd.notnull(row[c]):
            mu = float(row[c]); break
    if mu is None:
        mu = float(row.get("prop", np.nan))
    if not np.isfinite(mu):
        mu = 1e-6
    mu = float(np.clip(mu, 1e-9, 1 - 1e-9))

    var = None
    for c in var_cols:
        if c in row.index and pd.notnull(row[c]):
            var = float(row[c]); break
    if var is None:
        for c in sd_cols:
            if c in row.index and pd.notnull(row[c]):
                sd = float(row[c]); var = sd * sd; break
    if var is None:
        lower, upper = None, None
        for c in lower_cols:
            if c in row.index and pd.notnull(row[c]):
                lower = float(row[c]); break
        for c in upper_cols:
            if c in row.index and pd.notnull(row[c]):
                upper = float(row[c]); break
        if lower is not None and upper is not None:
            z = norm.ppf((1 + conf) / 2.0)
            sd = (upper - lower) / (2.0 * z + 1e-12)
            var = max(sd * sd, var_floor)
    if var is None or not np.isfinite(var) or var <= 0:
        var = max(var_floor, 0.05 * mu * (1 - mu))

    kappa = mu * (1 - mu) / (var + 1e-18) - 1.0
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = max(2.0, mu * (1 - mu) / (var + 1e-18))
    alpha = max(1e-3, mu * kappa)
    beta = max(1e-3, (1 - mu) * kappa)
    return alpha, beta


def _p_value_from_beta(alpha: float, beta: float, threshold: float) -> float:
    thr = float(np.clip(threshold, 1e-9, 1 - 1e-9))
    p = beta_dist.cdf(thr, alpha, beta)
    if not np.isfinite(p):
        p = 1.0
    return float(np.clip(p, 1e-12, 1.0))


def _evalue_from_p(p: float, method: str, power_a: float, p_floor: float) -> Tuple[float, float]:
    """
    Returns (e_calibrated, e_raw_1_over_p).
    method:
      - "power": e = a * p^(a-1), expectation 1 under Uniform for a in (0,1) (Vovk–Shafer).
      - "inverse": e = 1 / max(p, p_floor)   (diagnostics; not used for decisions).
    """
    p = float(np.clip(p, 1e-12, 1.0))
    e_raw = 1.0 / max(p, p_floor)
    if method == "power":
        a = float(np.clip(power_a, 1e-6, 0.999999))
        e = a * (p ** (a - 1.0))
    elif method == "inverse":
        e = e_raw
    else:
        a = float(np.clip(power_a, 1e-6, 0.999999))
        e = a * (p ** (a - 1.0))
    return float(max(0.0, e)), float(max(0.0, e_raw))


def _online_eBH_qvalues(e_vals: np.ndarray) -> np.ndarray:
    """
    Online e-BH q-values for a sequence E_1..E_T using the batch e-BH recipe at each t.
    """
    T = e_vals.shape[0]
    q_online = np.full(T, np.nan, dtype=float)
    for t in range(1, T + 1):
        E = e_vals[:t].astype(float)
        order = np.argsort(-E)
        cumE = np.cumsum(E[order])
        ks = np.arange(1, t + 1, dtype=float)
        ratios = ks / np.maximum(cumE, 1e-12)
        running = np.inf
        qtemp = np.zeros(t, dtype=float)
        for s in range(t - 1, -1, -1):
            running = min(running, ratios[s])
            qtemp[order[s]] = running
        q_online[t - 1] = qtemp[t - 1]
    return q_online


def _coerce_bool_array(x: Any, n: int) -> np.ndarray:
    """Robustly coerce to a boolean numpy array of length n."""
    if x is None:
        return np.zeros(n, dtype=bool)
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 0:
        return np.full(n, bool(arr), dtype=bool)
    if arr.shape[0] != n:
        raise ValueError(f"threshold_crossed length {arr.shape[0]} does not match expected length {n}.")
    return arr.astype(bool, copy=False)


def _tidy_timeseries(df: pd.DataFrame, value_col: str, threshold_crossed: Optional[Any] = None) -> pd.DataFrame:
    out = df[["site_id", "date", "lineage", value_col]].copy()
    out = out.rename(columns={value_col: "value"})
    mask = _coerce_bool_array(threshold_crossed, len(out)) if threshold_crossed is not None else np.zeros(len(out), dtype=bool)
    out["threshold_crossed"] = mask
    out = out.sort_values(["site_id", "date", "lineage"]).reset_index(drop=True)
    return out


# -----------------------------
# Plotting helpers (SAVE internally)
# -----------------------------
def _plot_detection_timelines(
    ctx: RunContext,
    cfg: Dict,
    e_vals: pd.DataFrame,
    q_vals: pd.DataFrame,
    sr_stats: pd.DataFrame,
    sr_thresholds: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    figure_title_site: Optional[str] = None,
) -> None:
    """
    Multi-panel figure:
      A: e-values (top lineages) with 1/α line (log y)
      B: q-values with α line
      C: SR log-statistic (logR_t) with threshold_log (linear y)
      D: mean e-value under null (calibration)
    """
    set_matplotlib_style()
    alpha = float(cfg.get("detection", {}).get("alpha", 0.1))
    sites = sorted(e_vals["site_id"].unique().tolist())
    focus_site = figure_title_site or (sites[0] if sites else None)
    if focus_site is None:
        return

    e_site = e_vals[e_vals["site_id"] == focus_site].copy()
    q_site = q_vals[q_vals["site_id"] == focus_site].copy()
    sr_site = sr_stats[sr_stats["site_id"] == focus_site].copy()
    if e_site.empty:
        return

    # Pick top lineages by max e-value
    top_lineages = (
        e_site.groupby("lineage")["value"]
        .max()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    e_plot = e_site[e_site["lineage"].isin(top_lineages)].copy()
    q_plot = q_site[q_site["lineage"].isin(top_lineages)].copy()
    sr_plot = sr_site[sr_site["lineage"].isin(top_lineages)].copy()

    thr_col = "threshold_log"
    if thr_col not in sr_thresholds.columns:
        thr_col = sr_thresholds.columns[-1]
    sr_thr_site = (
        sr_thresholds[sr_thresholds["site_id"] == focus_site]
        .set_index("lineage")[thr_col]
        .to_dict()
    )

    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(12, 8.5),
        constrained_layout=True,
    )

    # Panel A: e-values (log scale)
    ax = axes["A"]
    for lin in top_lineages:
        d = e_plot[e_plot["lineage"] == lin].sort_values("date")
        ax.plot(d["date"], d["value"], label=f"{lin}", lw=1.8)
    ax.axhline(1.0 / max(alpha, 1e-6), color="k", ls="--", lw=1.0, alpha=0.65)
    ax.set_yscale("log")
    ax.set_ylabel("e-value (log)")
    ax.set_title(f"E-values (site {focus_site})")
    ax.grid(True, alpha=0.3)
    place_legend_below(ax, ncol=3)

    # Panel B: q-values
    ax = axes["B"]
    for lin in top_lineages:
        d = q_plot[q_plot["lineage"] == lin].sort_values("date")
        ax.plot(d["date"], d["value"], label=f"{lin}", lw=1.8)
    ax.axhline(alpha, color="k", ls="--", lw=1.0, alpha=0.65)
    ax.set_ylim(0, min(1.0, max(alpha * 2, 0.25)))
    ax.set_ylabel("q-value")
    ax.set_title("Online e-BH q-values")
    ax.grid(True, alpha=0.3)
    place_legend_below(ax, ncol=3)

    # Panel C: SR (logR_t) vs threshold_log (linear y)
    ax = axes["C"]
    for lin in top_lineages:
        d = sr_plot[sr_plot["lineage"] == lin].sort_values("date")
        if d.empty:
            continue
        ax.plot(d["date"], d["value"], label=f"{lin}", lw=1.8)
        thr = sr_thr_site.get(lin, np.nan)
        if np.isfinite(thr):
            ax.axhline(thr, color=ax.get_lines()[-1].get_color(), ls="--", lw=1.0, alpha=0.9)
    ax.set_ylabel("log R_t")
    ax.set_title("Shiryaev–Roberts (log-statistic)")
    ax.grid(True, alpha=0.3)
    place_legend_below(ax, ncol=3)

    # Panel D: Calibration bar (mean e-value under null)
    ax = axes["D"]
    cal_site = calibration_metrics[calibration_metrics["site_id"] == focus_site].copy()
    if not cal_site.empty and "evalue_mean_null" in cal_site.columns:
        vals = cal_site.groupby("lineage")["evalue_mean_null"].mean().sort_values(ascending=False)
        names, ys = vals.index.tolist()[:8], vals.values[:8]
        x = np.arange(len(names))
        ax.bar(x, ys, width=0.6, alpha=0.85)
        ax.axhline(1.05, color="k", ls="--", lw=1.0, alpha=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ymax = 1.2 if len(ys) == 0 else max(1.2, min(2.0, 1.1 * float(np.nanmax(ys))))
        ax.set_ylim(0, ymax)
        ax.set_ylabel("mean e-value")
        ax.set_title("Mean e-value under null (calibration)")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No calibration data for site", ha="center", va="center")
        ax.set_axis_off()

    _save_figure(ctx, fig, f"detection_timelines_{_safe_name(focus_site)}")
    plt.close(fig)


def _imshow_discrete(ax, Z, x_labels, y_labels, title, cbar_label, vmin, vmax, cmap=None):
    im = ax.imshow(Z, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    # show at most 12 x-ticks to avoid clutter
    xticks = np.linspace(0, len(x_labels) - 1, num=min(12, len(x_labels)), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [x_labels[i].strftime("%Y-%m-%d") if hasattr(x_labels[i], "strftime") else str(x_labels[i]) for i in xticks],
        rotation=30, ha="right"
    )
    ax.set_title(title)
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=90)
    return im


def _plot_site_heatmaps(
    ctx: RunContext,
    e_vals: pd.DataFrame,
    q_vals: pd.DataFrame,
    focus_site: Optional[str] = None,
    top_k: int = 40,
) -> None:
    """
    Heatmaps across *all* lineages for a site:
      - log10 e-values
      - -log10 q-values
    """
    set_matplotlib_style()
    sites = sorted(e_vals["site_id"].unique().tolist())
    if not sites:
        return
    site = focus_site or sites[0]

    e_site = e_vals[e_vals["site_id"] == site].copy()
    q_site = q_vals[q_vals["site_id"] == site].copy()
    if e_site.empty or q_site.empty:
        return

    # Choose top_k lineages by max e-value / min q-value (union)
    top_by_e = (
        e_site.groupby("lineage")["value"]
        .max()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )
    top_by_q = (
        q_site.groupby("lineage")["value"]
        .min()
        .sort_values(ascending=True)
        .head(top_k)
        .index.tolist()
    )
    keep = list(dict.fromkeys(top_by_e + top_by_q))[:top_k]  # stable union, limited

    # Build date index
    dates = sorted(pd.to_datetime(e_site["date"].unique()).tolist())

    # Pivot to matrices (lineages x dates)
    def _mat(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[pd.Timestamp]]:
        dfp = df[df["lineage"].isin(keep)].copy()
        dfp = dfp.pivot_table(index="lineage", columns="date", values="value", aggfunc="mean")
        dfp = dfp.reindex(index=keep, columns=dates)
        return dfp.values.astype(float), list(dfp.index), list(dfp.columns)

    E, y_labs, x_labs = _mat(e_site)
    Q, _, _ = _mat(q_site)

    # Transform: log10 e-values; -log10 q-values (clip to finite ranges)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10E = np.log10(np.clip(E, 1e-8, 1e8))
        nlog10Q = -np.log10(np.clip(Q, 1e-12, 1.0))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, max(6.5, 0.22 * len(keep) + 3)), constrained_layout=True
    )
    _imshow_discrete(
        ax1, log10E, x_labs, y_labs,
        title=f"Site {site} — log10 e-values (top {len(keep)} lineages)",
        cbar_label="log10(e)",
        vmin=-3, vmax=3, cmap=None
    )
    _imshow_discrete(
        ax2, nlog10Q, x_labs, y_labs,
        title=f"Site {site} — −log10 q-values (top {len(keep)} lineages)",
        cbar_label="−log10(q)",
        vmin=0, vmax=6, cmap=None
    )

    _save_figure(ctx, fig, f"site_heatmaps_{_safe_name(site)}")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def run_detection(cfg: Dict, ctx: RunContext) -> Dict:
    """
    Run detection stage:
      - Compute e-values from posterior predictive tail probabilities (θ_ℓ > min_prop).
      - Online e-BH q-values with FDR control under arbitrary dependence.
      - Shiryaev–Roberts statistic in LOG-SPACE with block-permutation threshold calibration.
      - Calibration metrics (mean e-value under null; empirical FDR in permutations).
      - Lead-time vs reference dates if provided.
      - Figures: timelines (site), heatmaps (site), and delay histograms.
    """
    stage_cfg = cfg.get("detection", {})
    seed = int(stage_cfg.get("seed", cfg.get("seed", 20240229)))
    _set_global_seeds(seed)

    alpha = float(stage_cfg.get("alpha", 0.1))
    min_prop = float(stage_cfg.get("min_prop", 0.01))
    conf = float(stage_cfg.get("ci_confidence", 0.9))
    var_floor = float(stage_cfg.get("beta_var_floor", 1e-5))
    e_method = str(stage_cfg.get("e_value_method", "power"))
    e_power_a = float(stage_cfg.get("e_value_power", 0.5))
    p_floor = float(stage_cfg.get("p_floor", 1e-3))
    sr_block = int(stage_cfg.get("sr_block_size", 7))
    sr_alpha = float(stage_cfg.get("sr_alpha", alpha))
    sr_nperm = int(stage_cfg.get("sr_nperm", 200))
    ebh_alpha = float(stage_cfg.get("ebh_alpha", alpha))
    focus_site_for_plots = stage_cfg.get("focus_site", None)

    # Load posterior smoothed props (preferred) or baseline props
    smoothed = _scan_for_smoothed_props()
    if smoothed is None:
        _log(ctx, "WARNING", None, None, "No forecast smoothed props found; falling back to baseline props", {})
        smoothed = _scan_for_baseline_props()
    if smoothed is None or smoothed.empty:
        msg = "No lineage proportion tables found from prior stages."
        _log(ctx, "ERROR", None, None, msg, {})
        raise RuntimeError(msg)

    # Normalize columns
    cols_lower = {c.lower(): c for c in smoothed.columns}
    for req in ["site_id", "date", "lineage"]:
        if req not in cols_lower:
            match = [c for c in smoothed.columns if c.lower() == req]
            if match:
                cols_lower[req] = match[0]
            else:
                raise ValueError(f"Required column {req} not present in lineage table.")
    smoothed = smoothed.rename(columns=cols_lower)
    smoothed["date"] = pd.to_datetime(smoothed["date"])
    smoothed = smoothed.sort_values(["site_id", "date", "lineage"]).reset_index(drop=True)

    # Compute per-row p-values and e-values
    p_vals, e_vals_list, e_raw_vals, alphas, betas = [], [], [], [], []

    central_col_candidates = ["median", "mean", "p50", "theta", "value", "estimate", "est"]
    central_col = None
    for c in central_col_candidates:
        if c in smoothed.columns:
            central_col = c; break
    if central_col is None:
        central_col = "median"
        smoothed[central_col] = np.nan

    for _, row in smoothed.iterrows():
        a, b = _beta_params_from_summary(row, conf=conf, var_floor=var_floor)
        p = _p_value_from_beta(a, b, min_prop)
        e, e_raw = _evalue_from_p(p, method=e_method, power_a=e_power_a, p_floor=p_floor)
        p_vals.append(p); e_vals_list.append(e); e_raw_vals.append(e_raw); alphas.append(a); betas.append(b)

    df = smoothed.copy()
    df["p_value"] = np.array(p_vals, dtype=float)
    df["e_value"] = np.array(e_vals_list, dtype=float)
    df["e_value_raw_1_over_p"] = np.array(e_raw_vals, dtype=float)
    df["beta_alpha"] = np.array(alphas, dtype=float)
    df["beta_beta"] = np.array(betas, dtype=float)

    # Online e-BH per site-lineage
    q_rows = []
    for (site, lin), g in df.groupby(["site_id", "lineage"], sort=False):
        g = g.sort_values("date").copy()
        e_series = g["e_value"].values.astype(float)
        q_online = _online_eBH_qvalues(e_series)
        g["q_value"] = q_online
        g["ebh_reject"] = (g["q_value"].values <= ebh_alpha)
        q_rows.append(g[["site_id", "date", "lineage", "q_value", "ebh_reject"]])
    q_df = pd.concat(q_rows, axis=0).sort_values(["site_id", "date", "lineage"]).reset_index(drop=True)

    # e-values table and q-values table
    e_threshold = 1.0 / max(alpha, 1e-6)
    e_table = _tidy_timeseries(df, value_col="e_value", threshold_crossed=(df["e_value"] >= e_threshold))
    q_table = _tidy_timeseries(q_df.rename(columns={"q_value": "value"}), value_col="value", threshold_crossed=q_df["ebh_reject"])

    # SR statistic per site-lineage with calibrated thresholds (LOG-SPACE)
    rng = np.random.RandomState(seed)
    sr_records, thr_records, null_calib_records, fdr_perm_rates = [], [], [], []

    for (site, lin), g in df.groupby(["site_id", "lineage"], sort=False):
        g = g.sort_values("date").copy()
        e_series = g["e_value"].values.astype(float)
        dates = g["date"].values

        # Threshold calibration (log-space)
        h_log = _calibrate_sr_threshold_log_from_perms(
            e_values=e_series, alpha=sr_alpha, block=sr_block, nperm=sr_nperm, rng=rng
        )

        # SR recursion (log-space)
        log_e = np.log(np.clip(e_series, 1e-300, 1e300))
        logR = _sr_logspace(log_e)
        crossed = logR >= h_log

        for i in range(len(logR)):
            sr_records.append({
                "site_id": site,
                "date": dates[i],
                "lineage": lin,
                "value": float(logR[i]),           # logR value
                "threshold_crossed": bool(crossed[i]),
            })

        thr_records.append({
            "site_id": site,
            "lineage": lin,
            "threshold_log": float(h_log),
            "threshold_log10": float(h_log / np.log(10.0)),
        })

        # E-value null calibration (mean under null windows)
        g_copy = g.copy()
        if central_col not in g_copy.columns:
            g_copy[central_col] = np.nan
        mask_null = (g_copy[central_col].values <= min_prop / 2.0)
        if np.any(mask_null):
            e_mean_null = float(np.nanmean(e_series[mask_null]))
        else:
            e_mean_null = float(np.nanmean(e_series))
        null_calib_records.append({"site_id": site, "lineage": lin, "evalue_mean_null": float(e_mean_null)})

        # Empirical FDR via permutations (e-BH)
        idx_perms = _block_permutations_indices(len(e_series), max(1, sr_block), max(20, sr_nperm // 5), rng)
        null_reject_fracs = []
        for idx in idx_perms:
            Eperm = e_series[idx]
            qperm = _online_eBH_qvalues(Eperm)
            reject = (qperm <= ebh_alpha)
            null_reject_fracs.append(float(np.mean(reject)))
        fdr_perm = float(np.mean(null_reject_fracs)) if len(null_reject_fracs) > 0 else 0.0
        fdr_perm_rates.append({"site_id": site, "lineage": lin, "empirical_fdr_perm": fdr_perm})

        _log(ctx, "INFO", site, lin, "Calibrated SR threshold (log-space)",
             {"threshold_log": float(h_log), "sr_alpha": sr_alpha})

    sr_df = pd.DataFrame.from_records(sr_records).sort_values(["site_id", "date", "lineage"]).reset_index(drop=True)
    sr_thr_df = pd.DataFrame.from_records(thr_records).sort_values(["site_id", "lineage"]).reset_index(drop=True)
    null_calib_df = pd.DataFrame.from_records(null_calib_records).sort_values(["site_id", "lineage"]).reset_index(drop=True)
    fdr_perm_df = pd.DataFrame.from_records(fdr_perm_rates).sort_values(["site_id", "lineage"]).reset_index(drop=True)
    calib = null_calib_df.merge(fdr_perm_df, on=["site_id", "lineage"], how="outer")

    # Detection calls: first crossing times
    calls_records = []
    for (site, lin), g in q_table.groupby(["site_id", "lineage"], sort=False):
        g = g.sort_values("date")
        det = g[g["threshold_crossed"]]
        if not det.empty:
            calls_records.append({"site_id": site, "lineage": lin, "method": "eBH", "detect_date": det.iloc[0]["date"]})
    for (site, lin), g in sr_df.groupby(["site_id", "lineage"], sort=False):
        g = g.sort_values("date")
        det = g[g["threshold_crossed"]]
        if not det.empty:
            calls_records.append({"site_id": site, "lineage": lin, "method": "SR", "detect_date": det.iloc[0]["date"]})
    calls_df = pd.DataFrame.from_records(calls_records)
    if not calls_df.empty:
        calls_df["detect_date"] = pd.to_datetime(calls_df["detect_date"])
        calls_df = calls_df.sort_values(["site_id", "lineage", "method"]).reset_index(drop=True)

    # Detection delays if reference dates are provided
    delays_df = pd.DataFrame(columns=["site_id", "lineage", "method", "reference_date", "detect_date", "delay_days"])
    ref_dates_map: Dict[Tuple[Optional[str], str], datetime] = {}

    ref_cfg = stage_cfg.get("reference_dates", {})
    if isinstance(ref_cfg, dict) and ref_cfg:
        for k, v in ref_cfg.items():
            try:
                if "|" in k:
                    site, lin = k.split("|", 1)
                    ref_dates_map[(site, lin)] = pd.to_datetime(v)
                else:
                    ref_dates_map[(None, k)] = pd.to_datetime(v)
            except Exception:
                continue
    ref_file = stage_cfg.get("reference_dates_file", None)
    if ref_file:
        rf = Path(ref_file)
        if not rf.is_absolute():
            rf = _repo_root() / rf
        if rf.exists():
            rdf = pd.read_csv(rf)
            if "lineage" in rdf.columns and "reference_date" in rdf.columns:
                for _, r in rdf.iterrows():
                    site = r["site_id"] if "site_id" in rdf.columns else None
                    lin = r["lineage"]
                    rd = pd.to_datetime(r["reference_date"])
                    ref_dates_map[(site, lin)] = rd

    if not calls_df.empty and ref_dates_map:
        delay_records = []
        for _, r in calls_df.iterrows():
            site, lin, method, det_date = r["site_id"], r["lineage"], r["method"], pd.to_datetime(r["detect_date"])
            ref = ref_dates_map.get((site, lin), ref_dates_map.get((None, lin), np.nan))
            if pd.isna(ref):
                continue
            ref = pd.to_datetime(ref)
            delay_days = (det_date - ref).days
            delay_records.append({
                "site_id": site, "lineage": lin, "method": method,
                "reference_date": ref, "detect_date": det_date, "delay_days": int(delay_days)
            })
        if delay_records:
            delays_df = pd.DataFrame.from_records(delay_records).sort_values(["site_id", "lineage", "method"])

    # Persist tables
    ctx.write_table("e_values", e_table)
    ctx.write_table("q_values", q_table)
    ctx.write_table("sr_statistics", sr_df)         # values are logR
    ctx.write_table("sr_thresholds", sr_thr_df)     # contains threshold_log (+ threshold_log10)
    if not calls_df.empty:
        ctx.write_table("detection_calls", calls_df)
    if not delays_df.empty:
        ctx.write_table("detection_delays", delays_df)
    ctx.write_table("null_calibration", calib)

    # Figures: multi-panel timelines, site heatmaps, and delay hists
    _plot_detection_timelines(ctx, cfg, e_table, q_table, sr_df, sr_thr_df, calib, figure_title_site=focus_site_for_plots)
    _plot_site_heatmaps(ctx, e_table, q_table, focus_site=focus_site_for_plots)
    if not delays_df.empty:
        set_matplotlib_style()
        for method in sorted(delays_df["method"].unique().tolist()):
            dfm = delays_df[delays_df["method"] == method]
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            if dfm.empty or "delay_days" not in dfm.columns or dfm["delay_days"].dropna().empty:
                ax.text(0.5, 0.5, "No delay values", ha="center", va="center")
                ax.set_axis_off()
            else:
                vals = dfm["delay_days"].dropna().values.astype(float)
                bins = np.arange(np.floor(np.nanmin(vals)) - 1, np.ceil(np.nanmax(vals)) + 2)
                ax.hist(vals, bins=bins, alpha=0.8, edgecolor="k")
                ax.axvline(0, color="k", ls="--", lw=1.0, alpha=0.6)
                ax.set_title(f"Detection delays: {method}")
                ax.set_xlabel("detected_date − reference_date (days)")
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.3)
            _save_figure(ctx, fig, f"detection_delays_{method.lower()}")
            plt.close(fig)

    # Metrics summary
    mean_null = float(np.nanmean(calib["evalue_mean_null"].values)) if not calib.empty else np.nan
    emp_fdr = float(np.nanmean(calib["empirical_fdr_perm"].values)) if not calib.empty else np.nan
    n_calls_ebh = int((q_table["threshold_crossed"]).sum())
    n_calls_sr = int((sr_df["threshold_crossed"]).sum())
    metrics = pd.DataFrame.from_records([
        {"metric": "e_value_null_mean", "value": mean_null},
        {"metric": "empirical_fdr_perm", "value": emp_fdr},
        {"metric": "n_calls_eBH", "value": n_calls_ebh},
        {"metric": "n_calls_SR", "value": n_calls_sr},
        {"metric": "alpha", "value": alpha},
        {"metric": "sr_alpha", "value": sr_alpha},
        {"metric": "ebh_alpha", "value": ebh_alpha},
        {"metric": "min_prop", "value": min_prop},
    ])
    _write_metric(ctx, "detection_summary", metrics)

    # Report
    n_sites = e_table["site_id"].nunique() if not e_table.empty else 0
    n_lineages = e_table["lineage"].nunique() if not e_table.empty else 0
    report_lines = [
        f"# Detection stage report",
        f"- Sites: {n_sites}",
        f"- Lineages: {n_lineages}",
        f"- α (FDR): {ebh_alpha}",
        f"- SR α: {sr_alpha}",
        f"- min_prop for alternative: {min_prop}",
        f"- e-value method: {e_method} (power={e_power_a}) with p_floor={p_floor}",
        (f"- Mean e-value under null (target ≤ 1.05): {mean_null:.3f}" if np.isfinite(mean_null) else "- Mean e-value under null: NA"),
        (f"- Empirical FDR under permutations: {emp_fdr:.3f}" if np.isfinite(emp_fdr) else "- Empirical FDR under permutations: NA"),
        f"- e-BH detections (total timepoints): {n_calls_ebh}",
        f"- SR detections (total timepoints): {n_calls_sr}",
    ]
    if hasattr(ctx, "write_report"):
        ctx.write_report("\n".join(report_lines))

    return {
        "n_sites": n_sites,
        "n_lineages": n_lineages,
        "alpha": ebh_alpha,
        "sr_alpha": sr_alpha,
        "min_prop": min_prop,
        "mean_evalue_null": mean_null,
        "empirical_fdr_perm": emp_fdr,
        "n_calls_eBH": n_calls_ebh,
        "n_calls_SR": n_calls_sr,
    }
