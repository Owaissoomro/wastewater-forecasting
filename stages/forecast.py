# stages/forecast.py
# Forecast stage with Auxiliary Particle Filter + Backward Simulation Smoother.
# Ensures ALL priors mutations are used by expanding S with a GLOBAL lineage.
# Posterior predictive draws are vectorized and calibrated (PIT, WAIC).

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from typing import Dict, Any, Tuple, List, Optional, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import betabinom

from utils.plotting import set_matplotlib_style, place_legend_below
from utils.run import RunContext
from utils.metrics import rmse
from utils.logging import utcnow_iso

try:
    from utils.seeds import set_global_seeds
except Exception:
    import random
    def set_global_seeds(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)

# ------------------------- Numerics & palette -------------------------
EPS = 1e-12
SB_DEEP = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#64B5CD",
    "#CCB974", "#8C8C8C", "#E17C05", "#1B9E77", "#D95F02"
]

# ------------------------- Distributions & math -------------------------

def _log_beta_binom_pmf(y: np.ndarray, n: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Vectorized log PMF of Beta–Binomial under (mu, phi) parameterization."""
    y = np.asarray(y, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), EPS, 1.0 - EPS)
    phi = np.clip(np.asarray(phi, dtype=np.float64), 1.0, np.inf)
    a = mu * phi
    b = (1.0 - mu) * phi
    logc = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
    logp = (
        logc
        + (gammaln(y + a) - gammaln(a))
        + (gammaln(n - y + b) - gammaln(b))
        + (gammaln(a + b) - gammaln(n + a + b))
    )
    zero_trials = (n <= 0)
    logp = np.where(zero_trials & (y == 0), 0.0, logp)
    logp = np.where(zero_trials & (y != 0), -np.inf, logp)
    return logp


def _dirichlet_sample(pi: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet sample with alpha = kappa*pi, guarding zeros."""
    pi = np.asarray(pi, dtype=np.float64)
    pi = np.clip(pi, EPS, np.inf)
    pi = pi / pi.sum()
    alpha = np.clip(kappa * pi, 1e-8, np.inf)
    return rng.dirichlet(alpha)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling. Returns ancestor indices (length N)."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    idx = np.zeros(N, dtype=np.int64)
    i = 0; j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            idx[i] = j; i += 1
        else:
            j += 1
            if j >= N:
                j = N - 1
    return idx


def _pi_selection(theta: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Selection-adjusted target frequencies proportional to theta * (1 + s)^+."""
    theta = np.asarray(theta, dtype=np.float64)
    sel = np.maximum(1.0 + np.asarray(s, dtype=np.float64), EPS)
    f = theta * sel
    Z = np.sum(f)
    if not np.isfinite(Z) or Z <= 0.0:
        return np.full_like(theta, 1.0 / len(theta))
    return f / Z


def _compute_log_weights_for_particles(
    thetas_t: np.ndarray,  # (P, L)
    y_t: np.ndarray,       # (M,)
    n_t: np.ndarray,       # (M,)
    S: np.ndarray,         # (M, L)
    phi: np.ndarray,       # (M,)
) -> np.ndarray:
    """Stable per-particle log weight using Beta–Binomial emission: sum_m log p(y_t,m | theta_t)."""
    mu = np.clip(thetas_t @ S.T, EPS, 1.0 - EPS)  # (P, M)
    y_vec = y_t.astype(np.int64)
    n_vec = n_t.astype(np.int64)
    a = mu * phi  # (P, M)
    b = (1.0 - mu) * phi
    logc = (gammaln(n_vec + 1.0) - gammaln(y_vec + 1.0) - gammaln(n_vec - y_vec + 1.0)).astype(float)  # (M,)
    logpmf = (
        logc[None, :]
        + (gammaln(y_vec[None, :] + a) - gammaln(a))
        + (gammaln((n_vec - y_vec)[None, :] + b) - gammaln(b))
        + (gammaln(a + b) - gammaln(n_vec[None, :] + a + b))
    )  # (P, M)
    zero_trials = (n_vec <= 0)
    if np.any(zero_trials):
        mask_ok = zero_trials & (y_vec == 0)
        logpmf[:, zero_trials] = -np.inf
        if np.any(mask_ok):
            logpmf[:, mask_ok] = 0.0
    return np.sum(logpmf, axis=1)  # (P,)


def _ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    denom = np.sum(w * w)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return 1.0 / denom


def _normalize_weights(logw: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize log-weights; also return log-evidence estimate via log-mean-exp."""
    lse = logsumexp(logw)
    w = np.exp(logw - lse)
    logZ = lse - np.log(len(logw))
    return w, logZ


def _dirichlet_logpdf(x: np.ndarray, alpha: np.ndarray) -> float:
    """Log density of Dir(alpha) at x (both 1D), numerically safe."""
    x = np.clip(x, EPS, 1.0); x = x / x.sum()
    alpha = np.clip(alpha, 1e-12, np.inf)
    return float(gammaln(alpha.sum()) - np.sum(gammaln(alpha)) + np.sum((alpha - 1.0) * np.log(x)))

# -------------------- APF step (look-ahead + correct) --------------------

def _apf_step(
    particles_tm1: np.ndarray,          # (P, L)
    w_tm1: np.ndarray,                  # (P,) normalized
    y_t: np.ndarray, n_t: np.ndarray,   # (M,), (M,)
    S: np.ndarray, phi: np.ndarray, s_vec: np.ndarray,
    kappa: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One Auxiliary Particle Filter step:
      1) Compute look-ahead predictive weights m_i = p(y_t | E[theta_t | theta_{t-1}^i]).
      2) Resample ancestors ∝ w_{t-1,i} * m_i (systematic).
      3) Propagate theta_t^j ~ Dir(kappa * pi(theta_{t-1}^{a_j})).
      4) Correct weights: w_t^j ∝ p(y_t | theta_t^j) / m_{a_j}.
    Returns: (particles_t, logw_t, ancestor_idx)
    """
    P, L = particles_tm1.shape

    # 1) Look-ahead
    pi_pred = np.apply_along_axis(_pi_selection, 1, particles_tm1, s_vec)  # (P, L)
    mu_pred = np.clip(pi_pred @ S.T, EPS, 1.0 - EPS)                        # (P, M)
    y = y_t.astype(int); n = n_t.astype(int)
    a = mu_pred * phi; b = (1.0 - mu_pred) * phi
    logc = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)       # (M,)
    ll_pred = (logc[None,:] + (gammaln(y[None,:]+a) - gammaln(a))
               + (gammaln((n - y)[None,:] + b) - gammaln(b))
               + (gammaln(a + b) - gammaln(n[None,:] + a + b)))             # (P, M)
    zero_trials = (n <= 0)
    if np.any(zero_trials):
        mask_ok = zero_trials & (y == 0)
        ll_pred[:, zero_trials] = -np.inf
        if np.any(mask_ok): ll_pred[:, mask_ok] = 0.0
    log_m = np.sum(ll_pred, axis=1)                                         # (P,)

    # 2) Ancestor sampling ∝ w_tm1 * m_i
    log_aw = np.log(np.clip(w_tm1, EPS, 1.0)) + log_m
    log_aw -= logsumexp(log_aw)
    w_aw = np.exp(log_aw)
    anc_idx = _systematic_resample(w_aw, rng)

    ancestors = particles_tm1[anc_idx]         # (P, L)
    m_anc = log_m[anc_idx]                     # (P,)

    # 3) Propagate from prior q(θ_t | ancestor) = Dir(kappa * pi(ancestor))
    particles_t = np.zeros_like(ancestors)
    for p in range(P):
        pi = _pi_selection(ancestors[p], s_vec)
        particles_t[p] = _dirichlet_sample(pi, kappa, rng)

    # 4) Correct weights: log w_t = log p(y_t | θ_t) - log m_{ancestor}
    loglik_t = _compute_log_weights_for_particles(particles_t, y_t, n_t, S, phi)  # (P,)
    logw_t = loglik_t - m_anc
    return particles_t, logw_t, anc_idx


def _backward_simulation_smoother(
    particles_hist: List[np.ndarray],     # list of (P, L), t=0..T-1
    logw_hist: List[np.ndarray],          # list of (P,)
    s_vec: np.ndarray, kappa: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Godsill–Doucet–West backward simulator for one smoothed trajectory.
    p(i_t | i_{t+1}) ∝ w_t(i_t) * p(θ_{t+1}^{i_{t+1}} | θ_t^{i_t})
    Return θ_{1:T}^* as (T, L).
    """
    T = len(particles_hist)
    P, L = particles_hist[-1].shape
    theta_path = np.zeros((T, L), dtype=np.float64)

    # Sample index at T from w_T via Gumbel-max
    lwT = logw_hist[-1] - logsumexp(logw_hist[-1])
    g = -np.log(-np.log(np.clip(rng.random(P), EPS, 1.0 - EPS)))
    idx_t = int(np.argmax(lwT + g))
    theta_path[T-1] = particles_hist[-1][idx_t]

    # Backward steps
    for t in range(T-2, -1, -1):
        theta_next = theta_path[t+1]
        logprob = np.empty(P, dtype=np.float64)
        lw = logw_hist[t] - logsumexp(logw_hist[t])
        # Transition density Dir(kappa * pi(θ_t)) evaluated at theta_next
        for i in range(P):
            pi = _pi_selection(particles_hist[t][i], s_vec)
            alpha = kappa * pi
            logprob[i] = lw[i] + _dirichlet_logpdf(theta_next, alpha)
        lw_norm = logprob - logsumexp(logprob)
        g = -np.log(-np.log(np.clip(rng.random(P), EPS, 1.0 - EPS)))
        i_star = int(np.argmax(lw_norm + g))
        theta_path[t] = particles_hist[t][i_star]

    return theta_path

# -------------------- S builder: guarantee all priors mutations --------------------

def _ensure_S_covers_target_mutations(S_df: pd.DataFrame, target_mutations: pd.Index) -> pd.DataFrame:
    """
    Ensure S has rows for every mutation in `target_mutations`.
    Add/keep GLOBAL column; set GLOBAL=1.0 for missing or all-zero rows.
    Reindex to target_mutations (sorted), sort columns.
    """
    S_df = S_df.copy()
    # Add missing rows
    missing = target_mutations.difference(S_df.index)
    if len(missing):
        if "GLOBAL" not in S_df.columns:
            S_df["GLOBAL"] = 0.0
        add = pd.DataFrame(0.0, index=missing, columns=S_df.columns)
        add["GLOBAL"] = 1.0
        S_df = pd.concat([S_df, add], axis=0)
    # Fix all-zero rows
    row_max = S_df.max(axis=1)
    zero_rows = row_max[row_max <= 0.0].index
    if len(zero_rows):
        if "GLOBAL" not in S_df.columns:
            S_df["GLOBAL"] = 0.0
        S_df.loc[zero_rows, "GLOBAL"] = 1.0
    # Reindex and sort
    S_df = S_df.reindex(index=target_mutations, fill_value=0.0)
    S_df = S_df.sort_index(axis=0).sort_index(axis=1)
    return S_df

# ------------------------- Plot helpers & analytics -------------------------

def _format_date_axis(ax: plt.Axes) -> None:
    locator = AutoDateLocator(minticks=3, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _color_map(keys: Iterable[str]) -> Dict[str, str]:
    keys = list(keys)
    cdict = {}
    for i, k in enumerate(keys):
        cdict[k] = SB_DEEP[i % len(SB_DEEP)]
    return cdict


def _stacked_area(ax: plt.Axes, dates: np.ndarray, series_dict: Dict[str, np.ndarray], others: Optional[np.ndarray] = None) -> None:
    labels = list(series_dict.keys())
    values = [series_dict[l] for l in labels]
    colors = [_color_map(labels)[l] for l in labels]
    ax.stackplot(dates, values, labels=labels, colors=colors, alpha=0.85, step="mid")
    if others is not None:
        ax.plot(dates, others, lw=1.5, color="#000000", alpha=0.6, label="Others (sum)")
    ax.set_ylim(0.0, 1.0)
    _format_date_axis(ax)
    ax.set_ylabel("Proportion")
    ax.grid(True, alpha=0.25)
    place_legend_below(ax, ncol=min(4, len(labels)))


def _alt_ref_scatter(ax: plt.Axes, y_list: List[np.ndarray], n_list: List[np.ndarray], sample_max: int = 5000) -> None:
    ys = np.concatenate([y.astype(float) for y in y_list], axis=0)
    ns = np.concatenate([n.astype(float) for n in n_list], axis=0)
    refs = np.maximum(ns - ys, 0.0)
    if ys.size > sample_max:
        rng = np.random.default_rng(123)
        idx = rng.choice(np.arange(ys.size), size=sample_max, replace=False)
        ys = ys[idx]; refs = refs[idx]
    ax.scatter(np.maximum(refs, 0.5), np.maximum(ys, 0.5), s=8, alpha=0.35)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Reference count (log)")
    ax.set_ylabel("Alternate count (log)")
    ax.grid(True, which="both", alpha=0.25)
    if ys.size > 1:
        r = np.corrcoef(np.log1p(refs), np.log1p(ys))[0, 1]
        ax.set_title(f"Alt vs Ref (ρ≈{r:.2f})")


def _pred_vs_obs_counts(ax: plt.Axes, qq_df: pd.DataFrame) -> None:
    if qq_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    x = qq_df["pred"].values + EPS
    y = qq_df["obs"].values + EPS
    ax.scatter(x, y, s=12, alpha=0.5)
    lo = max(min(x.min(), y.min()), 1e-3)
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], lw=1.0, alpha=0.7, color="#000000")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Predicted (median, log)")
    ax.set_ylabel("Observed (log)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Predicted vs observed counts")


def _compute_obs_pred_af(
    y_list: List[np.ndarray],
    n_list: List[np.ndarray],
    theta_path: np.ndarray,     # (T,L)
    S: np.ndarray,              # (M,L)
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (obs_af[T,M], pred_af[T,M]); NaN in obs_af where n==0."""
    T = len(y_list)
    M = S.shape[0]
    obs_af = np.full((T, M), np.nan, dtype=float)
    for t in range(T):
        y = y_list[t].astype(float)
        n = np.maximum(n_list[t].astype(float), 0.0)
        af = np.where(n > 0, y / np.maximum(n, 1.0), np.nan)
        obs_af[t, :] = np.clip(af, 0.0, 1.0)
    pred_af = np.clip(theta_path @ S.T, 0.0, 1.0)
    return obs_af, pred_af


def _top_k_by_variability(arr_2d: np.ndarray, names: List[str], k: int) -> List[str]:
    if arr_2d.size == 0:
        return []
    v = np.nanvar(arr_2d, axis=0)
    k = min(k, arr_2d.shape[1])
    idx = np.argsort(-v)[:k]
    return [names[i] for i in idx]


def _mutation_timeseries_2x2_figure(
    site_id: str,
    dates: np.ndarray,
    mutations: List[str],
    obs_af: np.ndarray,   # (T,M)
    pred_af: np.ndarray,  # (T,M)
    top4: Optional[List[str]] = None
) -> plt.Figure:
    set_matplotlib_style()
    if top4 is None or len(top4) == 0:
        top4 = _top_k_by_variability(obs_af, mutations, 4)
    idx_map = {m: i for i, m in enumerate(mutations)}
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    for j, mut in enumerate(top4[:4]):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        mi = idx_map[mut]
        y = obs_af[:, mi]
        p = pred_af[:, mi]
        ax.scatter(dates, y, s=14, alpha=0.6, label="Observed AF")
        ax.plot(dates, p, lw=2.0, alpha=0.9, label="Predicted AF")
        ax.set_ylim(-0.02, 1.02)
        _format_date_axis(ax)
        ax.grid(True, alpha=0.25)
        ax.set_title(mut)
        if j // 2 == 1:
            ax.set_xlabel("Date")
        ax.set_ylabel("Allele frequency")
        if j == 0:
            place_legend_below(ax, ncol=2)
    fig.suptitle(f"Top-4 mutation AF time series – site {site_id}", y=0.98)
    return fig


def _mutation_heatmap_figure(
    site_id: str,
    dates: np.ndarray,
    mutations: List[str],
    obs_af: np.ndarray,   # (T,M)
    k: int = 40
) -> plt.Figure:
    set_matplotlib_style()
    topk = _top_k_by_variability(obs_af, mutations, k)
    idx_map = {m: i for i, m in enumerate(mutations)}
    mat = np.vstack([obs_af[:, idx_map[m]] for m in topk]).T  # (T, k)
    fig, ax = plt.subplots(figsize=(1.2 + 0.22 * len(topk), 0.9 + 0.25 * len(dates)))
    im = ax.imshow(mat.T, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="magma_r")
    ax.set_yticks(np.arange(len(topk)))
    ax.set_yticklabels(topk, fontsize=8)
    show_idx = np.linspace(0, len(dates)-1, num=min(16, len(dates)), dtype=int)
    xlabels = np.array([pd.to_datetime(d).date().isoformat() for d in dates], dtype=object)
    ax.set_xticks(show_idx); ax.set_xticklabels(xlabels[show_idx], rotation=90, fontsize=7)
    ax.set_xlabel("Date")
    ax.set_title(f"Observed AF heatmap (top-{len(topk)} mutations) – site {site_id}")
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Observed AF")
    fig.tight_layout()
    return fig


def _evolution_2x2_panel(
    site_id: str,
    dates: np.ndarray,
    lineages: List[str],
    traj_site: pd.DataFrame,
    y_list: List[np.ndarray],
    n_list: List[np.ndarray],
    qq_site: pd.DataFrame,
    topK: int = 6
) -> plt.Figure:
    set_matplotlib_style()
    fig = plt.figure(figsize=(11.5, 8.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # (A) Trajectories
    med_by_lin = traj_site.groupby("lineage")["median"].mean().sort_values(ascending=False)
    tops = list(med_by_lin.head(topK).index)
    cmap = _color_map(tops)
    for lin in tops:
        df_lin = traj_site[traj_site["lineage"] == lin].sort_values("date")
        axA.plot(df_lin["date"], df_lin["median"], label=lin, lw=1.8, color=cmap[lin])
        axA.fill_between(df_lin["date"], df_lin["q05"], df_lin["q95"], alpha=0.18, color=cmap[lin])
        # Optional MAP curve if present
        if "map" in df_lin.columns and df_lin["map"].notna().any():
            axA.plot(df_lin["date"], df_lin["map"], linestyle="--", lw=1.2, alpha=0.9, color=cmap[lin])
    _format_date_axis(axA)
    axA.set_ylim(-0.02, 1.02)
    axA.set_title(f"Lineage trajectories – site {site_id}")
    axA.set_xlabel("Date"); axA.set_ylabel("Proportion")
    axA.grid(True, alpha=0.3)
    place_legend_below(axA, ncol=min(3, len(tops)))

    # (B) Stacked area (topK + Others)
    series = {}
    for lin in tops:
        df_lin = traj_site[traj_site["lineage"] == lin].sort_values("date")
        series[lin] = df_lin["median"].values
    sum_top = np.sum(np.vstack([series[k] for k in series]), axis=0) if series else np.zeros(len(dates))
    others = np.clip(1.0 - sum_top, 0.0, 1.0)
    _stacked_area(axB, dates, series, others)
    axB.set_title("Evolution (stacked area) – top lineages")

    # (C) Alt vs Ref scatter
    _alt_ref_scatter(axC, y_list, n_list)

    # (D) Predicted vs observed counts
    _pred_vs_obs_counts(axD, qq_site)

    fig.suptitle(f"Per-site diagnostics (2×2) – {site_id}", y=0.99)
    return fig


def _shannon_entropy(theta_row: np.ndarray) -> float:
    p = np.clip(theta_row, EPS, 1.0); p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p / np.sum(p), EPS, 1.0)
    q = np.clip(q / np.sum(q), EPS, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def _turnover_metrics(dates: np.ndarray, theta_path: np.ndarray) -> pd.DataFrame:
    T, L = theta_path.shape
    H = np.array([_shannon_entropy(theta_path[t]) for t in range(T)], dtype=float)
    J = np.zeros(T, dtype=float)
    for t in range(1, T):
        J[t] = _jsd(theta_path[t - 1], theta_path[t])
    return pd.DataFrame({"date": pd.to_datetime(dates), "H": H, "JSD_prev": J})


def _turnover_figure(site_id: str, df_turn: pd.DataFrame) -> plt.Figure:
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(df_turn["date"], df_turn["H"], lw=1.8, label="Shannon H")
    ax2 = ax.twinx()
    ax2.plot(df_turn["date"], df_turn["JSD_prev"], lw=1.5, linestyle="--", alpha=0.9, label="JSD vs previous", color="#C44E52")
    _format_date_axis(ax)
    ax.set_ylabel("Shannon entropy H")
    ax2.set_ylabel("JSD")
    ax.set_title(f"Lineage turnover metrics – site {site_id}")
    ax.grid(True, alpha=0.3)
    # Combine legends across both axes safely using a figure-level legend.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()
    return fig

# ------------------------------ Stage runner ------------------------------

def run_forecast(cfg: Dict[str, Any], ctx: RunContext) -> None:
    """
    Stage: Wright–Fisher state-space + Auxiliary Particle Filter + Backward Simulation Smoother
           + Posterior Predictive.
    Guarantees: uses EXACT mutation set from priors_hyperparams (e.g., all 62 SNVs).
    """
    stage_name = "forecast"
    set_matplotlib_style()

    seed = int(cfg.get("forecast", {}).get("seed", 1234))
    set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    forecast_cfg = cfg.get("forecast", {})
    inputs_cfg = forecast_cfg.get("inputs", {})
    pf_cfg = forecast_cfg.get("pf", {})
    smoothing_cfg = forecast_cfg.get("smoothing", {})
    sel_cfg = forecast_cfg.get("selection", {})
    pp_cfg = forecast_cfg.get("posterior_predictive", {})
    filter_cfg = forecast_cfg.get("filter", {})
    tplus1_cfg = forecast_cfg.get("tplus1", {})

    # Optional compute caps for calibration summaries
    waic_max_particles = int(pp_cfg.get("waic_max_particles", 128))
    pit_max_particles  = int(pp_cfg.get("pit_max_particles", 128))

    # Inputs
    snv_counts_path = pathlib.Path(inputs_cfg.get("snv_counts_path", "results/preprocessing/tables/feature_store_snv.csv"))
    signatures_path = pathlib.Path(inputs_cfg.get("signatures_path", "data/signatures.csv"))
    priors_hyperparams_path = pathlib.Path(inputs_cfg.get("priors_hyperparams_path", "results/priors/tables/priors_hyperparams.csv"))
    baseline_theta_path = inputs_cfg.get("baseline_theta_path", None)
    baseline_theta_path = pathlib.Path(baseline_theta_path) if baseline_theta_path else None

    # Load counts
    df_counts = pd.read_csv(snv_counts_path)
    required_cols = {"sample_id", "site_id", "date", "mutation", "count", "coverage"}
    missing_cols = required_cols - set(df_counts.columns)
    if missing_cols:
        raise ValueError(f"Counts table missing required columns: {missing_cols}")
    df_counts["date"] = pd.to_datetime(df_counts["date"])
    df_counts = df_counts[df_counts["coverage"] >= 0].copy()
    df_counts["mutation"] = df_counts["mutation"].astype(str)

    # Load priors hyperparams -> authoritative mutations set
    df_priors = pd.read_csv(priors_hyperparams_path)
    if "mutation" not in df_priors.columns:
        raise ValueError("Priors hyperparams must include 'mutation' column")
    df_priors["mutation"] = df_priors["mutation"].astype(str)
    mutations_target = pd.Index(df_priors["mutation"].astype(str).unique()).sort_values()  # e.g., 62 SNVs

    # Restrict counts to priors mutation set
    df_counts = df_counts[df_counts["mutation"].isin(mutations_target)].copy()

    # Optional site/lineage filters
    sites_subset = filter_cfg.get("sites", None)
    if sites_subset:
        df_counts = df_counts[df_counts["site_id"].isin(sites_subset)].copy()

    # Load signatures and build S on ALL priors mutations
    df_sign = pd.read_csv(signatures_path)
    if not {"mutation", "lineage", "weight"}.issubset(df_sign.columns):
        raise ValueError("Signatures table must have columns: mutation,lineage,weight")
    df_sign["mutation"] = df_sign["mutation"].astype(str)
    df_sign["lineage"]  = df_sign["lineage"].astype(str)
    df_sign["weight"]   = pd.to_numeric(df_sign["weight"], errors="coerce").fillna(0.0).astype(float)

    lineage_subset = filter_cfg.get("lineages", None)
    if lineage_subset:
        df_sign = df_sign[df_sign["lineage"].isin(lineage_subset)].copy()

    sig_keep = df_sign[df_sign["mutation"].isin(mutations_target)].copy()
    S_df = sig_keep.pivot_table(index="mutation", columns="lineage", values="weight", aggfunc="mean", fill_value=0.0)
    S_df = _ensure_S_covers_target_mutations(S_df, mutations_target)
    if "GLOBAL" not in S_df.columns:
        S_df["GLOBAL"] = 0.0
    S_df = S_df.sort_index(axis=0).sort_index(axis=1)

    mutations = S_df.index.to_list()
    lineages  = S_df.columns.to_list()
    S = S_df.values.astype(np.float64)  # (M, L)
    M, L = S.shape

    # Condition of S: identifiability diagnostic
    try:
        u, svals, vh = np.linalg.svd(S, full_matrices=False)
        cond = (svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
        rank = int((svals > 1e-10).sum())
    except Exception:
        cond, rank = np.nan, np.nan

    # Priors hyperparams -> phi per mutation (aligned)
    df_phi = df_priors[["mutation"]].copy()
    if "phi" in df_priors.columns:
        df_phi["phi"] = df_priors["phi"].astype(float)
    elif {"alpha", "beta"}.issubset(df_priors.columns):
        df_phi["phi"] = (df_priors["alpha"].astype(float) + df_priors["beta"].astype(float)).values
    else:
        df_phi["phi"] = 200.0
    phi_series = df_phi.set_index("mutation").reindex(mutations)["phi"].fillna(df_phi["phi"].median())
    phi = np.clip(phi_series.values.astype(np.float64), 1.0, np.inf)

    # Selection vector
    s_default = float(sel_cfg.get("default", 0.0))
    s_by_lineage = sel_cfg.get("by_lineage", {})
    s_vec = np.array([float(s_by_lineage.get(lin, s_default)) for lin in lineages], dtype=np.float64)

    # Baseline theta (optional)
    theta_baseline: Optional[pd.DataFrame] = None
    if baseline_theta_path is not None and baseline_theta_path.exists():
        tb = pd.read_csv(baseline_theta_path)
        tb["date"] = pd.to_datetime(tb["date"])
        val_col = None
        for c in ["theta", "prop", "value", "estimate"]:
            if c in tb.columns:
                val_col = c; break
        if val_col is None: val_col = tb.columns[-1]
        theta_baseline = tb.rename(columns={val_col: "theta"})[["site_id", "date", "lineage", "theta"]]
        theta_baseline["lineage"] = theta_baseline["lineage"].astype(str)
        theta_baseline = theta_baseline[theta_baseline["lineage"].isin(lineages)].copy()

    # Log build summary
    ctx.log({
        "time": utcnow_iso(),
        "level": "INFO",
        "stage": stage_name,
        "site_id": None,
        "lineage": None,
        "message": "Signature matrix built",
        "context": {"num_mutations": int(M), "num_lineages": int(L), "has_GLOBAL": "GLOBAL" in lineages,
                    "rank_S": rank, "cond_S": float(cond) if np.isfinite(cond) else "inf"}
    })

    # Particle filter configuration
    P = int(pf_cfg.get("num_particles", 256))
    kappa = float(pf_cfg.get("kappa", 200.0))
    init_kappa = float(pf_cfg.get("init_kappa", kappa))
    resample_tau = float(pf_cfg.get("resample_threshold", 0.5))  # informational only in APF (we always resample)
    jitter_eps = float(pf_cfg.get("jitter_epsilon", 1e-6))

    # Smoothing config (we will always compute BSS; MAP is optional)
    smoothing_method = str(smoothing_cfg.get("method", "bss")).lower()  # 'bss' | 'map' | 'none'
    lam_kl = float(smoothing_cfg.get("lambda_kl", 10.0))  # only used if method == 'map'
    step_size = float(smoothing_cfg.get("step_size", 0.05))
    max_iter = int(smoothing_cfg.get("max_iter", 200))
    tol = float(smoothing_cfg.get("tol", 1e-6))

    # Posterior predictive config
    pit_bins = int(pp_cfg.get("pit_bins", 20))
    pp_num_draws = int(pp_cfg.get("num_draws", 200))

    # Prepare counts per site/date strictly aligned to `mutations`
    df_counts_site_mut = df_counts[df_counts["mutation"].isin(mutations)].copy()

    # Containers
    traj_rows: List[Dict[str, Any]] = []
    ess_rows: List[Dict[str, Any]] = []
    loglik_rows: List[Dict[str, Any]] = []
    tplus1_rows: List[Dict[str, Any]] = []
    pit_values_all: List[float] = []
    qq_rows: List[Dict[str, Any]] = []
    waic_site_rows: List[Dict[str, Any]] = []
    elpd_delta_rows: List[Dict[str, Any]] = []
    rmse_rows: List[Dict[str, Any]] = []
    turnover_all: List[pd.DataFrame] = []
    first_site_fig_data: Dict[str, Any] = {}

    # Iterate sites
    for site_id, df_site in df_counts_site_mut.groupby("site_id"):
        ctx.log({
            "time": utcnow_iso(),
            "level": "INFO",
            "stage": stage_name,
            "site_id": site_id,
            "lineage": "",
            "message": "Starting APF for site",
            "context": {"num_mutations": int(M), "num_particles": int(P)}
        })

        dates = np.sort(df_site["date"].unique())
        T = len(dates)

        # Build y_list, n_list aligned to S mutations
        y_list: List[np.ndarray] = []
        n_list: List[np.ndarray] = []
        for dt in dates:
            df_d = df_site[df_site["date"] == dt]
            y = df_d.set_index("mutation").reindex(mutations)["count"].fillna(0.0).values.astype(np.float64)
            n = df_d.set_index("mutation").reindex(mutations)["coverage"].fillna(0.0).values.astype(np.float64)
            assert y.shape[0] == M and n.shape[0] == M, f"y/n shape mismatch vs S rows (M={M})"
            y_list.append(y)
            n_list.append(n)

        # Initial theta (baseline or uniform) and initial particles at t=0
        if theta_baseline is not None:
            tb_site0 = theta_baseline[(theta_baseline["site_id"] == site_id) & (theta_baseline["date"] == dates[0])]
            if len(tb_site0) >= L:
                theta0 = tb_site0.set_index("lineage").reindex(lineages)["theta"].fillna(0.0).values.astype(np.float64)
                theta0 = np.clip(theta0, EPS, np.inf)
                theta0 = theta0 / theta0.sum() if theta0.sum() > 0 else np.full(L, 1.0 / L)
            else:
                theta0 = np.full(L, 1.0 / L)
        else:
            theta0 = np.full(L, 1.0 / L)

        particles = np.zeros((P, L), dtype=np.float64)
        for p in range(P):
            particles[p] = _dirichlet_sample(theta0, init_kappa, rng)

        # First-step weights and traces
        logw = _compute_log_weights_for_particles(particles, y_list[0], n_list[0], S, phi)
        w, logZ = _normalize_weights(logw)
        ess_val = _ess(w)
        ness = ess_val / P

        particles_hist: List[np.ndarray] = [particles.copy()]
        logw_hist: List[np.ndarray] = [logw.copy()]

        # Record first time step stats
        med = np.quantile(particles, 0.5, axis=0)
        lo = np.quantile(particles, 0.05, axis=0)
        hi = np.quantile(particles, 0.95, axis=0)
        mean = particles.mean(axis=0)
        for l_idx, lin in enumerate(lineages):
            traj_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[0]), "lineage": lin,
                              "mean": float(mean[l_idx]), "median": float(med[l_idx]),
                              "q05": float(lo[l_idx]), "q95": float(hi[l_idx])})
        ess_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[0]), "ESS": float(ess_val), "NESS": float(ness), "resampled": True})
        loglik_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[0]), "loglik": float(logZ)})

        # PIT/QQ/WAIC for t=0
        mu = np.clip(particles @ S.T, EPS, 1.0 - EPS)  # (P,M)
        # PIT
        def _posterior_predictive_pit_time(weights_t, thetas_t, y_t, n_t, S, phi, rng, max_particles=None):
            wloc = np.asarray(weights_t, dtype=np.float64); wloc = np.clip(wloc, EPS, np.inf); wloc /= wloc.sum()
            thetas_loc = thetas_t
            if max_particles is not None and len(wloc) > max_particles:
                idx = rng.choice(np.arange(len(wloc)), size=max_particles, replace=False, p=wloc / np.sum(wloc))
                wloc = wloc[idx] / np.sum(wloc[idx])
                thetas_loc = thetas_loc[idx]
            mu_ = np.clip(thetas_loc @ S.T, EPS, 1.0 - EPS)
            y = y_t.astype(int); n = n_t.astype(int)
            pits: List[float] = []
            a = mu_ * phi; b = (1.0 - mu_) * phi
            for m in range(y.shape[0]):
                if n[m] <= 0: continue
                F_y = np.sum(wloc * betabinom.cdf(y[m], n[m], a[:, m], b[:, m]))
                F_y_minus = np.sum(wloc * betabinom.cdf(max(y[m] - 1, 0), n[m], a[:, m], b[:, m]))
                v = rng.random(); u = F_y_minus + v * max(F_y - F_y_minus, 0.0)
                pits.append(float(np.clip(u, 0.0, 1.0)))
            return pits

        pit_values_all.extend(_posterior_predictive_pit_time(w, particles, y_list[0], n_list[0], S, phi, rng, max_particles=pit_max_particles))
        mu_mean = np.mean(mu, axis=0)
        nvec = n_list[0].astype(int)
        for obs, pred in zip(y_list[0].tolist(), (nvec * mu_mean).tolist()):
            qq_rows.append({"site_id": site_id, "obs": float(obs), "pred": float(pred)})

        # WAIC at t=0
        y_mat = y_list[0][None, :]; n_mat = n_list[0][None, :]
        a0 = mu * phi; b0 = (1.0 - mu) * phi
        logc0 = gammaln(n_mat + 1.0) - gammaln(y_mat + 1.0) - gammaln(n_mat - y_mat + 1.0)
        logpmf0 = logc0 + (gammaln(y_mat + a0) - gammaln(a0)) + (gammaln(n_mat - y_mat + b0) - gammaln(b0)) + (gammaln(a0 + b0) - gammaln(n_mat + a0 + b0))
        zero_trials = (n_mat <= 0)
        logpmf0 = np.where(zero_trials & (y_mat == 0), 0.0, logpmf0)
        logpmf0 = np.where(zero_trials & (y_mat != 0), -np.inf, logpmf0)

        class _WAICAggregator:
            def __init__(self, max_particles: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
                self.lppd_sum = 0.0; self.p_waic_sum = 0.0
                self.max_particles = max_particles; self.rng = rng or np.random.default_rng(123)
            def update_from_loglik_matrix(self, loglik_pm: np.ndarray) -> None:
                if self.max_particles is not None and loglik_pm.shape[0] > self.max_particles:
                    idx = self.rng.choice(np.arange(loglik_pm.shape[0]), size=self.max_particles, replace=False)
                    loglik_pm = loglik_pm[idx]
                lppd = logsumexp(loglik_pm, axis=0) - np.log(loglik_pm.shape[0])
                var_ll = np.var(loglik_pm, axis=0, ddof=1)
                self.lppd_sum += float(np.sum(lppd))
                self.p_waic_sum += float(np.sum(var_ll))
            def finalize(self) -> Tuple[float, float]:
                elpd_waic = self.lppd_sum - self.p_waic_sum
                waic = -2.0 * elpd_waic
                return float(elpd_waic), float(waic)

        waic_aggr = _WAICAggregator(max_particles=waic_max_particles, rng=rng)
        waic_aggr.update_from_loglik_matrix(logpmf0)

        # Forward APF for t=1..T-1
        for t in range(1, T):
            particles, logw, _ = _apf_step(particles, w, y_list[t], n_list[t], S, phi, s_vec, kappa, rng)
            w, logZ = _normalize_weights(logw)
            ess_val = _ess(w); ness = ess_val / P

            particles_hist.append(particles.copy())
            logw_hist.append(logw.copy())

            # Summaries at time t
            med = np.quantile(particles, 0.5, axis=0)
            lo  = np.quantile(particles, 0.05, axis=0)
            hi  = np.quantile(particles, 0.95, axis=0)
            mean = particles.mean(axis=0)
            for l_idx, lin in enumerate(lineages):
                traj_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[t]), "lineage": lin,
                                  "mean": float(mean[l_idx]), "median": float(med[l_idx]),
                                  "q05": float(lo[l_idx]), "q95": float(hi[l_idx])})
            ess_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[t]), "ESS": float(ess_val), "NESS": float(ness), "resampled": True})
            loglik_rows.append({"site_id": site_id, "date": pd.to_datetime(dates[t]), "loglik": float(logZ)})

            # PIT
            pit_values_all.extend(_posterior_predictive_pit_time(w, particles, y_list[t], n_list[t], S, phi, rng, max_particles=pit_max_particles))
            # QQ
            mu = np.clip(particles @ S.T, EPS, 1.0 - EPS)
            mu_mean = np.mean(mu, axis=0)
            nvec = n_list[t].astype(int)
            for obs, pred in zip(y_list[t].tolist(), (nvec * mu_mean).tolist()):
                qq_rows.append({"site_id": site_id, "obs": float(obs), "pred": float(pred)})
            # WAIC
            y_mat = y_list[t][None, :]; n_mat = n_list[t][None, :]
            a = mu * phi; b = (1.0 - mu) * phi
            logc = gammaln(n_mat + 1.0) - gammaln(y_mat + 1.0) - gammaln(n_mat - y_mat + 1.0)
            logpmf = logc + (gammaln(y_mat + a) - gammaln(a)) + (gammaln(n_mat - y_mat + b) - gammaln(b)) + (gammaln(a + b) - gammaln(n_mat + a + b))
            zero_trials = (n_mat <= 0)
            logpmf = np.where(zero_trials & (y_mat == 0), 0.0, logpmf)
            logpmf = np.where(zero_trials & (y_mat != 0), -np.inf, logpmf)
            waic_aggr.update_from_loglik_matrix(logpmf)

        # Smoothing: BSS (default) or MAP or none
        theta_plot_path = None
        if smoothing_method == "bss":
            theta_bss = _backward_simulation_smoother(particles_hist, logw_hist, s_vec, kappa, rng)
            theta_plot_path = theta_bss
        elif smoothing_method == "map":
            # Optional: MAP smoother
            def _objective_and_grad_theta(theta_path, y_list, n_list, S, phi, s, lam_kl):
                Tloc, Lloc = theta_path.shape
                sel = np.maximum(1.0 + s, EPS)
                fval = 0.0; grad = np.zeros_like(theta_path)
                for tt in range(Tloc):
                    theta_t = np.clip(theta_path[tt], EPS, 1.0)
                    mu_t = np.clip(theta_t @ S.T, EPS, 1.0 - EPS)
                    y = y_list[tt]; n = n_list[tt]
                    a = mu_t * phi; b = (1.0 - mu_t) * phi
                    logc = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
                    logpmf = logc + (gammaln(y + a) - gammaln(a)) + (gammaln(n - y + b) - gammaln(b)) + (gammaln(a + b) - gammaln(n + a + b))
                    zero_trials = (n <= 0)
                    logpmf = np.where(zero_trials & (y == 0), 0.0, logpmf)
                    logpmf = np.where(zero_trials & (y != 0), -np.inf, logpmf)
                    nll_t = -np.sum(logpmf); fval += float(nll_t)
                    term = phi * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
                    term = np.where(zero_trials, 0.0, term)
                    grad[tt] += -(S.T @ term)
                    if tt > 0:
                        theta_prev = np.clip(theta_path[tt - 1], EPS, 1.0)
                        Z_prev = float(theta_prev @ sel); pi_prev = (theta_prev * sel) / Z_prev
                        fval += lam_kl * float(np.sum(theta_t * (np.log(theta_t) - np.log(pi_prev))))
                        grad[tt] += lam_kl * (np.log(theta_t) - np.log(pi_prev) + 1.0)
                    if tt < Tloc - 1:
                        theta_next = np.clip(theta_path[tt + 1], EPS, 1.0)
                        Z_t = float(theta_t @ sel)
                        grad[tt] += lam_kl * (-(theta_next / theta_t) + (sel / Z_t))
                return fval, grad

            def _grad_phi_from_grad_theta(theta_path, grad_theta):
                Tloc, _ = theta_path.shape
                grad_phi = np.zeros_like(theta_path)
                for tt in range(Tloc):
                    theta_t = theta_path[tt]; g_t = grad_theta[tt]; dot = float(np.dot(theta_t, g_t))
                    grad_phi[tt] = theta_t * (g_t - dot)
                return grad_phi

            def _softmax_matrix(phi):
                out = np.empty_like(phi)
                for tt in range(phi.shape[0]):
                    z = phi[tt] - np.max(phi[tt]); e = np.exp(z); out[tt] = e / np.sum(e)
                return out

            # Use filtered medians as init
            med_path = np.vstack([
                np.quantile(particles_hist[t], 0.5, axis=0)
                for t in range(T)
            ])
            theta = np.clip(med_path.copy(), EPS, 1.0); theta /= theta.sum(axis=1, keepdims=True)
            phi_logits = np.log(theta)
            f_cur, grad_theta = _objective_and_grad_theta(theta, y_list, n_list, S, phi, s_vec, lam_kl)
            obj_hist: List[float] = [float(f_cur)]
            for _ in range(max_iter):
                grad_phi = _grad_phi_from_grad_theta(theta, grad_theta)
                dir_phi = grad_phi; grad_dot = float(np.sum(grad_phi * dir_phi))
                if not np.isfinite(grad_dot) or grad_dot <= tol: break
                eta = float(step_size); accepted = False
                for _bt in range(50):
                    phi_new = phi_logits - eta * dir_phi
                    theta_new = _softmax_matrix(phi_new)
                    f_new, _ = _objective_and_grad_theta(theta_new, y_list, n_list, S, phi, s_vec, lam_kl)
                    if f_new <= f_cur - 1e-4 * eta * grad_dot:
                        accepted = True; break
                    eta *= 0.5
                if not accepted: break
                phi_logits = phi_new; theta = theta_new
                f_cur, grad_theta = _objective_and_grad_theta(theta, y_list, n_list, S, phi, s_vec, lam_kl)
                obj_hist.append(float(f_cur))
                if len(obj_hist) >= 2 and abs(obj_hist[-1] - obj_hist[-2]) < tol: break
            theta_plot_path = theta
            fig_conv, ax_conv = plt.subplots(figsize=(5.2, 3.4))
            ax_conv.plot(np.arange(len(obj_hist)), obj_hist, lw=1.8)
            ax_conv.set_xlabel("Iteration"); ax_conv.set_ylabel("Objective"); ax_conv.set_title(f"MAP convergence ({site_id})")
            ax_conv.grid(True, alpha=0.4)
            # No legend here; avoid calling place_legend_below
            # place_legend_below(ax_conv, ncol=1)
            ctx.write_figure(f"map_convergence_{site_id}", fig_conv); plt.close(fig_conv)
        else:
            # No smoother: use filtered medians
            theta_plot_path = np.vstack([
                np.quantile(particles_hist[t], 0.5, axis=0)
                for t in range(T)
            ])

        # t+1 predictions (lineage + SNV counts)
        last_particles = particles_hist[-1]  # (P,L)
        next_particles = np.zeros_like(last_particles)
        for p in range(P):
            pi = _pi_selection(last_particles[p], s_vec)
            next_particles[p] = _dirichlet_sample(pi, kappa, rng)

        med = np.quantile(next_particles, 0.5, axis=0)
        lo  = np.quantile(next_particles, 0.05, axis=0)
        hi  = np.quantile(next_particles, 0.95, axis=0)
        mean = next_particles.mean(axis=0)
        next_date = pd.to_datetime(dates[-1]) + pd.Timedelta(days=1)
        for l_idx, lin in enumerate(lineages):
            tplus1_rows.append({"site_id": site_id, "date": next_date, "datatype": "lineage_prop",
                                "lineage": lin, "mutation": "", "mean": float(mean[l_idx]),
                                "median": float(med[l_idx]), "q05": float(lo[l_idx]), "q95": float(hi[l_idx])})

        # Predict SNV counts at t+1 using last coverage as proxy
        use_last_cov = bool(tplus1_cfg.get("use_last_coverage", True))
        n_next = n_list[-1].astype(int) if use_last_cov else np.median(np.vstack(n_list), axis=0).astype(int)
        mu_next = np.clip(next_particles @ S.T, EPS, 1.0 - EPS)  # (P,M)
        # Vectorized Beta (Gamma ratio) + Binomial draws
        Kdraw = int(pp_num_draws)
        idx_p = rng.integers(0, P, size=Kdraw)
        a_next = mu_next[idx_p, :] * phi[None, :]
        b_next = (1.0 - mu_next[idx_p, :]) * phi[None, :]
        ga = rng.gamma(shape=np.maximum(a_next, EPS), scale=1.0)
        gb = rng.gamma(shape=np.maximum(b_next, EPS), scale=1.0)
        p_draw = ga / np.maximum(ga + gb, EPS)
        draws = rng.binomial(n=np.maximum(n_next, 0)[None, :].astype(int), p=np.clip(p_draw, 0.0, 1.0))
        med_counts = np.quantile(draws, 0.5, axis=0); lo_counts = np.quantile(draws, 0.05, axis=0)
        hi_counts = np.quantile(draws, 0.95, axis=0); mean_counts = draws.mean(axis=0)
        for m_idx, mut in enumerate(mutations):
            tplus1_rows.append({"site_id": site_id, "date": next_date, "datatype": "snv_count",
                                "lineage": "", "mutation": mut, "mean": float(mean_counts[m_idx]),
                                "median": float(med_counts[m_idx]), "q05": float(lo_counts[m_idx]), "q95": float(hi_counts[m_idx])})

        # WAIC for this site
        elpd_waic_site, waic_site = waic_aggr.finalize()
        waic_site_rows.append({"site_id": site_id, "elpd_waic": float(elpd_waic_site), "waic": float(waic_site)})

        # ΔELPD vs baseline (plug-in)
        if theta_baseline is not None:
            tb_site = theta_baseline[(theta_baseline["site_id"] == site_id) & (theta_baseline["lineage"].isin(lineages))]
            if len(tb_site) > 0:
                theta_base_path = np.zeros((T, L), dtype=np.float64)
                for t, dt in enumerate(dates):
                    row = tb_site[tb_site["date"] == dt].set_index("lineage").reindex(lineages)["theta"].fillna(0.0).values
                    ssum = np.sum(row); row = row / ssum if ssum > 0 else np.full(L, 1.0 / L)
                    theta_base_path[t] = row
                elpd = 0.0
                for t in range(T):
                    mu_b = np.clip(theta_base_path[t] @ S.T, EPS, 1.0 - EPS)
                    elpd += float(np.sum(_log_beta_binom_pmf(y_list[t], n_list[t], mu_b, phi)))
                elpd_delta_rows.append({"site_id": site_id, "delta_elpd": float(elpd_waic_site - elpd)})

        # Store first site data for overview
        if not first_site_fig_data:
            traj_site = pd.DataFrame([r for r in traj_rows if r["site_id"] == site_id]).copy()
            ess_site = pd.DataFrame([r for r in ess_rows if r["site_id"] == site_id]).copy()
            loglik_site = pd.DataFrame([r for r in loglik_rows if r["site_id"] == site_id]).copy()
            qq_site = pd.DataFrame([r for r in qq_rows if r["site_id"] == site_id]).copy()
            first_site_fig_data = {"site_id": site_id, "dates": dates, "lineages": lineages,
                                   "traj": traj_site, "ess": ess_site, "loglik": loglik_site, "qq": qq_site}

        # Per-site figures & metrics
        traj_site = pd.DataFrame([r for r in traj_rows if r["site_id"] == site_id]).copy().sort_values(["lineage", "date"])
        ess_site = pd.DataFrame([r for r in ess_rows if r["site_id"] == site_id]).copy()
        loglik_site = pd.DataFrame([r for r in loglik_rows if r["site_id"] == site_id]).copy()
        qq_site = pd.DataFrame([r for r in qq_rows if r["site_id"] == site_id]).copy()

        fig_diag = _evolution_2x2_panel(site_id, dates, lineages, traj_site, y_list, n_list, qq_site)
        ctx.write_figure(f"site_panel_2x2_{site_id}", fig_diag); plt.close(fig_diag)

        obs_af, pred_af = _compute_obs_pred_af(y_list, n_list, theta_plot_path, S)
        fig_muts = _mutation_timeseries_2x2_figure(site_id, dates, mutations, obs_af, pred_af)
        ctx.write_figure(f"site_mutations_2x2_{site_id}", fig_muts); plt.close(fig_muts)

        fig_heat = _mutation_heatmap_figure(site_id, dates, mutations, obs_af, k=40)
        ctx.write_figure(f"site_mutation_heatmap_{site_id}", fig_heat); plt.close(fig_heat)

        df_turn = _turnover_metrics(dates, theta_plot_path); df_turn["site_id"] = site_id
        turnover_all.append(df_turn)
        fig_turn = _turnover_figure(site_id, df_turn)
        ctx.write_figure(f"site_turnover_{site_id}", fig_turn); plt.close(fig_turn)

        if not qq_site.empty:
            s_rmse = float(rmse(qq_site["obs"].values, qq_site["pred"].values))
            rmse_rows.append({"site_id": site_id, "rmse_counts": s_rmse})

        median_ness = float(np.median(ess_site["NESS"])) if not ess_site.empty else np.nan
        pct_low = float((ess_site["NESS"] < 0.2).mean()) if not ess_site.empty else np.nan
        ctx.write_metric(f"pf_ness_{site_id}", {"site_id": site_id, "median_ness": median_ness, "pct_steps_lt_0.2": pct_low})

    # --------------------------- Save tables (global) ---------------------------
    df_traj = pd.DataFrame(traj_rows)
    if "map" not in df_traj.columns: df_traj["map"] = np.nan
    df_traj = df_traj.sort_values(["site_id", "date", "lineage"]).reset_index(drop=True)
    ctx.write_table("forecast_smoothed_props", df_traj)

    df_ess = pd.DataFrame(ess_rows).sort_values(["site_id", "date"]).reset_index(drop=True)
    ctx.write_table("forecast_ess_trace", df_ess)

    df_loglik = pd.DataFrame(loglik_rows).sort_values(["site_id", "date"]).reset_index(drop=True)
    ctx.write_table("forecast_loglik_trace", df_loglik)

    df_tplus1 = pd.DataFrame(tplus1_rows).sort_values(["site_id", "date", "datatype", "lineage", "mutation"]).reset_index(drop=True)
    ctx.write_table("tplus1_predictions", df_tplus1)

    df_waic = pd.DataFrame(waic_site_rows)
    ctx.write_table("waic", df_waic)

    if len(elpd_delta_rows) > 0:
        ctx.write_table("delta_elpd", pd.DataFrame(elpd_delta_rows))

    if len(rmse_rows) > 0:
        ctx.write_table("rmse_counts_per_site", pd.DataFrame(rmse_rows).sort_values("site_id").reset_index(drop=True))

    if len(turnover_all) > 0:
        df_turn_all = pd.concat(turnover_all, ignore_index=True)
        df_turn_all = df_turn_all[["site_id", "date", "H", "JSD_prev"]].sort_values(["site_id", "date"])
        ctx.write_table("turnover_metrics", df_turn_all)

    # --------------------------- First-site overview figure ---------------------------
    if first_site_fig_data:
        site_id = first_site_fig_data["site_id"]
        dates = pd.to_datetime(first_site_fig_data["dates"])
        traj_site = first_site_fig_data["traj"]
        ess_site = first_site_fig_data["ess"]
        loglik_site = first_site_fig_data["loglik"]
        qq_site = first_site_fig_data["qq"]
        K = min(6, len(traj_site["lineage"].unique()))

        fig = plt.figure(figsize=(10.5, 8.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.25)
        ax_traj = fig.add_subplot(gs[0, :])
        ax_ess = fig.add_subplot(gs[1, 0])
        ax_ll = fig.add_subplot(gs[1, 1])
        ax_pit = fig.add_subplot(gs[2, 0])
        ax_qq = fig.add_subplot(gs[2, 1])

        med_by_lin = traj_site.groupby("lineage")["median"].mean().sort_values(ascending=False)
        top_lineages = list(med_by_lin.head(K).index)
        cmap = _color_map(top_lineages)
        for lin in top_lineages:
            df_lin = traj_site[traj_site["lineage"] == lin].sort_values("date")
            ax_traj.plot(df_lin["date"], df_lin["median"], label=lin, lw=1.8, color=cmap[lin])
            ax_traj.fill_between(df_lin["date"], df_lin["q05"], df_lin["q95"], alpha=0.15, color=cmap[lin])
        _format_date_axis(ax_traj)
        ax_traj.set_title(f"θ trajectories (site: {site_id})")
        ax_traj.set_xlabel("Date"); ax_traj.set_ylabel("Proportion")
        ax_traj.grid(True, alpha=0.4)
        place_legend_below(ax_traj, ncol=min(K, 3))

        if not ess_site.empty:
            ax_ess.plot(ess_site["date"], ess_site["ESS"], label="ESS", lw=1.6)
            ax_ess.plot(ess_site["date"], ess_site["NESS"], label="NESS", lw=1.6)
            ax_ess.axhline(y=0.2, lw=0.8, alpha=0.3)
            _format_date_axis(ax_ess)
            ax_ess.set_title("ESS / NESS trace")
            ax_ess.set_xlabel("Date"); ax_ess.set_ylabel("Value")
            ax_ess.grid(True, alpha=0.4)
            place_legend_below(ax_ess, ncol=2)

        if not loglik_site.empty:
            ax_ll.plot(loglik_site["date"], loglik_site["loglik"], lw=1.6)
            _format_date_axis(ax_ll)
            ax_ll.set_title("Log-evidence (per step)")
            ax_ll.set_xlabel("Date"); ax_ll.set_ylabel("log p(y_t | y_{1:t-1})")
            ax_ll.grid(True, alpha=0.4)

        # PIT (all sites)
        if len(pit_values_all) > 0:
            bins = np.linspace(0.0, 1.0, pit_bins + 1)
            ax_pit.hist(pit_values_all, bins=bins, alpha=0.8)
        ax_pit.set_title("Posterior predictive PIT")
        ax_pit.set_xlabel("PIT"); ax_pit.set_ylabel("Frequency")
        ax_pit.grid(True, alpha=0.4)

        # QQ
        if not qq_site.empty:
            ax_qq.scatter(qq_site["pred"] + EPS, qq_site["obs"] + EPS, s=12, alpha=0.5)
            lim = [
                max(min(qq_site["pred"].min(), qq_site["obs"].min()), 1e-3),
                max(qq_site["pred"].max(), qq_site["obs"].max()),
            ]
            ax_qq.plot(lim, lim, lw=1.0, alpha=0.7)
            ax_qq.set_xscale("log"); ax_qq.set_yscale("log")
        ax_qq.set_title("Predicted vs observed (counts)")
        ax_qq.set_xlabel("Predicted (median)"); ax_qq.set_ylabel("Observed")
        ax_qq.grid(True, which="both", alpha=0.4)

        ctx.write_figure("forecast_panel", fig)
        plt.close(fig)

    # Diagnostics tables
    df_pit = pd.DataFrame({"pit": np.array(pit_values_all, dtype=np.float64)})
    ctx.write_table("posterior_predictive_pit", df_pit)
    df_qq = pd.DataFrame(qq_rows)
    ctx.write_table("posterior_predictive_qq", df_qq)

    # Report
    df_ess = pd.DataFrame(ess_rows)
    df_waic = pd.DataFrame(waic_site_rows) if len(waic_site_rows) else pd.DataFrame(columns=["elpd_waic","waic"])
    df_rmse = pd.DataFrame(rmse_rows) if len(rmse_rows) else pd.DataFrame(columns=["rmse_counts"])

    median_ness_overall = df_ess["NESS"].median() if not df_ess.empty else np.nan
    frac_low_overall = float((df_ess["NESS"] < 0.2).mean()) if not df_ess.empty else np.nan
    coverage_pit = {"pit_mean": float(df_pit["pit"].mean()) if not df_pit.empty else np.nan,
                    "pit_var": float(df_pit["pit"].var()) if not df_pit.empty else np.nan}
    waic_summary = {"mean_elpd_waic": float(df_waic["elpd_waic"].mean()) if not df_waic.empty else np.nan,
                    "mean_waic": float(df_waic["waic"].mean()) if not df_waic.empty else np.nan}
    mean_rmse = float(df_rmse["rmse_counts"].mean()) if not df_rmse.empty else np.nan

    ctx.write_metric("forecast_acceptance", {
        "median_ness_overall": float(median_ness_overall),
        "frac_steps_ness_lt_0.2": float(frac_low_overall),
        **coverage_pit,
        **waic_summary,
        "rmse_counts_mean": float(mean_rmse),
        "rank_S": float(rank) if rank == rank else None,
        "cond_S": float(cond) if np.isfinite(cond) else None,
    })

    report_lines = []
    report_lines.append("# Forecast stage report")
    report_lines.append("")
    report_lines.append(f"- Mutations used (M): {M}  (from priors; expected full set)")
    report_lines.append(f"- S rank: {rank}, cond(S): {cond if np.isfinite(cond) else 'inf'}")
    report_lines.append(f"- Median NESS across sites: {median_ness_overall:.3f}")
    report_lines.append(f"- Fraction of steps with NESS < 0.2: {frac_low_overall:.3f}")
    report_lines.append(f"- PIT mean: {coverage_pit['pit_mean']:.3f}, PIT var: {coverage_pit['pit_var']:.3f}")
    report_lines.append(f"- Mean ELPD (WAIC): {waic_summary['mean_elpd_waic']:.2f}, Mean WAIC: {waic_summary['mean_waic']:.2f}")
    report_lines.append(f"- Mean RMSE (pred vs obs counts): {mean_rmse:.3f}")
    report_lines.append("")
    report_lines.append("Key outputs:")
    report_lines.append("- forecast_smoothed_props.csv: θ trajectories (median; MAP column present if enabled)")
    report_lines.append("- forecast_ess_trace.csv: ESS/NESS per step")
    report_lines.append("- forecast_loglik_trace.csv: incremental log-evidence per step")
    report_lines.append("- tplus1_predictions.csv: predictive summaries for next day (lineage proportions and SNV counts)")
    report_lines.append("- waic.csv: WAIC and ELPD per site; delta_elpd.csv vs baseline if available")
    report_lines.append("- rmse_counts_per_site.csv: RMSE of predicted median counts vs observed counts")
    report_lines.append("- turnover_metrics.csv: Shannon H and JSD between successive dates per site")
    report_lines.append("- posterior_predictive_pit.csv, posterior_predictive_qq.csv: calibration diagnostics")
    ctx.write_report("\\n".join(report_lines))
