import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from scipy.stats import norm

_EPS = np.finfo(float).eps


def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values (same shape as y_true or broadcastable).
    sample_weight : array-like, optional
        Weights for each observation, broadcastable to y_true.
    axis : int or tuple of ints, optional
        Axis along which to compute the RMSE. If None, computes a scalar over all entries.

    Returns
    -------
    np.ndarray
        RMSE values aggregated along `axis` (or scalar if `axis` is None).

    Notes
    -----
    - NaNs are ignored (via np.nansum). If all weights are NaN/masked, result may be NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err2 = (y_true - y_pred) ** 2

    if sample_weight is None:
        num = np.nansum(err2, axis=axis)
        den = np.sum(~np.isnan(err2), axis=axis) + _EPS
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.broadcast_to(w, err2.shape)
        mask = ~np.isnan(err2)
        num = np.nansum(w * np.where(mask, err2, 0.0), axis=axis)
        den = np.nansum(w * mask, axis=axis) + _EPS

    return np.sqrt(num / den)


def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values (same shape as y_true or broadcastable).
    sample_weight : array-like, optional
        Weights for each observation, broadcastable to y_true.
    axis : int or tuple of ints, optional
        Axis along which to compute the MAE. If None, computes a scalar over all entries.

    Returns
    -------
    np.ndarray
        MAE aggregated along `axis` (or scalar if `axis` is None).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)

    if sample_weight is None:
        num = np.nansum(err, axis=axis)
        den = np.sum(~np.isnan(err), axis=axis) + _EPS
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.broadcast_to(w, err.shape)
        mask = ~np.isnan(err)
        num = np.nansum(w * np.where(mask, err, 0.0), axis=axis)
        den = np.nansum(w * mask, axis=axis) + _EPS

    return num / den


def _normalize_simplex(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize nonnegative array to the probability simplex along `axis`."""
    a = np.maximum(a, 0.0)
    s = np.sum(a, axis=axis, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return a / s


def _kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1, base: float = math.e) -> np.ndarray:
    """Kullback-Leibler divergence KL(p || q) with numerical safeguards."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    ratio = p / q
    log_ratio = np.log(ratio)
    kl = np.sum(p * log_ratio, axis=axis)
    if base != math.e:
        kl = kl / math.log(base)
    return kl


def js_divergence(
    p: ArrayLike,
    q: ArrayLike,
    axis: int = -1,
    base: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Jensen-Shannon divergence between two probability distributions.

    Parameters
    ----------
    p : array-like
        First distribution(s). Nonnegative; will be normalized along `axis` if `normalize=True`.
    q : array-like
        Second distribution(s). Nonnegative; will be normalized along `axis` if `normalize=True`.
    axis : int, default -1
        Axis along which to treat values as categories (must be same for p and q).
    base : float, default 2.0
        Logarithm base for the divergence units. Base=2 gives bits.
    normalize : bool, default True
        Whether to normalize `p` and `q` to sum to 1 along `axis`.

    Returns
    -------
    np.ndarray
        JS divergence for each slice orthogonal to `axis`. Nonnegative and bounded by log(base).

    Notes
    -----
    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5 (p + q).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if normalize:
        p = _normalize_simplex(p, axis=axis)
        q = _normalize_simplex(q, axis=axis)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, axis=axis, base=base) + 0.5 * _kl_divergence(q, m, axis=axis, base=base)


def crps(
    y: ArrayLike,
    pred: ArrayLike,
    kind: str = "samples",
    sigma: Optional[ArrayLike] = None,
    samples_axis: int = 0,
    reduce: bool = True,
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Continuous Ranked Probability Score (CRPS).

    Parameters
    ----------
    y : array-like
        Observations (shape (...,)).
    pred : array-like
        Predictive representation.
        - If kind="samples": predictive samples of shape (S, ...) where axis=S is `samples_axis`.
        - If kind="gaussian": predictive mean μ with same shape as `y`.
    kind : {"samples", "gaussian"}, default "samples"
        Choice of CRPS computation.
    sigma : array-like, optional
        Predictive standard deviation (only for kind="gaussian"), broadcastable to y.
    samples_axis : int, default 0
        Axis index of sample dimension when kind="samples".
    reduce : bool, default True
        If True, returns weighted mean CRPS over all entries. If False, returns pointwise CRPS with shape of y.
    sample_weight : array-like, optional
        Weights for averaging over entries (not over samples). Broadcastable to y.

    Returns
    -------
    np.ndarray
        Scalar if `reduce=True`; else array of CRPS values matching shape of y.

    Notes
    -----
    - For samples: CRPS ≈ E|X − y| − 0.5 E|X − X'| estimated via empirical formulas.
    - For Gaussian: closed form CRPS for N(μ, σ²).
    """
    y = np.asarray(y, dtype=float)

    if kind.lower() in ("samples", "empirical"):
        samples = np.asarray(pred, dtype=float)
        if samples_axis != 0:
            samples = np.moveaxis(samples, samples_axis, 0)  # (S, ...)
        S = samples.shape[0]
        if S < 2:
            raise ValueError("Need at least two samples to compute CRPS with 'samples' kind.")
        # term1 = mean |x - y| over samples
        term1 = np.mean(np.abs(samples - y[None, ...]), axis=0)

        # term2 = 0.5 * E|X - X'|, where E|X - X'| = (2 / S^2) * sum_{i<j} (x_j - x_i)
        # Efficient computation using sorted samples along sample axis
        sorted_samples = np.sort(samples, axis=0)
        k = np.arange(1, S + 1, dtype=float)[:, None]
        w = (2.0 * k - S - 1.0)  # shape (S,1), will broadcast to (S, ...)
        # sum_{k} w_k * x_(k) for each (...), then multiply by 2
        weighted_sum = np.sum(w * sorted_samples, axis=0)
        e_abs = (2.0 * weighted_sum) / (S * S)
        term2 = 0.5 * e_abs

        crps_pointwise = term1 - term2

    elif kind.lower() in ("gaussian", "normal"):
        if sigma is None:
            raise ValueError("sigma must be provided for kind='gaussian'.")
        mu = np.asarray(pred, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, _EPS)
        z = (y - mu) / sigma
        crps_pointwise = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    else:
        raise ValueError("kind must be one of {'samples','gaussian'}")

    if not reduce:
        return crps_pointwise

    if sample_weight is None:
        return float(np.nanmean(crps_pointwise))
    sw = np.asarray(sample_weight, dtype=float)
    sw = np.broadcast_to(sw, crps_pointwise.shape)
    mask = ~np.isnan(crps_pointwise)
    num = np.nansum(sw * np.where(mask, crps_pointwise, 0.0))
    den = np.nansum(sw * mask) + _EPS
    return float(num / den)


def brier(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Brier score for binary events.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes in {0,1}.
    y_prob : array-like
        Predicted probabilities in [0,1].
    sample_weight : array-like, optional
        Weights broadcastable to y_true.
    axis : int or tuple of ints, optional
        Axis along which to aggregate. If None, returns scalar.

    Returns
    -------
    np.ndarray
        Brier score aggregated along `axis` (or scalar if `axis` is None).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    se = (y_prob - y_true) ** 2

    if sample_weight is None:
        num = np.nansum(se, axis=axis)
        den = np.sum(~np.isnan(se), axis=axis) + _EPS
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.broadcast_to(w, se.shape)
        mask = ~np.isnan(se)
        num = np.nansum(w * np.where(mask, se, 0.0), axis=axis)
        den = np.nansum(w * mask, axis=axis) + _EPS

    return num / den


def coverage_rate(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    nominal: Optional[float] = None,
    sample_weight: Optional[ArrayLike] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Empirical coverage rate of an interval forecast.

    Parameters
    ----------
    y : array-like
        Observations.
    lower : array-like
        Lower bounds (broadcastable to y).
    upper : array-like
        Upper bounds (broadcastable to y).
    nominal : float, optional
        Nominal coverage level in (0,1). If provided, returns (coverage, coverage - nominal).
    sample_weight : array-like, optional
        Weights for averaging coverage.
    axis : int or tuple of ints, optional
        Axis along which to compute coverage. If None, returns scalar.

    Returns
    -------
    coverage : np.ndarray
        Empirical coverage rate along `axis`.
    delta : np.ndarray or None
        coverage - nominal if `nominal` is provided; else None.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y >= lower) & (y <= upper)

    if sample_weight is None:
        num = np.sum(covered, axis=axis)
        den = np.size(covered, axis=axis) if axis is not None else covered.size
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.broadcast_to(w, covered.shape)
        num = np.sum(w * covered, axis=axis)
        den = np.sum(w, axis=axis) + _EPS

    cov = num / (den + _EPS)
    if nominal is None:
        return cov, None
    return cov, cov - nominal


def ece(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: Optional[ArrayLike] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error (ECE) for binary probabilities.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes in {0,1}.
    y_prob : array-like
        Predicted probabilities in [0,1].
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform","quantile"}, default "uniform"
        Binning strategy. "uniform" uses equal-width bins; "quantile" uses quantile bins.
    sample_weight : array-like, optional
        Weights for averaging.

    Returns
    -------
    ece_value : float
        Expected calibration error (lower is better).
    bin_table : pandas.DataFrame
        Table with per-bin coverage (acc), confidence (avg prob), count, weight mass, and contribution.

    Notes
    -----
    ECE = sum_i (weight_i) * |acc_i - conf_i|, where weight_i = n_i / N (or mass of weights).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    assert y_true.shape == y_prob.shape, "Shapes of y_true and y_prob must match."

    if sample_weight is None:
        w = np.ones_like(y_true, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).ravel()
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.maximum(w, 0.0)

    N = np.sum(w) + _EPS

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(y_prob, edges, right=False) - 1, 0, n_bins - 1)
    elif strategy == "quantile":
        # Avoid duplicated edges by using ranks
        ranks = (np.argsort(np.argsort(y_prob)) + 1) / (len(y_prob) + 1.0)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(ranks, edges, right=False) - 1, 0, n_bins - 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    df = pd.DataFrame({"y": y_true, "p": y_prob, "w": w, "bin": bin_idx})
    grouped = df.groupby("bin", sort=True, observed=True)

    acc = grouped.apply(lambda g: np.average(g["y"], weights=g["w"]) if g["w"].sum() > 0 else np.nan)
    conf = grouped.apply(lambda g: np.average(g["p"], weights=g["w"]) if g["w"].sum() > 0 else np.nan)
    n = grouped.size().astype(float)
    mass = grouped["w"].sum()
    mass = mass / (N + _EPS)

    contrib = mass * np.abs(acc - conf)
    table = pd.DataFrame(
        {
            "bin": acc.index.values,
            "acc": acc.values,
            "conf": conf.values,
            "count": n.values,
            "mass": mass.values,
            "contribution": contrib.values,
            "lower": edges[:-1],
            "upper": edges[1:],
        }
    )
    table = table.sort_values("bin").reset_index(drop=True)
    ece_value = float(np.nansum(table["contribution"].values))
    return ece_value, table


def ess(
    weights: ArrayLike,
    axis: Optional[int] = None,
    normalized: bool = False,
) -> np.ndarray:
    """
    Effective Sample Size (ESS) for importance/particle weights.

    Parameters
    ----------
    weights : array-like
        Nonnegative weights.
    axis : int, optional
        Axis along which to compute ESS. If None, flatten array.
    normalized : bool, default False
        If True, returns NESS = ESS / N along the specified axis.

    Returns
    -------
    np.ndarray
        ESS or NESS along `axis`.

    Notes
    -----
    ESS = (sum w)^2 / sum w^2, with safeguards for zero-sum weights.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    if axis is None:
        w = w.ravel()
        num = (np.sum(w)) ** 2
        den = np.sum(w ** 2) + _EPS
        val = num / den
        if normalized:
            N = len(w)
            return float(val / (N + _EPS))
        return float(val)
    # Along axis
    num = np.sum(w, axis=axis, keepdims=False) ** 2
    den = np.sum(w ** 2, axis=axis, keepdims=False) + _EPS
    val = num / den
    if normalized:
        N = w.shape[axis]
        return val / (N + _EPS)
    return val


def weight_entropy(
    weights: ArrayLike,
    axis: Optional[int] = None,
    base: float = math.e,
    normalize: bool = False,
) -> np.ndarray:
    """
    Entropy of normalized weights (importance or categorical).

    Parameters
    ----------
    weights : array-like
        Nonnegative weights to be normalized to a simplex along `axis`.
    axis : int, optional
        Axis along which to compute entropy. If None, flattens input.
    base : float, default e
        Logarithm base for the entropy.
    normalize : bool, default False
        If True, returns entropy normalized by log(N) to lie in [0,1].

    Returns
    -------
    np.ndarray
        Entropy values along `axis`.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    if axis is None:
        w = w.ravel()
        p = w / (np.sum(w) + _EPS)
        p = np.clip(p, _EPS, 1.0)
        H = -np.sum(p * np.log(p)) / math.log(base)
        if normalize:
            N = len(p)
            return float(H / (math.log(N) / math.log(base)))
        return float(H)
    # along axis
    s = np.sum(w, axis=axis, keepdims=True) + _EPS
    p = w / s
    p = np.clip(p, _EPS, 1.0)
    H = -np.sum(p * np.log(p), axis=axis) / math.log(base)
    if normalize:
        N = w.shape[axis]
        Hnorm = H / (math.log(N) / math.log(base))
        return Hnorm
    return H


def degeneracy_index(weights: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Degeneracy index for particle filters: 1 - NESS.

    Parameters
    ----------
    weights : array-like
        Nonnegative weights.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    np.ndarray
        Degeneracy index in [0,1], where 0 is ideal (uniform weights) and 1 is degenerate (all mass on one).
    """
    ness = ess(weights, axis=axis, normalized=True)
    return 1.0 - ness


def elpd_waic_loo_skeleton(
    log_lik_samples: ArrayLike,
    baseline_log_lik_samples: Optional[ArrayLike] = None,
    truncate_tau: float = 10.0,
) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    """
    Compute WAIC and an approximate LOO (truncated IS-LOO) from log-likelihood samples.

    Parameters
    ----------
    log_lik_samples : array-like, shape (S, N)
        Log-likelihood draws for N observations and S posterior samples.
    baseline_log_lik_samples : array-like, shape (S0, N), optional
        Baseline model log-likelihood draws for Δ comparisons.
    truncate_tau : float, default 10.0
        Truncation factor for importance weights in TIS-LOO, following Ionides (2008)-style truncation.

    Returns
    -------
    dict
        Dictionary with summary metrics and pointwise table:
        - 'waic': float, smaller is better.
        - 'elpd_waic': float
        - 'p_waic': float
        - 'lppd': float
        - 'elpd_loo': float
        - 'elpd_loo_se': float (standard error across observations)
        - 'pointwise': DataFrame with columns ['lppd','p_waic','elpd_waic','elpd_loo','var_loglik'].
        - If baseline provided: 'delta_elpd_waic', 'delta_elpd_waic_se', 'delta_elpd_loo', 'delta_elpd_loo_se'.

    Notes
    -----
    - WAIC uses lppd and variance of log-likelihood across draws (per-observation).
    - LOO is approximated by truncated importance sampling (TIS-LOO). For robust results, PSIS is preferred,
      but TIS-LOO is more stable than naive harmonic mean.
    """
    lp = np.asarray(log_lik_samples, dtype=float)
    if lp.ndim != 2:
        raise ValueError("log_lik_samples must be 2D array of shape (S, N).")
    S, N = lp.shape
    if S < 2:
        raise ValueError("Require at least 2 posterior draws for WAIC/LOO computations.")

    # lppd and p_waic
    lppd_i = logsumexp(lp, axis=0) - math.log(S)  # size N
    var_lp_i = np.var(lp, axis=0, ddof=1) if S > 1 else np.zeros(N, dtype=float)
    p_waic_i = var_lp_i
    elpd_waic_i = lppd_i - p_waic_i

    # TIS-LOO: log w_is = -lp_is; truncate at tau * mean w
    logw = -lp  # shape (S, N)
    # For each i, compute a = log(mean w)
    a = logsumexp(logw, axis=0) - math.log(S)  # shape (N,)
    log_tau = math.log(max(truncate_tau, 1.0))
    # Truncated log weights
    logw_trunc = np.minimum(logw, (log_tau + a)[None, :])
    # log mean of truncated w
    log_mean_wt = logsumexp(logw_trunc, axis=0) - math.log(S)
    # log p_loo_i ≈ - log_mean_wt
    elpd_loo_i = -log_mean_wt

    # Summaries
    elpd_waic = float(np.sum(elpd_waic_i))
    lppd = float(np.sum(lppd_i))
    p_waic = float(np.sum(p_waic_i))
    waic = float(-2.0 * elpd_waic)

    elpd_loo = float(np.sum(elpd_loo_i))
    # SE via sqrt(N * Var(pointwise))
    se_elpd_loo = float(np.sqrt(N * np.var(elpd_loo_i, ddof=1)) if N > 1 else 0.0)

    pointwise = pd.DataFrame(
        {
            "lppd": lppd_i,
            "p_waic": p_waic_i,
            "elpd_waic": elpd_waic_i,
            "elpd_loo": elpd_loo_i,
            "var_loglik": var_lp_i,
        }
    )

    out: Dict[str, Union[float, np.ndarray, pd.DataFrame]] = {
        "waic": waic,
        "elpd_waic": elpd_waic,
        "p_waic": p_waic,
        "lppd": lppd,
        "elpd_loo": elpd_loo,
        "elpd_loo_se": se_elpd_loo,
        "pointwise": pointwise,
    }

    if baseline_log_lik_samples is not None:
        base = np.asarray(baseline_log_lik_samples, dtype=float)
        if base.shape[1] != N:
            raise ValueError("baseline_log_lik_samples must have the same number of observations (N) as log_lik_samples.")
        # Compute baseline summaries
        base_summ = elpd_waic_loo_skeleton(base, baseline_log_lik_samples=None, truncate_tau=truncate_tau)
        base_pw = base_summ["pointwise"]
        # Δ pointwise
        d_waic_i = elpd_waic_i - base_pw["elpd_waic"].to_numpy()
        d_loo_i = elpd_loo_i - base_pw["elpd_loo"].to_numpy()
        delta_elpd_waic = float(np.sum(d_waic_i))
        delta_elpd_loo = float(np.sum(d_loo_i))
        se_delta_waic = float(np.sqrt(N * np.var(d_waic_i, ddof=1)) if N > 1 else 0.0)
        se_delta_loo = float(np.sqrt(N * np.var(d_loo_i, ddof=1)) if N > 1 else 0.0)
        out.update(
            {
                "delta_elpd_waic": delta_elpd_waic,
                "delta_elpd_waic_se": se_delta_waic,
                "delta_elpd_loo": delta_elpd_loo,
                "delta_elpd_loo_se": se_delta_loo,
            }
        )

    return out


def detection_delay(
    detections: pd.DataFrame,
    references: pd.DataFrame,
    detection_flag_col: str = "threshold_crossed",
    first_detect_value: Union[int, bool] = True,
) -> pd.DataFrame:
    """
    Compute detection delays (in days) per site and lineage.

    Parameters
    ----------
    detections : DataFrame
        Must contain columns: ['site_id','date','lineage', detection_flag_col].
        'date' should be datetime-like or string parseable as date.
    references : DataFrame
        Must contain columns: ['site_id','lineage','ref_date'] with datetime-like 'ref_date'.
    detection_flag_col : str, default "threshold_crossed"
        Column indicating detection status (boolean or 0/1).
    first_detect_value : {True, False, 1, 0}, default True
        Value in `detection_flag_col` that declares a detection.

    Returns
    -------
    DataFrame
        Columns: ['site_id','lineage','detected_date','ref_date','delay_days','detected'].

    Notes
    -----
    - If a site/lineage never crosses the threshold, 'detected_date' and 'delay_days' are NaN and 'detected' is False.
    - Negative delays indicate detection before the reference date (lead time).
    """
    det = detections.copy()
    if "date" not in det.columns:
        raise ValueError("detections must include column 'date'")
    det["date"] = pd.to_datetime(det["date"])
    det_flag = det[detection_flag_col].astype(int) == int(bool(first_detect_value))

    # First detection per site_id/lineage
    det = det.assign(_flag=det_flag)
    first = (
        det[det["_flag"]]
        .sort_values(["site_id", "lineage", "date"])
        .groupby(["site_id", "lineage"], as_index=False)
        .first()[["site_id", "lineage", "date"]]
        .rename(columns={"date": "detected_date"})
    )

    ref = references.copy()
    if "ref_date" not in ref.columns:
        raise ValueError("references must include column 'ref_date'")
    ref["ref_date"] = pd.to_datetime(ref["ref_date"])

    merged = pd.merge(ref, first, on=["site_id", "lineage"], how="left")
    merged["detected"] = merged["detected_date"].notna()
    merged["delay_days"] = (merged["detected_date"] - merged["ref_date"]).dt.days.astype("float64")
    return merged[["site_id", "lineage", "detected_date", "ref_date", "delay_days", "detected"]]


def _weighted_agg(
    x: np.ndarray,
    w: Optional[np.ndarray],
    agg: str = "mean",
) -> float:
    """Weighted aggregation with support for 'mean','median','quantile-XX','rmse','mae'."""
    x = np.asarray(x, dtype=float)
    if w is None:
        if agg == "mean":
            return float(np.nanmean(x))
        if agg == "median":
            return float(np.nanmedian(x))
        if agg.startswith("quantile-"):
            q = float(agg.split("-")[1])
            return float(np.nanquantile(x, q))
        if agg == "rmse":
            return float(np.sqrt(np.nanmean(x ** 2)))
        if agg == "mae":
            return float(np.nanmean(np.abs(x)))
        raise ValueError(f"Unsupported agg: {agg}")
    # weights
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    if agg == "mean":
        return float(np.nansum(w * x) / (np.nansum(w) + _EPS))
    if agg == "rmse":
        return float(np.sqrt(np.nansum(w * (x ** 2)) / (np.nansum(w) + _EPS)))
    if agg == "mae":
        return float(np.nansum(w * np.abs(x)) / (np.nansum(w) + _EPS))
    if agg == "median":
        # Weighted median via sorting
        idx = np.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]
        cw = np.cumsum(w_sorted)
        cutoff = 0.5 * (np.nansum(w) + _EPS)
        return float(x_sorted[np.searchsorted(cw, cutoff, side="left")])
    if agg.startswith("quantile-"):
        q = float(agg.split("-")[1])
        idx = np.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]
        cw = np.cumsum(w_sorted)
        cutoff = q * (np.nansum(w) + _EPS)
        return float(x_sorted[np.searchsorted(cw, cutoff, side="left")])
    raise ValueError(f"Unsupported agg: {agg}")


def fairness_stratify(
    df: pd.DataFrame,
    metric_col: str,
    strata_cols: Sequence[str],
    weight_col: Optional[str] = None,
    agg: str = "mean",
    ci_level: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """
    Stratify a metric by coverage/site size/overlap bins and compute fairness disparities.

    Parameters
    ----------
    df : DataFrame
        Input data with at least `metric_col` and `strata_cols`.
    metric_col : str
        Column name of the metric to aggregate (e.g., RMSE).
    strata_cols : sequence of str
        Columns indicating categorical strata (pre-binned).
    weight_col : str, optional
        Column with observation weights for weighted aggregation.
    agg : {"mean","rmse","mae","median","quantile-<p>"}, default "mean"
        Aggregation per stratum.
    ci_level : float, default 0.95
        Confidence level for normal approximation intervals on the mean (unweighted).

    Returns
    -------
    dict
        Mapping:
        - 'overall': overall summary across all data
        - 'by_<col>': per-stratum summaries for each column in `strata_cols`
        - 'by_all': multi-way stratification (if len(strata_cols) > 1)

    Notes
    -----
    - Disparities are reported as absolute gap (max - min) and ratio (max / (min + eps)).
    - If `weight_col` provided, 'mean' aggregates are weighted; intervals are not adjusted for weights.
    """
    if metric_col not in df.columns:
        raise ValueError(f"{metric_col} not in DataFrame")
    for c in strata_cols:
        if c not in df.columns:
            raise ValueError(f"{c} not in DataFrame")

    x = df[metric_col].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float) if weight_col is not None else None

    overall_val = _weighted_agg(x, w, agg=agg)
    overall = pd.DataFrame({"metric": [metric_col], "agg": [agg], "value": [overall_val]})

    out: Dict[str, pd.DataFrame] = {"overall": overall}

    def summarize(group: pd.DataFrame) -> Tuple[float, int, float, float]:
        vals = group[metric_col].to_numpy(dtype=float)
        ww = group[weight_col].to_numpy(dtype=float) if weight_col is not None else None
        val = _weighted_agg(vals, ww, agg=agg)
        n = group.shape[0]
        mean_unw = float(np.nanmean(vals)) if n > 0 else np.nan
        std_unw = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
        sem_unw = std_unw / math.sqrt(max(n, 1))
        z = abs(norm.ppf(0.5 * (1 + ci_level)))
        lwr = mean_unw - z * sem_unw
        upr = mean_unw + z * sem_unw
        return val, n, lwr, upr

    # Per single stratum
    for c in strata_cols:
        frames = []
        for level, g in df.groupby(c, dropna=False, sort=True):
            val, n, lwr, upr = summarize(g)
            frames.append(
                {
                    c: level,
                    "value": val,
                    "n": n,
                    "ci_lower": lwr,
                    "ci_upper": upr,
                }
            )
        tab = pd.DataFrame(frames).sort_values(c).reset_index(drop=True)
        # Disparities
        v = tab["value"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            gap = float(np.nanmax(v) - np.nanmin(v))
            ratio = float(np.nanmax(v) / (np.nanmin(v) + _EPS))
        else:
            gap, ratio = np.nan, np.nan
        tab.attrs["disparity_abs"] = gap
        tab.attrs["disparity_ratio"] = ratio
        out[f"by_{c}"] = tab

    # Multi-way stratification
    if len(strata_cols) > 1:
        frames = []
        for levels, g in df.groupby(list(strata_cols), dropna=False, sort=True):
            val, n, lwr, upr = summarize(g)
            row = {col: lev for col, lev in zip(strata_cols, levels if isinstance(levels, tuple) else (levels,))}
            row.update({"value": val, "n": n, "ci_lower": lwr, "ci_upper": upr})
            frames.append(row)
        tab_all = pd.DataFrame(frames).sort_values(list(strata_cols)).reset_index(drop=True)
        out["by_all"] = tab_all

    return out

# --- QQ-plot helper -----------------------------------------------------------
from typing import Tuple

def qq_plot_points(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical vs empirical quantiles for a Normal QQ plot.

    Parameters
    ----------
    x : array-like
        1D sample (e.g., residuals). NaNs/inf are removed.

    Returns
    -------
    q_theory : np.ndarray
        Theoretical Normal quantiles (Blom plotting positions).
    q_emp : np.ndarray
        Sorted empirical quantiles from the data.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([]), np.array([])

    # Blom plotting positions (≈unbiased for Normal)
    p = (np.arange(1, n + 1, dtype=float) - 0.375) / (n + 0.25)
    q_theory = norm.ppf(p)
    q_emp = np.sort(x)
    return q_theory, q_emp
